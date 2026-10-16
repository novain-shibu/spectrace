# Exact mechanism check (cheap): record the p_i and q_i that HF's _speculative_sampling ACTUALLY uses for every drafted token,
# then recompute them the way the predictor does (teacher-forced softmax(logits / T) on the same prefix). If they match, the
# predictor is unbiased by construction and any outcome gap is noise. Also computes HF's own Rao-Blackwellized expected acceptance.
# Setting = the one that looked biased: real base policy Qwen2.5-1.5B (T=1), drafters Qwen2.5-0.5B (over-predicted) and -Instruct (under).
import torch, json, numpy as np
import transformers.generation.utils as gu
from transformers import AutoTokenizer, AutoModelForCausalLM
T, K, L = 1.0, 4, 96
TG = "Qwen/Qwen2.5-1.5B"; DRAFTS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"]
QS = ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?",
      "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
      "Write a short Python function that returns the n-th Fibonacci number.",
      "Explain why the sky is blue in two sentences."]
REC = []
_orig = gu._speculative_sampling


def wrapped(candidate_input_ids, candidate_logits, candidate_length, new_logits, is_done_candidate, *a, **kw):
    ids = candidate_input_ids[:, -candidate_length:]
    q = candidate_logits.float().softmax(-1)[0, torch.arange(candidate_length), ids[0]]
    p = new_logits.float().softmax(-1)[0, torch.arange(candidate_length), ids[0]]
    out = _orig(candidate_input_ids, candidate_logits, candidate_length, new_logits, is_done_candidate, *a, **kw)
    REC.append({"ids": candidate_input_ids[0].tolist(), "clen": int(candidate_length), "p": p.tolist(), "q": q.tolist(), "n": int(out[1])})
    return out


gu._speculative_sampling = wrapped


def neutral(m):
    g = m.generation_config
    g.top_k, g.top_p, g.repetition_penalty, g.temperature, g.do_sample = 0, 1.0, 1.0, T, True


tok = AutoTokenizer.from_pretrained(TG)
tgt = AutoModelForCausalLM.from_pretrained(TG, dtype=torch.float16).cuda().eval(); neutral(tgt)
print("target gen cfg", {k: v for k, v in tgt.generation_config.to_dict().items() if v not in (None, False)}, flush=True)
GEN = dict(do_sample=True, temperature=T, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=L, min_new_tokens=L)
for dn in DRAFTS:
    drf = AutoModelForCausalLM.from_pretrained(dn, dtype=torch.float16).cuda().eval(); neutral(drf)
    drf.generation_config.num_assistant_tokens = K; drf.generation_config.num_assistant_tokens_schedule = "constant"
    drf.generation_config.assistant_confidence_threshold = 0.0
    print(dn, "draft gen cfg", {k: v for k, v in drf.generation_config.to_dict().items() if v not in (None, False)}, flush=True)
    REC.clear()
    with torch.no_grad():
        for i, q in enumerate(QS):
            ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].cuda()
            for r in range(3):
                torch.manual_seed(100 * i + r); tgt.generate(ids, assistant_model=drf, **GEN)
        # recompute p, q the predictor's way on exactly the same prefixes + drafted tokens
        dq, dp, ratio_ok, eHF, eOurs, nReal = [], [], 0, 0.0, 0.0, 0
        for rec in REC:
            x = torch.tensor([rec["ids"]]).cuda(); c = rec["clen"]; d = x[0, -c:]
            lq = drf(x[:, :-1]).logits[0, -c:].float() / T; lp = tgt(x[:, :-1]).logits[0, -c:].float() / T
            q2 = lq.softmax(-1)[torch.arange(c), d]; p2 = lp.softmax(-1)[torch.arange(c), d]
            qh, ph = torch.tensor(rec["q"]).cuda(), torch.tensor(rec["p"]).cuda()
            dq.append((q2 / qh).log().abs().max().item()); dp.append((p2 / ph).log().abs().max().item())
            def expn(pp, qq):
                a = torch.clamp(pp / qq, max=1.0).cpu().numpy(); s, pre = 0.0, 1.0
                for v in a: pre *= v; s += pre
                return s
            eHF += expn(ph, qh); eOurs += expn(p2, q2); nReal += rec["n"]
        print("RESULT", json.dumps({"drafter": dn, "verify_steps": len(REC),
              "max_abs_log_ratio_q_ours_vs_HF": float(np.max(dq)), "median_q": float(np.median(dq)),
              "max_abs_log_ratio_p_ours_vs_HF": float(np.max(dp)), "median_p": float(np.median(dp)),
              "expected_accepted_HF_pq": eHF, "expected_accepted_our_pq": eOurs, "realized_accepted": nReal}, default=float), flush=True)
    del drf; torch.cuda.empty_cache()

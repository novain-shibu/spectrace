# RL drafter-selection spark: from ordinary T=1 rollouts (what RL already generates, with chosen-token logprobs), predict each
# candidate drafter's expected target passes WITHOUT deploying it, and check against real speculative sampling (HF assisted).
# Two "policies": Qwen2.5-1.5B-Instruct (before) and Qwen2.5-Math-1.5B-Instruct (stand-in for a policy after math RL).
# Three candidate drafters (same tokenizer): Qwen2.5-0.5B-Instruct, Qwen2.5-0.5B (base), Qwen2.5-Coder-0.5B-Instruct.
# Kill (pre-registered): predicted ranking differs from measured ranking, or any |pred - real| / real > 5%.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"])
import torch, json, time, random, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
T, K, L, NP, REP = 1.0, 4, 128, 24, 2
TARGETS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct"]
DRAFTS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-Coder-0.5B-Instruct"]
GEN = dict(do_sample=True, temperature=T, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=L, min_new_tokens=L)
random.seed(0)
ds = load_dataset("openai/gsm8k", "main", split="test")
qs = [ds[i]["question"] for i in random.sample(range(len(ds)), NP)]


def neutral(m):
    g = m.generation_config
    g.top_k, g.top_p, g.repetition_penalty, g.temperature, g.do_sample = 0, 1.0, 1.0, T, True


def dp(a, K):
    n = len(a); E = np.zeros(n + K + 2)
    for s in range(n - 1, -1, -1):
        tot, pre, done = 1.0, 1.0, False
        for m in range(K):
            if s + m >= n: done = True; break
            tot += pre * (1 - a[s + m]) * E[s + m + 1]; pre *= a[s + m]
        if not done: tot += pre * E[s + K + 1]
        E[s] = tot
    return E[0]


res = {}
for tg in TARGETS:
    tok = AutoTokenizer.from_pretrained(tg)
    tgt = AutoModelForCausalLM.from_pretrained(tg, dtype=torch.float16).cuda().eval(); neutral(tgt)
    calls = [0]; h = tgt.register_forward_hook(lambda m, i, o: calls.__setitem__(0, calls[0] + 1))
    ids_list = [tok.apply_chat_template([{"role": "user", "content": q + "\nSolve step by step."}], add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True)["input_ids"].cuda() for q in qs]
    # 1) rollouts once (no drafter), with chosen-token target probs: this is all an RL step already has
    rolls = []
    with torch.no_grad():
        for i, ids in enumerate(ids_list):
            for r in range(REP):
                torch.manual_seed(1000 * i + r); g = tgt.generate(ids, **GEN)
                pt = torch.softmax(tgt(g[:, :-1]).logits[0, ids.shape[1] - 1:].float() / T, -1).gather(1, g[0, ids.shape[1]:, None])[:, 0]
                rolls.append((ids, g, pt))
    for dn in DRAFTS:
        drf = AutoModelForCausalLM.from_pretrained(dn, dtype=torch.float16).cuda().eval(); neutral(drf)
        drf.generation_config.num_assistant_tokens = K; drf.generation_config.num_assistant_tokens_schedule = "constant"
        drf.generation_config.assistant_confidence_threshold = 0.0
        t0 = time.time(); est = []
        with torch.no_grad():
            for ids, g, pt in rolls:   # 2) prediction: one teacher-forced draft pass per rollout
                pd = torch.softmax(drf(g[:, :-1]).logits[0, ids.shape[1] - 1:].float() / T, -1).gather(1, g[0, ids.shape[1]:, None])[:, 0]
                est.append(dp(torch.clamp(pd / pt, max=1.0).cpu().numpy(), K))
        t_pred = time.time() - t0; t0 = time.time(); real = []
        with torch.no_grad():
            for i, ids in enumerate(ids_list):   # 3) ground truth: real speculative sampling with this drafter
                for r in range(REP):
                    torch.manual_seed(50000 + 1000 * i + r); calls[0] = 0
                    tgt.generate(ids, assistant_model=drf, **GEN); real.append(calls[0])
        t_real = time.time() - t0
        key = f"{tg.split('/')[1]} <- {dn.split('/')[1]}"
        res[key] = {"pred": float(np.mean(est)), "pred_se": float(np.std(est) / len(est) ** .5), "real": float(np.mean(real)),
                    "real_se": float(np.std(real) / len(real) ** .5), "t_pred_s": t_pred, "t_real_s": t_real}
        print("ROW", key, json.dumps(res[key]), flush=True)
        del drf; torch.cuda.empty_cache()
    h.remove(); del tgt; torch.cuda.empty_cache()
print("RESULT", json.dumps(res))

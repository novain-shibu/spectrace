# RL drift demo: real RL pair Qwen2.5-1.5B (base, step 0) -> hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo (after RL, Apache-2.0).
# Intermediate policies = linear weight interpolation (PROXY, labeled as such). For each policy: T=1 rollouts with chosen-token
# probs (what RL has), predicted passes per candidate drafter from one teacher-forced draft pass. Real speculative sampling only
# at the two REAL endpoints (validation). Question: does the best drafter / its speed-up change with RL, and does the predictor track it?
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"])
import torch, json, time, random, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
T, K, L, NP, REP = 1.0, 4, 128, 20, 2
BASE, RL = "Qwen/Qwen2.5-1.5B", "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo"
DRAFTS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-Coder-0.5B-Instruct"]
ALPHAS = [0.0, 0.33, 0.67, 1.0]
GEN = dict(do_sample=True, temperature=T, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=L, min_new_tokens=L)
random.seed(0)
ds = load_dataset("openai/gsm8k", "main", split="test")
qs = [ds[i]["question"] for i in random.sample(range(len(ds)), NP)]
tok = AutoTokenizer.from_pretrained(RL)
# SimpleRL-Zoo prompt style (chat template + boxed answer instruction)
ids_list = [tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                                    add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].cuda() for q in qs]


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


sd_base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).state_dict()
sd_rl = AutoModelForCausalLM.from_pretrained(RL, dtype=torch.float32).state_dict()
drafts = {}
for dn in DRAFTS:
    d = AutoModelForCausalLM.from_pretrained(dn, dtype=torch.float16).cuda().eval(); neutral(d)
    d.generation_config.num_assistant_tokens = K; d.generation_config.num_assistant_tokens_schedule = "constant"
    d.generation_config.assistant_confidence_threshold = 0.0
    drafts[dn] = d
res = {}
tgt = AutoModelForCausalLM.from_pretrained(RL, dtype=torch.float16).cuda().eval(); neutral(tgt)
calls = [0]; tgt.register_forward_hook(lambda m, i, o: calls.__setitem__(0, calls[0] + 1))
for a in ALPHAS:
    with torch.no_grad():
        mix = {k: ((1 - a) * sd_base[k] + a * sd_rl[k]).half() for k in sd_rl if k in sd_base}
        tgt.load_state_dict(mix, strict=False)
    rolls = []; t0 = time.time()
    with torch.no_grad():
        for i, ids in enumerate(ids_list):
            for r in range(REP):
                torch.manual_seed(1000 * i + r); g = tgt.generate(ids, **GEN)
                pt = torch.softmax(tgt(g[:, :-1]).logits[0, ids.shape[1] - 1:].float() / T, -1).gather(1, g[0, ids.shape[1]:, None])[:, 0]
                rolls.append((ids, g, pt))
    t_roll = time.time() - t0
    for dn, drf in drafts.items():
        t0 = time.time(); est = []
        with torch.no_grad():
            for ids, g, pt in rolls:
                pd = torch.softmax(drf(g[:, :-1]).logits[0, ids.shape[1] - 1:].float() / T, -1).gather(1, g[0, ids.shape[1]:, None])[:, 0]
                est.append(dp(torch.clamp(pd / pt, max=1.0).cpu().numpy(), K))
        row = {"alpha": a, "pred": float(np.mean(est)), "pred_se": float(np.std(est) / len(est) ** .5), "t_pred_s": time.time() - t0,
               "t_rollouts_s": t_roll}
        if a in (0.0, 1.0):   # validate at the two REAL policies only
            t0 = time.time(); real = []
            with torch.no_grad():
                for i, ids in enumerate(ids_list):
                    for r in range(REP):
                        torch.manual_seed(50000 + 1000 * i + r); calls[0] = 0
                        tgt.generate(ids, assistant_model=drf, **GEN); real.append(calls[0])
            row.update(real=float(np.mean(real)), real_se=float(np.std(real) / len(real) ** .5), t_real_s=time.time() - t0)
        res[f"{a}|{dn.split('/')[1]}"] = row
        print("ROW", a, dn.split("/")[1], json.dumps(row), flush=True)
print("RESULT", json.dumps(res))

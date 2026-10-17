# SpecTrace real_sampling2 (corrects real_sampling1). Fixes from Zhao & Cai review 2026-10-17:
# (1) HF's first target forward already verifies the first draft block (no separate prefill call), so compare E[0] with the
#     FULL calls[0] count (real_sampling1 used E[1] vs calls[0] - 1, an off-by-one mismatch);
# (2) apply the same EOS mask as HF's MinNewTokensLogitsProcessor (min_new_tokens = L) to BOTH p and q (diag-4: <|endoftext|>);
# (3) 2x the runs (REP 8, 240 pairs) and save per-run acceptance vectors so any later convention can be recomputed offline.
# Temperature only: top_k=0, top_p=1, no penalty.
import torch, time, json, numpy as np, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
print("transformers", transformers.__version__, torch.cuda.get_device_name(0))
TG, DR = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(TG)
tgt = AutoModelForCausalLM.from_pretrained(TG, dtype=torch.float16).cuda().eval()
drf = AutoModelForCausalLM.from_pretrained(DR, dtype=torch.float16).cuda().eval()
calls = [0]; tgt.register_forward_hook(lambda m, i, o: calls.__setitem__(0, calls[0] + 1))
P = ["Write a Python function that checks whether a string is a palindrome.", "Explain what a hash table is in three sentences.",
 "Convert this to TypeScript: def add(a, b): return a + b", "List five capital cities in Europe and one fact about each.",
 "What is 17 * 23? Show your steps.", "Write a SQL query that returns the ten most recent orders per customer.",
 "Summarize the plot of Romeo and Juliet in one paragraph.", "Write a bash script that counts lines in all .py files.",
 "Fix the bug: for i in range(len(xs)): if xs[i] = 0: print(i)", "Give three tips for writing clear commit messages.",
 "Translate to French: The weather is nice today and we are going to the park.", "Write a JSON object describing a book with title, author, year, tags.",
 "Explain the difference between TCP and UDP.", "Write a Rust function that reverses a vector in place.",
 "What causes the seasons on Earth?", "Write a regular expression that matches an email address and explain it.",
 "Refactor: x = []\nfor i in range(10):\n    x.append(i*i)", "Describe binary search and its time complexity.",
 "Write a haiku about debugging.", "Write a Go HTTP handler that returns hello world as JSON.",
 "If a train travels 120 km in 1.5 hours, what is its average speed?", "Explain what a Python decorator is with an example.",
 "Write a markdown table of three programming languages and their creators.", "What is the difference between a list and a tuple in Python?",
 "Write a C function that computes the factorial of n iteratively.", "Explain gradient descent to a high school student.",
 "Write unit tests with pytest for a function add(a, b).", "Give an example of a YAML config for a web server with port and host.",
 "Explain what Docker containers are.", "Write a JavaScript function that debounces another function."]
L, T, K, REP = 96, 0.8, 4, 8
EOS_MASK = [151643]   # <|endoftext|>; diag-4: what HF masks under min_new_tokens
print("target eos", tgt.generation_config.eos_token_id, "draft eos", drf.generation_config.eos_token_id, flush=True)
GEN = dict(do_sample=True, temperature=T, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=L, min_new_tokens=L)
for m in (tgt, drf):
    m.generation_config.top_k = 0; m.generation_config.top_p = 1.0; m.generation_config.repetition_penalty = 1.0; m.generation_config.temperature = T
drf.generation_config.num_assistant_tokens = K; drf.generation_config.num_assistant_tokens_schedule = "constant"
drf.generation_config.assistant_confidence_threshold = 0.0
def dp(a, K):
    n = len(a); E = np.zeros(n + K + 2)
    for s in range(n - 1, -1, -1):         # first target call verifies the first draft block (no separate prefill)
        tot, pre, done = 1.0, 1.0, False
        for m in range(K):
            if s + m >= n: done = True; break
            tot += pre * (1 - a[s + m]) * E[s + m + 1]; pre *= a[s + m]
        if not done: tot += pre * E[s + K + 1]
        E[s] = tot
    return E[0]
real, est, est_raw, avecs = [], [], [], []
for i, p in enumerate(P):
    ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].cuda()
    n0 = ids.shape[1]
    with torch.no_grad():
        for r in range(REP):
            torch.manual_seed(1000 * i + r)
            g = tgt.generate(ids, **GEN); y = g[0, n0:]
            lt = tgt(g[:, :-1]).logits[0, n0 - 1:].float() / T; ld = drf(g[:, :-1]).logits[0, n0 - 1:].float() / T
            a_raw = torch.clamp(ld.softmax(-1).gather(1, y[:, None]) / lt.softmax(-1).gather(1, y[:, None]), max=1.0)[:, 0].cpu().numpy()
            lt[:, EOS_MASK] = -float("inf"); ld[:, EOS_MASK] = -float("inf")
            a = torch.clamp(ld.softmax(-1).gather(1, y[:, None]) / lt.softmax(-1).gather(1, y[:, None]), max=1.0)[:, 0].cpu().numpy()
            est.append(dp(a, K)); est_raw.append(dp(a_raw, K)); avecs.append([round(float(v), 6) for v in a])
            torch.manual_seed(5000 + 1000 * i + r); calls[0] = 0
            o = tgt.generate(ids, assistant_model=drf, **GEN); real.append(calls[0])
    print(i, np.mean(real[-REP:]), np.mean(est[-REP:]), flush=True)
real, est, est_raw = np.array(real, float), np.array(est), np.array(est_raw)
json.dump({"real": real.tolist(), "est": est.tolist(), "est_raw": est_raw.tolist(), "a": avecs}, open("per_run.json", "w"))
print("RESULT", json.dumps({"n": int(len(real)), "real_mean": float(real.mean()), "real_se": float(real.std() / len(real) ** .5),
      "est_mean": float(est.mean()), "est_se": float(est.std() / len(est) ** .5), "est_raw_mean": float(est_raw.mean()),
      "rel_diff": float((est.mean() - real.mean()) / real.mean()), "T": T, "K": K, "L": L, "REP": REP}), flush=True)

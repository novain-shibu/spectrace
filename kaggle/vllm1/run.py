# Conformance spark 1: does vLLM's MEASURED speculative acceptance equal the answer key on a T4 (fp16, the neglected path)?
# ngram drafter is deterministic given the tokens, so replaying the same proposer on vLLM's own outputs must reproduce
# vLLM's drafted/accepted counters EXACTLY (greedy). Axes: per-request (batch 1) vs all 30 at once (batch 30).
import os, subprocess, sys, json, time
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

def main():
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], capture_output=True, text=True); print("pip rc", r.returncode)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"])
    import torch, vllm; print("vllm", vllm.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)
    from vllm import LLM, SamplingParams
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
    K, NMIN, NMAX, L = 4, 2, 5, 128
    llm = LLM("Qwen/Qwen2.5-1.5B-Instruct", dtype="float16", max_model_len=2048, gpu_memory_utilization=0.85, disable_log_stats=False,
              speculative_config={"method": "ngram", "num_speculative_tokens": K, "prompt_lookup_min": NMIN, "prompt_lookup_max": NMAX})
    tok = llm.get_tokenizer(); sp = SamplingParams(temperature=0, max_tokens=L, repetition_penalty=1.0)
    def counters():
        out = {}
        for m in llm.get_metrics():
            if "spec_decode" in m.name:
                v = getattr(m, "value", None); vals = getattr(m, "values", None)
                out[m.name] = v if v is not None else list(vals) if vals is not None else None
        return out
    try:
        from vllm.v1.spec_decode.ngram_proposer import _find_longest_matched_ngram_and_propose_tokens as vprop; print("using vLLM's own proposer fn")
    except Exception as e:
        vprop = None; print("own proposer fn not importable:", repr(e)[:200])
    def propose(ctx):
        import numpy as np
        if vprop is not None:
            try:
                d = vprop(np.array(ctx, dtype=np.int32), NMIN, NMAX, 2048, K); return [] if d is None else list(map(int, d))
            except TypeError:
                pass
        for n in range(NMAX, NMIN - 1, -1):          # fallback: longest suffix n-gram, earliest earlier occurrence
            if len(ctx) <= n: continue
            suf = ctx[-n:]
            for i in range(len(ctx) - n):
                if ctx[i:i + n] == suf: return ctx[i + n:i + n + K]
        return []
    def replay(prompt, y):
        pos, drafts, dtoks, acc = 1, 0, 0, 0
        while pos < len(y):
            d = propose(prompt + y[:pos]); d = d[:max(0, len(y) - pos)]   # vLLM caps drafts at remaining budget (assumed)
            n = 0
            while n < len(d) and d[n] == y[pos + n]: n += 1
            if d: drafts += 1; dtoks += len(d); acc += n
            pos += n + 1
        return drafts, dtoks, acc
    msgs = [[{"role": "user", "content": p}] for p in P]
    res = {"per_request": [], "batch": None}
    for i, m in enumerate(msgs):
        c0 = counters(); o = llm.chat(m, sp, use_tqdm=False)[0]; c1 = counters()
        prompt = list(o.prompt_token_ids); y = list(o.outputs[0].token_ids)
        dv = {k: (c1[k] - c0.get(k, 0)) if isinstance(c1[k], (int, float)) else None for k in c1}
        res["per_request"].append({"i": i, "vllm": dv, "replay": replay(prompt, y), "len": len(y), "y": y}); print(i, res["per_request"][-1], flush=True)
    c0 = counters(); outs = llm.chat(msgs, sp, use_tqdm=False); c1 = counters()
    rep = [replay(list(o.prompt_token_ids), list(o.outputs[0].token_ids)) for o in outs]
    res["batch"] = {"vllm": {k: (c1[k] - c0.get(k, 0)) if isinstance(c1[k], (int, float)) else None for k in c1},
                    "replay_sum": [sum(x[j] for x in rep) for j in range(3)],
                    "same_outputs_as_batch1": sum(list(o.outputs[0].token_ids) == r["y"] for o, r in zip(outs, res["per_request"]))}
    print("RESULT", json.dumps(res["batch"])); print("counter names", list(c1))
    json.dump(res, open("vllm1.json", "w"))


if __name__ == '__main__':
    main()

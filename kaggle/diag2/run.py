# SpecTrace diagnostic: why do HF assisted-generation pass counts differ from the teacher-forced simulator?
# Logs every target call's input (= the drafts HF proposed), in fp32 and fp16, and compares each draft with the
# teacher-forced draft prediction at the same position. Sim counted two ways: first token from prefill (pos 1) or drafted (pos 0).
import torch, json, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
TG, DR = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(TG)
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
P = P[:30]; L = 128; K = 4
def sim(y, d, k, start):
    pos, passes = start, 0
    while pos < len(y):
        n = 0
        while n < k and pos + n < len(y) and d[pos + n] == y[pos + n]: n += 1
        pos += n + 1; passes += 1
    return passes
report = {}
for dt in (torch.float32, torch.float16):
    tgt = AutoModelForCausalLM.from_pretrained(TG, dtype=dt).cuda().eval(); drf = AutoModelForCausalLM.from_pretrained(DR, dtype=dt).cuda().eval()
    log = []; tgt.register_forward_hook(lambda m, i, o: None)
    h = tgt.register_forward_pre_hook(lambda m, a, kw: log.append((kw.get("input_ids") if kw.get("input_ids") is not None else a[0]).shape[1]) , with_kwargs=True)
    dlog = []; drf.register_forward_pre_hook(lambda m, a, kw: dlog.append((kw.get("input_ids") if kw.get("input_ids") is not None else a[0]).shape[1]), with_kwargs=True)
    for mm in (tgt, drf):
        mm.generation_config.repetition_penalty = 1.0; mm.generation_config.do_sample = False
        mm.generation_config.temperature = None; mm.generation_config.top_p = None; mm.generation_config.top_k = None
    print('gen cfg target', tgt.generation_config.to_dict())
    for k in ("num_assistant_tokens",): setattr(drf.generation_config, k, K)
    drf.generation_config.num_assistant_tokens_schedule = "constant"; drf.generation_config.assistant_confidence_threshold = 0.0
    rows = []
    for i, p in enumerate(P):
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].cuda(); n0 = ids.shape[1]
        with torch.no_grad():
            log.clear(); g = tgt.generate(ids, max_new_tokens=L, do_sample=False, temperature=None, top_p=None, top_k=None); y = g[0, n0:].tolist()
            d = drf(torch.cat([ids, g[:, n0:-1]], 1)).logits[0, n0 - 1:].argmax(-1).tolist()
            log.clear(); dlog.clear()
            a = tgt.generate(ids, assistant_model=drf, max_new_tokens=L, do_sample=False, temperature=None, top_p=None, top_k=None); ya = a[0, n0:].tolist()
        calls = list(log)
        rows.append({"i": i, "identical": ya == y, "target_calls": len(calls), "call_lens": calls[:12], "draft_call_lens": dlog[:16],
                     "sim_pos0": sim(y, d, K, 0), "sim_pos1": sim(y, d, K, 1), "len": len(y),
                     "first_div": next((j for j, (u, v) in enumerate(zip(ya, y)) if u != v), None)})
        print(dt, rows[-1], flush=True)
    report[str(dt)] = rows; del tgt, drf; torch.cuda.empty_cache()
for dt, rows in report.items():
    print("SUMMARY", dt, "calls", sum(r["target_calls"] for r in rows), "sim_pos0", sum(r["sim_pos0"] for r in rows), "sim_pos1", sum(r["sim_pos1"] for r in rows), "identical", sum(r["identical"] for r in rows), "/", len(rows))

# SpecTrace conformance matrix 1 (vLLM, T4). ONE install (offline wheelhouse), many configs, each in its own subprocess.
# Oracle: for a deterministic drafter (ngram), a draft token is accepted iff the token vLLM actually emitted equals it -
# under greedy AND sampling (rejection sampling with a one-hot draft). So replaying vLLM's proposer on vLLM's own outputs
# must reproduce drafts / draft tokens / accepted EXACTLY, for every sampling setting. Any difference = engine bug.
import os, sys, json, subprocess, time, glob
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
#__PROMPTS__
import inspect, textwrap as _t
LONG = [
    "Add type hints to every function in this code and return the full file:\n" + inspect.getsource(_t)[:6000],
    "Return this JSON with every key in UPPERCASE, keep values unchanged:\n"
    + json.dumps({f"key_{i}": {"name": f"item {i}", "tags": ["a", "b", str(i)], "n": i} for i in range(60)}),
    "Repeat the following text exactly, then add one sentence at the end:\n"
    + " ".join(["The quick brown fox jumps over the lazy dog number %d." % i for i in range(120)]),
]
CONFIGS = [
    dict(name="A_greedy_b1"),
    dict(name="B_greedy_batch", batch=True),
    dict(name="C_greedy_noprefixcache", engine=dict(enable_prefix_caching=False)),
    dict(name="D_greedy_long_chunked", long=True, engine=dict(max_num_batched_tokens=512, max_model_len=4096)),
    dict(name="E_greedy_k8", k=8),
    dict(name="F_greedy_k2_n1_3", k=2, nmin=1, nmax=3),
    dict(name="G_sample_T08_b1", sp=dict(temperature=0.8, seed=1)),
    dict(name="H_sample_T1_batch", batch=True, sp=dict(temperature=1.0, seed=2)),
    dict(name="I_sample_topp_topk", sp=dict(temperature=0.8, top_p=0.9, top_k=20, seed=3)),
    dict(name="J_sample_reppen", sp=dict(temperature=0.8, repetition_penalty=1.1, seed=4)),
    dict(name="K_greedy_reppen", sp=dict(repetition_penalty=1.1)),
    dict(name="L_greedy_fp32", engine=dict(dtype="float32")),
    dict(name="M_greedy_batch_maxseqs4", batch=True, engine=dict(max_num_seqs=4)),
    dict(name="N_greedy_stop", sp=dict(stop=["\n\n"])),
    dict(name="O_sample_freqpen", sp=dict(temperature=0.8, frequency_penalty=0.5, presence_penalty=0.5, seed=5)),
    dict(name="P_greedy_logprobs", sp=dict(logprobs=5)),
    dict(name="Q_long_batch_sample", long=True, batch=True, sp=dict(temperature=0.7, seed=6), engine=dict(max_model_len=4096)),
]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def setup():
    t = time.time()
    wh = glob.glob("/kaggle/input/**/wheels", recursive=True)
    if wh:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wh[0], "vllm"],
                           capture_output=True, text=True)
        print("offline install rc", r.returncode, r.stderr[-300:], flush=True)
        if r.returncode:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"])
    else:
        print("no wheelhouse found, online install", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"])
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"])
    md = glob.glob("/kaggle/input/**/models/Qwen2.5-1.5B-Instruct", recursive=True)
    print("setup %.0f s, local model: %s" % (time.time() - t, md), flush=True)
    return md[0] if md else MODEL


def propose(t, K, NMIN, NMAX, MML):
    """Port of vLLM's _find_longest_matched_ngram_and_propose_tokens (Apache-2.0); verified 30/30 vs vLLM 0.31."""
    n = len(t)
    if n < NMIN:
        return []
    k = min(K, MML - n)
    if k <= 0:
        return []
    r = t[::-1]
    lps = [0] * NMAX
    longest = position = prev = 0
    i = 1
    while i < n:
        if r[prev] == r[i]:
            prev += 1
            if prev >= longest:
                longest, position = prev, i
            if i < NMAX:
                lps[i] = prev
            if prev == NMAX:
                prev = lps[NMAX - 1]
            i += 1
        elif prev != 0:
            prev = lps[prev - 1]
        else:
            i += 1
    if longest < NMIN:
        return []
    start = n - 1 - position + longest
    k = min(k, n - start)
    return t[start:start + k]


def replay(prompt, y, K, NMIN, NMAX, MML):
    pos = dr = dt = ac = 0
    pos = 1
    while pos < len(y):
        d = propose(prompt + y[:pos], K, NMIN, NMAX, MML)
        n = 0
        while n < len(d) and pos + n < len(y) and d[n] == y[pos + n]:
            n += 1
        if d:
            dr += 1; dt += len(d); ac += n
        pos += n + 1
    return [dr, dt, ac]


def child(cfg, model_path):
    from vllm import LLM, SamplingParams
    K, NMIN, NMAX = cfg.get("k", 4), cfg.get("nmin", 2), cfg.get("nmax", 5)
    eng = dict(dtype="float16", max_model_len=2048, gpu_memory_utilization=0.85)
    eng.update(cfg.get("engine", {}))
    MML = eng["max_model_len"]
    llm = LLM(model_path, disable_log_stats=False, speculative_config={
        "method": "ngram", "num_speculative_tokens": K, "prompt_lookup_min": NMIN, "prompt_lookup_max": NMAX}, **eng)
    spk = dict(temperature=0, max_tokens=256 if cfg.get("long") else 128, repetition_penalty=1.0)
    spk.update(cfg.get("sp", {}))
    prompts = LONG if cfg.get("long") else P
    if cfg["name"] == "preflight":
        prompts = P[:2]
    msgs = [[{"role": "user", "content": p}] for p in prompts]

    def cnt():
        o = {}
        for m in llm.get_metrics():
            if "spec_decode" in m.name and getattr(m, "value", None) is not None:
                o[m.name.split(":")[-1]] = m.value
        return o

    def diff(c1, c0):
        return [c1.get(k, 0) - c0.get(k, 0) for k in
                ("spec_decode_num_drafts", "spec_decode_num_draft_tokens", "spec_decode_num_accepted_tokens")]

    rows = []
    if cfg.get("batch"):
        c0 = cnt(); outs = llm.chat(msgs, SamplingParams(**spk), use_tqdm=False); c1 = cnt()
        reps = [replay(list(o.prompt_token_ids), list(o.outputs[0].token_ids), K, NMIN, NMAX, MML) for o in outs]
        eng_tot = diff(c1, c0); rep_tot = [sum(r[j] for r in reps) for j in range(3)]
        rows = [{"prompt": list(o.prompt_token_ids), "y": list(o.outputs[0].token_ids), "replay": r,
                 "finish": o.outputs[0].finish_reason} for o, r in zip(outs, reps)]
        res = {"engine_total": eng_tot, "replay_total": rep_tot, "match": eng_tot == rep_tot}
    else:
        mism = 0
        for i, m in enumerate(msgs):
            s = dict(spk)
            if "seed" in s:
                s["seed"] = s["seed"] * 1000 + i
            c0 = cnt(); o = llm.chat(m, SamplingParams(**s), use_tqdm=False)[0]; c1 = cnt()
            e = diff(c1, c0); r = replay(list(o.prompt_token_ids), list(o.outputs[0].token_ids), K, NMIN, NMAX, MML)
            mism += e != r
            rows.append({"prompt": list(o.prompt_token_ids), "y": list(o.outputs[0].token_ids), "engine": e, "replay": r,
                         "finish": o.outputs[0].finish_reason})
        res = {"per_request_mismatch": mism, "n": len(msgs),
               "engine_total": [sum(x["engine"][j] for x in rows) for j in range(3)],
               "replay_total": [sum(x["replay"][j] for x in rows) for j in range(3)], "match": mism == 0}
    res["name"] = cfg["name"]
    json.dump({"cfg": cfg, "res": res, "rows": rows}, open(f"/kaggle/working/{cfg['name']}.json", "w"))
    print("ROW", json.dumps(res), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        child(json.loads(sys.argv[1]), sys.argv[2])
        sys.exit(0)
    t0 = time.time()
    mp = setup()
    pf = subprocess.run([sys.executable, __file__, json.dumps(dict(name="preflight", engine=dict(max_model_len=1024))), mp],
                        capture_output=True, text=True, timeout=900)
    print("preflight rc", pf.returncode, [l for l in pf.stdout.splitlines() if l.startswith("ROW")],
          pf.stderr[-1500:] if pf.returncode else "", flush=True)
    if pf.returncode:
        sys.exit(1)
    for cfg in CONFIGS:
        t = time.time()
        try:
            r = subprocess.run([sys.executable, __file__, json.dumps(cfg), mp], capture_output=True, text=True, timeout=1500)
            rows = [l for l in r.stdout.splitlines() if l.startswith("ROW")]
            print(cfg["name"], "rc", r.returncode, "%.0f s" % (time.time() - t), rows[-1] if rows else r.stderr[-800:], flush=True)
        except subprocess.TimeoutExpired:
            print(cfg["name"], "TIMEOUT", flush=True)
    print("TOTAL %.0f s" % (time.time() - t0))

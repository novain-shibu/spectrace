# Spark: is pass-exact simulation of DRAFT-MODEL speculative decoding possible from ONE teacher-forced draft pass?
# Real loop = KV-cache greedy chain speculation (draft proposes k, target verifies k+1 in one pass).
# Sim = target greedy tokens + draft argmax under teacher forcing (one full-sequence pass) -> pass count for every k.
import numpy as np, time, sys
rng = np.random.default_rng(0)
V, D, Lyr, Hh, T = 512, 64, 2, 4, 320
def make(seed, base=None, eps=0.0):
    r = np.random.default_rng(seed); s = 1 / np.sqrt(D)
    p = {"E": r.normal(0, 1, (V, D)), "P": r.normal(0, .3, (T, D)), "O": r.normal(0, s * 3, (D, V))}
    for l in range(Lyr):
        for n in ("q", "k", "v", "o", "f1", "f2"):
            sh = (D, 4 * D) if n == "f1" else (4 * D, D) if n == "f2" else (D, D)
            p[f"{n}{l}"] = r.normal(0, s, sh)
    if base is not None: p = {k: base[k] + eps * r.normal(0, 1, v.shape) * v.std() for k, v in p.items()}
    return {k: v.astype(np.float32) for k, v in p.items()}
def ln(x): return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
def fwd(p, toks, start, cache):  # toks at positions start.., cache: list per layer of (K,V) for positions < start
    x = p["E"][toks] + p["P"][start:start + len(toks)]
    n = len(toks); hd = D // Hh
    for l in range(Lyr):
        h = ln(x); q, k, v = h @ p[f"q{l}"], h @ p[f"k{l}"], h @ p[f"v{l}"]
        K = np.concatenate([cache[l][0][:start], k]); Vv = np.concatenate([cache[l][1][:start], v]); cache[l] = (K, Vv)
        q4, K4, V4 = q.reshape(n, Hh, hd), K.reshape(-1, Hh, hd), Vv.reshape(-1, Hh, hd)
        a = np.einsum("qhd,khd->hqk", q4, K4) / np.sqrt(hd)
        mask = np.arange(K.shape[0])[None, :] > (start + np.arange(n))[:, None]; a[:, mask] = -1e9
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        x = x + np.einsum("hqk,khd->qhd", a, V4).reshape(n, D) @ p[f"o{l}"]
        x = x + np.maximum(ln(x) @ p[f"f1{l}"], 0) @ p[f"f2{l}"]
    return ln(x) @ p["O"]
def empty(): return [(np.zeros((0, D), np.float32), np.zeros((0, D), np.float32)) for _ in range(Lyr)]
def greedy(p, prompt, L):
    c = empty(); lg = fwd(p, prompt, 0, c); out = [int(lg[-1].argmax())]
    while len(out) < L: out.append(int(fwd(p, [out[-1]], len(prompt) + len(out) - 1, c)[-1].argmax()))
    return out
def real_spec(tp, dp, prompt, L, k):
    tc, dc = empty(), empty(); P = len(prompt)
    t_last = fwd(tp, prompt, 0, tc)[-1]; fwd(dp, prompt, 0, dc)
    out = [int(t_last.argmax())]; passes = 0; dlen = P   # dc holds positions < dlen
    while len(out) < L:
        seq = list(prompt) + out; drafts = []; cur = len(seq)
        # draft: catch up on accepted tokens not yet in its cache, then propose k
        lg = fwd(dp, seq[dlen:cur], dlen, dc)[-1]; pos = cur
        for _ in range(k):
            t = int(lg.argmax()); drafts.append(t); lg = fwd(dp, [t], pos, dc)[-1]; pos += 1
        # target verifies last accepted token + k drafts in ONE pass
        vl = fwd(tp, [seq[-1]] + drafts, cur - 1, tc); passes += 1
        n = 0
        while n < k and int(vl[n].argmax()) == drafts[n]: n += 1
        out += drafts[:n] + [int(vl[n].argmax())]
        acc = P + len(out) - 1          # positions whose K/V are valid in target cache = seq up to last accepted-before-bonus
        tc = [(K[:acc], Vv[:acc]) for K, Vv in tc]; dlen = min(cur + n, acc); dc = [(K[:dlen], Vv[:dlen]) for K, Vv in dc]
    return out[:L], passes
def sim(y, d, k):
    L, pos, passes = len(y), 0, 0
    pos = 1  # first token comes from prefill
    while pos < L:
        n = 0
        while n < k and pos + n < L and d[pos + n] == y[pos + n]: n += 1
        pos += n + 1; passes += 1
    return passes
tp = make(1); NP, L = int(sys.argv[1]) if len(sys.argv) > 1 else 20, 200
res = []; t_real = t_sim = 0.0; mism = 0; ident = 0
for eps in (0.05, 0.15):
    dp = make(2, tp, eps)
    for i in range(NP):
        prompt = list(rng.integers(0, V, 16))
        y = greedy(tp, prompt, L)
        t0 = time.time(); dl = fwd(dp, prompt + y[:-1], 0, empty()); d = [None] + [int(a) for a in dl[len(prompt):].argmax(-1)]
        # d[j] = draft prediction of y[j] given prompt+y[:j]  (index j>=1)
        sims = {k: sim(y, d, k) for k in (1, 2, 4, 8)}; t_sim += time.time() - t0
        for k in (1, 2, 4, 8):
            t0 = time.time(); o, ps = real_spec(tp, dp, prompt, L, k); t_real += time.time() - t0
            ident += (o == y); mism += (ps != sims[k]); res.append((eps, k, ps, sims[k]))
tot = len(res)
print(f"runs {tot}: outputs identical to greedy {ident}/{tot}; pass-count mismatches sim vs real {mism}/{tot}")
for eps in (0.05, 0.15):
    for k in (1, 2, 4, 8):
        r = [x for x in res if x[0] == eps and x[1] == k]
        print(f"  eps={eps} k={k}: real passes {sum(x[2] for x in r)}  sim {sum(x[3] for x in r)}  (vanilla {len(r)*(L-1)})")
print(f"time: real spec loops {t_real:.1f} s vs sim (1 teacher-forced pass, all 4 k) {t_sim:.2f} s -> {t_real/t_sim:.0f}x")

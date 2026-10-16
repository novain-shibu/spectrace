# Spark 4: can speculative SAMPLING (temperature > 0) be scored from logs? Identity: P(draft accepts tokens x1..xm) =
# prod q(xi) * min(1, p(xi)/q(xi)) = prod min(p,q)(xi), so summing over sequences = E_{x~p}[ prod min(1, q(xi)/p(xi)) ].
# => from ordinary target samples (logs) + the chosen token's target prob + ONE teacher-forced draft pass, an UNBIASED
# estimate of P(accept >= m) for every m. Check vs Monte-Carlo of the real speculative sampling loop (Leviathan et al.).
import numpy as np, sys
sys.argv = [sys.argv[0]]
exec(open("spark.py").read().split("tp = make(1)")[0])   # reuse toy model + fwd
rng = np.random.default_rng(7); TEMP = 1.0
def probs(lg): z = lg / TEMP; z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
tp = make(1); dp = make(2, tp, 0.10)
K, L, NPR = 4, 120, 40
prompts = [list(rng.integers(0, V, 16)) for _ in range(NPR)]
def sample_target(prompt):  # a "production log": plain target sampling, records chosen-token prob
    c = empty(); lg = fwd(tp, prompt, 0, c)[-1]; out = []
    while len(out) < L:
        p = probs(lg); t = int(rng.choice(V, p=p)); out.append(t); lg = fwd(tp, [t], len(prompt) + len(out) - 1, c)[-1]
    return out
def estimator(prompt, y):  # needs target prob of chosen token + draft prob of chosen token (one teacher-forced pass each)
    pt = probs(fwd(tp, prompt + y[:-1], 0, empty())[len(prompt) - 1:]); pd = probs(fwd(dp, prompt + y[:-1], 0, empty())[len(prompt) - 1:])
    a = np.minimum(1.0, pd[np.arange(L), y] / pt[np.arange(L), y])
    # token z is "drafted+accepted" w.p. a(z), "corrected (pass boundary)" w.p. 1-a(z); bonus after K accepts has weight 1.
    E = np.zeros(L + K + 2)                       # E[s] = expected passes to finish from position s (s >= L -> 0)
    for s in range(L - 1, -1, -1):
        tot, pre = 1.0, 1.0
        for m in range(K):
            if s + m >= L: break
            tot += pre * (1 - a[s + m]) * E[s + m + 1]; pre *= a[s + m]
        else:
            tot += pre * E[s + K + 1]
        if s + K > L - 1 and s + m >= L: pass
        E[s] = tot
    return E[0]
def real_spec_sampling(prompt):  # real loop, full recompute for clarity
    seq = list(prompt); n_out = 0; passes = 0
    while n_out < L:
        drafts, qs = [], []
        for _ in range(K):
            q = probs(fwd(dp, seq + drafts, 0, empty())[-1]); x = int(rng.choice(V, p=q)); drafts.append(x); qs.append(q)
        P = probs(fwd(tp, seq + drafts, 0, empty())[len(seq) - 1:]); passes += 1; acc = 0
        for i, x in enumerate(drafts):
            if rng.random() < min(1.0, P[i][x] / qs[i][x]): acc += 1
            else:
                r = np.maximum(P[i] - qs[i], 0); r /= r.sum(); drafts[i] = int(rng.choice(V, p=r)); break
        new = drafts[:acc + 1] if acc < K else drafts + [int(rng.choice(V, p=P[K]))]
        seq += new; n_out += len(new)
    return passes
est = [estimator(p, sample_target(p)) for p in prompts]
real = [real_spec_sampling(p) for p in prompts]
print(f"temp {TEMP}, k={K}: real spec-sampling passes {np.mean(real):.3f} (se {np.std(real)/np.sqrt(NPR):.3f})  "
      f"log-based DP estimator {np.mean(est):.3f} (se {np.std(est)/np.sqrt(NPR):.3f})")

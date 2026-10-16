"""Speculative SAMPLING: expected verification passes from an ordinary target sample (a production log).

Identity (standard speculative sampling, one draft chain of length k): the probability that drafts x1..xm are all accepted is
prod_i min(p_i, q_i)(x_i). Summed over sequences this equals E_{x ~ p}[prod_i min(1, q_i(x_i) / p_i(x_i))]. Along one logged target
sample, a token z is "drafted and accepted" with weight a(z) = min(1, q(z)/p(z)), "corrected" (ends a pass) with weight 1 - a(z),
and a bonus token after k accepts has weight 1. A renewal dynamic program over the logged path gives an unbiased estimate of the
expected number of target passes.

Inputs per position: the target probability of the CHOSEN token (what serving logs and RL rollouts already store) and the draft
probability of the same token (one teacher-forced draft pass). Both must come from the same logits pipeline the sampler used
(temperature, top-p/top-k, penalties).
Related work: Not-a-Bandit (Liu et al., arXiv 2510.20064) estimates per-chunk acceptance from full distributions (1 - TV).
"""


def expected_passes_from_log(p_chosen, q_chosen, k, start=0):
    a = [min(1.0, q / p) if p > 0 else 0.0 for p, q in zip(p_chosen, q_chosen)]
    n = len(a)
    E = [0.0] * (n + k + 2)
    for s in range(n - 1, start - 1, -1):
        tot, pre, ended = 1.0, 1.0, False
        for m in range(k):
            if s + m >= n:
                ended = True
                break
            tot += pre * (1 - a[s + m]) * E[s + m + 1]
            pre *= a[s + m]
        if not ended:
            tot += pre * E[s + k + 1]
        E[s] = tot
    return E[start]

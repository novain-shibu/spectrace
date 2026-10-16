"""Greedy speculative decoding: exact verification-pass counts from a recorded target output.

Under greedy verification a drafted token is accepted only if it equals the target's argmax, and every accepted prefix is a true
prefix. So a drafter is fully described by, at every position j, the rank of the true token y[j] in the drafter's prediction given
prompt + y[:j]. For a draft model this is ONE teacher-forced forward pass over prompt + y.

Exact for: chain drafting with k tokens per pass; trees that expand the top-b tokens at every node (local rule).
Not exact for: trees that choose nodes by a global budget over off-path probabilities (e.g. EAGLE-2 dynamic trees).
Engine detail: HF assisted generation and vLLM draft the very first token too (start=0); pass start=1 if the first token
comes from a separate prefill step.
The replay must use the same logits pipeline the engine used (e.g. repetition_penalty from generation_config).
"""


def passes_from_ranks(ranks, k, b=1, start=0):
    """Target verification passes for one output. ranks[j] = rank of y[j] under the drafter (0 = its top guess)."""
    pos, n_pass = start, 0
    while pos < len(ranks):
        n = 0
        while n < k and pos + n < len(ranks) and ranks[pos + n] < b:
            n += 1
        pos += n + 1
        n_pass += 1
    return n_pass


def tree_tokens(k, b):
    """Drafted tokens per pass for a full b-ary tree of depth k (a chain when b == 1)."""
    return k if b == 1 else sum(b ** d for d in range(1, k + 1))

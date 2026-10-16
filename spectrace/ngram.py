"""Deterministic drafters as an exact engine oracle.

For a drafter that is a pure function of the tokens (n-gram / prompt lookup), the engine's drafted and accepted counts can be
reproduced exactly by replaying the drafter on the engine's OWN output. This holds for greedy AND sampling: with a one-hot draft d,
rejection sampling accepts d with probability p(d) and otherwise resamples from a residual that excludes d, so the emitted token
equals d iff the draft was accepted. Any mismatch between the replay and the engine counters is an engine bug.

vllm_ngram_propose is a Python port of vLLM's _find_longest_matched_ngram_and_propose_tokens (Apache-2.0, vllm-project/vllm),
verified identical on 30/30 requests against vLLM 0.31.0.
"""


def vllm_ngram_propose(tokens, k, min_n, max_n, max_model_len):
    n = len(tokens)
    if n < min_n:
        return []
    k = min(k, max_model_len - n)
    if k <= 0:
        return []
    r = tokens[::-1]
    lps = [0] * max_n
    longest = position = prev = 0
    i = 1
    while i < n:
        if r[prev] == r[i]:
            prev += 1
            if prev >= longest:
                longest, position = prev, i
            if i < max_n:
                lps[i] = prev
            if prev == max_n:
                prev = lps[max_n - 1]
            i += 1
        elif prev != 0:
            prev = lps[prev - 1]
        else:
            i += 1
    if longest < min_n:
        return []
    start = n - 1 - position + longest
    return tokens[start:start + min(k, n - start)]


def replay_deterministic(prompt_ids, output_ids, propose, start=1):
    """Replay a deterministic drafter on an engine's output. Returns [drafts, drafted_tokens, accepted]."""
    pos, drafts, drafted, accepted = start, 0, 0, 0
    while pos < len(output_ids):
        d = propose(prompt_ids + output_ids[:pos])
        n = 0
        while n < len(d) and pos + n < len(output_ids) and d[n] == output_ids[pos + n]:
            n += 1
        if d:
            drafts += 1
            drafted += len(d)
            accepted += n
        pos += n + 1
    return [drafts, drafted, accepted]

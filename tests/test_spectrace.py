# Tests that need no GPU and no downloads. Run: python -m pytest -q tests   (or: python tests/test_spectrace.py)
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from spectrace import passes_from_ranks, expected_passes_from_log, vllm_ngram_propose, replay_deterministic


def test_greedy_passes():
    assert passes_from_ranks([0] * 10, k=4) == 2       # positions 0-4, then 5-9
    assert passes_from_ranks([9] * 5, k=4) == 5        # nothing accepted


def test_sampling_dp_limits():
    # always accepted / never accepted reduce to the deterministic counts
    assert abs(expected_passes_from_log([1.0] * 10, [1.0] * 10, k=4) - passes_from_ranks([0] * 10, 4)) < 1e-12
    assert abs(expected_passes_from_log([0.5] * 7, [0.0] * 7, k=4) - 7) < 1e-12


def test_vllm_ngram_replay_matches_engine_30_of_30():
    """Replaying vLLM's n-gram proposer on vLLM's own outputs reproduces vLLM's spec-decode counters exactly."""
    d = json.load(open(os.path.join(HERE, "vllm_ngram_30_requests.json")))
    c = d["speculative_config"]
    ok = 0
    for r in d["requests"]:
        got = replay_deterministic(r["prompt_ids"], r["output_ids"],
                                   lambda t: vllm_ngram_propose(t, c["num_speculative_tokens"], c["prompt_lookup_min"],
                                                                c["prompt_lookup_max"], d["max_model_len"]))
        ok += got == r["engine"]
    assert ok == 30, ok


if __name__ == "__main__":
    for f in (test_greedy_passes, test_sampling_dp_limits, test_vllm_ngram_replay_matches_engine_30_of_30):
        f()
        print("PASS", f.__name__)

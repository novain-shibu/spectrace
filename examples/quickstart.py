# Same snippet as README "Use"; runs on CPU with no dependencies.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectrace import expected_passes_from_log, passes_from_ranks, replay_deterministic, vllm_ngram_propose

p_chosen = [0.9, 0.8, 0.95, 0.4, 0.7, 0.9]
q_chosen = [0.85, 0.9, 0.9, 0.1, 0.6, 0.95]
print(expected_passes_from_log(p_chosen, q_chosen, k=4))
print(passes_from_ranks([0, 0, 3, 0, 0, 0, 1, 0], k=4))
propose = lambda toks: vllm_ngram_propose(toks, k=4, min_n=2, max_n=5, max_model_len=2048)
print(replay_deterministic(prompt_ids=[1, 2, 3, 1, 2], output_ids=[3, 1, 2, 3, 9], propose=propose))

# spectrace

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23413796.svg)](https://doi.org/10.5281/zenodo.23413796)

Predicting speculative-decoding pass counts (target forward passes per output, not wall-clock) from outputs a model has already produced, checked against real engines (Hugging Face,
vLLM), used to choose drafters for RL rollouts, plus an exact conformance check for inference-engine speculative-decoding counters.

**Read first:** the offline estimator of expected speculative-sampling rounds is from Zhao & Cai, arXiv:2610.10411 (2026-10-07).
This repository contributes engine validation, RL drafter-selection experiments and the conformance check. Full write-up:
[REPORT.md](REPORT.md).

## Headline numbers (T4, Qwen2.5 0.5B/1.5B)
- Greedy, HF assisted generation, fp32: predicted target passes 931 = measured 931 (30/30 requests).
- Sampling, HF (T=0.8, 120 runs): predicted 27.36 vs measured 27.28 passes.
- Exact mechanism check: expected accepted tokens from HF's own p,q vs ours: -0.02% and -0.31%.
- Drafter selection (6 policy/drafter pairs): predicted vs real passes within 0.2-1.2%, best drafter ranked correctly, and a swap
  of the 2nd/3rd drafters between two policies followed; ~2 s to predict vs ~5 min to measure.
- Real RL pair (Qwen2.5-1.5B -> SimpleRL-Zoo-1.5B): drafter ranking unchanged by RL and preserved by the prediction; the 11-19%
  drop in passes after RL is tracked; errors within ~1 standard error.
- vLLM 0.31: n-gram counters reproduced exactly (30/30). Isolated a previously unreported (to our knowledge) accepted-token
  overcount at `max_tokens` (counted before the output is trimmed), and independently quantified the known `ngram_gpu`
  drafted-token inflation (~10x; open vLLM PR #44056).

## Use
```python
from spectrace import expected_passes_from_log, passes_from_ranks, replay_deterministic, vllm_ngram_propose

# Sampling: one recorded output; probability of each chosen token under the target (p) and a candidate drafter (q)
p_chosen = [0.9, 0.8, 0.95, 0.4, 0.7, 0.9]
q_chosen = [0.85, 0.9, 0.9, 0.1, 0.6, 0.95]
print(expected_passes_from_log(p_chosen, q_chosen, k=4))   # expected target forward passes with this drafter

# Greedy: rank of the true next token under the drafter (0 = the drafter's top guess was right)
print(passes_from_ranks([0, 0, 3, 0, 0, 0, 1, 0], k=4))   # exact pass count

# Deterministic drafter replayed on an engine's output -> [drafts, drafted tokens, accepted]
propose = lambda toks: vllm_ngram_propose(toks, k=4, min_n=2, max_n=5, max_model_len=2048)
print(replay_deterministic(prompt_ids=[1, 2, 3, 1, 2], output_ids=[3, 1, 2, 3, 9], propose=propose))
```
- `passes_from_ranks(ranks, k)`: exact greedy passes from per-position drafter ranks (0 = drafter's top guess is correct).
- `expected_passes_from_log(p_chosen, q_chosen, k)`: expected speculative-sampling passes for one recorded output.
- `replay_deterministic(prompt_ids, output_ids, propose)`: drafts / drafted / accepted for a deterministic drafter on an engine's output.

Test (CPU, no downloads): `python tests/test_spectrace.py`

## Limits
Small models, one GPU type, HF + vLLM only; EAGLE/MTP drafters untested; passes are not wall-clock. See REPORT.md section 4.

## Cite
Novain Shibu (2026). spectrace. Zenodo. https://doi.org/10.5281/zenodo.23413796 (v0.1.0: 10.5281/zenodo.23413797).
Please also cite Zhao & Cai, arXiv:2610.10411, for the offline round estimator.

## Licence
Apache-2.0. See NOTICE (vLLM-derived code, prior work, models and data).

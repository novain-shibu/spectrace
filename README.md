# spectrace

Predicting speculative-decoding speed-ups from outputs a model has already produced, checked against real engines (Hugging Face,
vLLM), used to choose drafters for RL rollouts, plus an exact conformance check for inference-engine speculative-decoding counters.

**Read first:** the offline estimator of expected speculative-sampling rounds is from Zhao & Cai, arXiv:2610.10411 (2026-10-07).
This repository contributes engine validation, RL drafter-selection experiments and the conformance check. Full write-up:
[REPORT.md](REPORT.md).

## Headline numbers (T4, Qwen2.5 0.5B/1.5B)
- Greedy, HF assisted generation, fp32: predicted target passes 931 = measured 931 (30/30 requests).
- Sampling, HF (T=0.8, 120 runs): predicted 27.36 vs measured 27.28 passes.
- Exact mechanism check: expected accepted tokens from HF's own p,q vs ours: -0.02% and -0.31%.
- Drafter selection (6 policy/drafter pairs): predicted vs real passes within 0.2-1.2%, best drafter ranked correctly; ~2 s vs ~5 min.
- Real RL pair (Qwen2.5-1.5B -> SimpleRL-Zoo-1.5B): ranking correct before and after RL; errors within ~1 standard error.
- vLLM 0.31: n-gram counters reproduced exactly (30/30); found a `max_tokens` accounting tail and quantified the open `ngram_gpu`
  draft-count inflation (~10x, vLLM PR #44056).

## Use
```python
from spectrace import passes_from_ranks, expected_passes_from_log, vllm_ngram_propose, replay_deterministic
```
- `passes_from_ranks(ranks, k)`: exact greedy passes from per-position drafter ranks (0 = drafter's top guess is correct).
- `expected_passes_from_log(p_chosen, q_chosen, k)`: expected speculative-sampling passes for one recorded output.
- `replay_deterministic(prompt_ids, output_ids, propose)`: drafts / drafted / accepted for a deterministic drafter on an engine's output.

Test (CPU, no downloads): `python tests/test_spectrace.py`

## Limits
Small models, one GPU type, HF + vLLM only; EAGLE/MTP drafters untested; passes are not wall-clock. See REPORT.md section 4.

## Licence
Apache-2.0. See NOTICE (vLLM-derived code, prior work, models and data).

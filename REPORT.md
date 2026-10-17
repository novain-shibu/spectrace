# Predicting speculative-decoding pass counts without running the drafter: real-engine validation, drafter selection for RL rollouts, and an exact engine check

Novain Shibu, 2026-10-17. Version 0.1.2 (corrections: section 2.2 in 0.1.2, section 2.6 in 0.1.1). DOI: 10.5281/zenodo.23413796 (all versions); v0.1.2: 10.5281/zenodo.23431513. All numbers below are regenerated from the raw Kaggle logs by `results/make_results.py`
(output: `results/summary.md`). Hardware: one free NVIDIA T4 (Kaggle). Models: Qwen2.5 0.5B / 1.5B. Engines: Hugging Face
transformers 5.16 (assisted generation) and vLLM 0.31.0.

## 0. Prior work and what is (and is not) claimed here

The core estimator is **not new**. Zhao and Cai (arXiv:2610.10411, 2026-10-07) give an exact offline evaluator of the expected
number of speculative-sampling rounds over target-model rollouts, with acceptance terms min{1, q(x)/p(x)} on the sampled tokens,
and use it for paired drafter comparison without running speculative decoding. Liu et al. (Not-a-Bandit, arXiv:2510.20064)
evaluate drafters counterfactually from one trajectory using full distributions. We derived the same estimator independently and
learned of 2610.10411 on 2026-10-16, before this release.

What this report adds:
1. **Validation against real engines** (the papers above report the estimate itself): HF and vLLM, greedy and sampling.
2. **Drafter selection for RL rollouts**: ranking candidate drafters from ordinary rollouts, including across a real RL policy
   change (Qwen2.5-1.5B -> SimpleRL-Zoo-1.5B).
3. **An exact engine conformance check** for deterministic or greedy drafters. On vLLM 0.31 it independently detected two KNOWN
   counter issues: (a) accepted drafts counted before the output is trimmed at a stop point such as `max_tokens` (vLLM issue
   #56101; open PRs #56137, #56195), and (b) the `ngram_gpu` drafted-token inflation (open PR #44056). It found them without
   prior knowledge of either, which is evidence that the check works; neither is a new finding.

Throughout, "speed-up" means fewer target forward passes (rounds); wall-clock time is not measured here.

## 1. The quantities

Speculative decoding: a small drafter proposes k tokens, the target verifies them in one forward pass ("pass" / "round").
Fewer passes per output = faster. Two predictors, both computed from outputs the target has already produced:

- **Greedy (exact).** A drafted token is accepted iff it equals the target's argmax. Along the recorded output, a drafter is fully
  described by whether its top guess equals the true next token; one teacher-forced drafter pass gives the exact pass count for chain
  drafting with any k (and for local top-b trees). Not exact for trees chosen by a global budget (e.g. EAGLE-2 dynamic trees).
- **Sampling (unbiased in expectation).** Weight each recorded token z by a(z) = min(1, q(z)/p(z)) ("drafted and accepted") or 1 - a(z)
  ("corrected, pass ends"); a renewal dynamic program over the recorded output gives the expected number of passes
  (`spectrace/sampling.py`). Inputs: the target probability of the chosen token (what serving logs and RL rollouts store) and the
  drafter's probability of the same token. This is the estimator of 2610.10411.
- **Deterministic / one-hot drafters in a real engine (exact per run).** With a one-hot draft d, rejection sampling accepts d with
  probability p(d) and otherwise resamples from a residual that excludes d, so the emitted token equals d iff the draft was accepted.
  Replaying the drafter on the engine's own outputs therefore reproduces its drafted/accepted counters exactly, for greedy and for any
  sampling setting. A difference is an engine issue.

## 2. Results

### 2.1 Greedy, draft-model drafter, HF assisted generation (Qwen2.5-1.5B-Instruct target, Qwen2.5-0.5B-Instruct drafter, k=4, 30 prompts)
- fp32: target forward calls 931 = predicted 931; per request 30/30 equal; outputs identical to plain greedy 30/30.
- fp16: 932 vs 929; the 4 differing requests are among the 5 where HF's own fp16 assisted output diverged from HF plain greedy.
- Pitfall found on the way: Qwen2.5 instruct `generation_config` sets `repetition_penalty=1.1`; HF applies it to target and drafter,
  so a replay that ignores it disagrees (fp32: 360 vs 374). The replay must use the engine's full logits pipeline.

### 2.2 Speculative sampling, HF (T=0.8, k=4, 30 prompts x 8 seeds = 240 runs; job real_sampling2)
Real passes 28.23 (SE 0.33) vs predicted 28.28 (SE 0.31), +0.19%, using only ordinary target samples + chosen-token probabilities +
one teacher-forced drafter pass. Paired by prompt, the relative difference is +0.19% +- 0.84% (95% CI -1.45% to +1.84%): the
prediction agrees with the engine within Monte Carlo noise; this run cannot resolve sub-percent bias. Both sides mask <|endoftext|>
as HF's `min_new_tokens` processor does (unmasked: 28.28, no visible effect at this size).

**Correction (v0.1.2).** v0.1.0/v0.1.1 reported 27.36 predicted vs 27.28 measured (real_sampling1, 120 runs). That comparison was
off by one: it used the estimator from position 1 (E[1]) against `calls - 1`, treating HF's first target forward as a separate
prefill. In HF assisted decoding the first target forward already verifies the first draft block, so the right comparison is E[0]
against the full call count (as already used in 2.1 and 2.4). real_sampling2 uses that convention, matches the EOS mask on both
sides and doubles the runs. Thanks to the authors of arXiv:2610.10411 for pointing this out, and for noting that runs of this size
carry 1-4% standard errors. The old log is kept in `results/logs` and listed as superseded in `results/summary.md`.

### 2.3 Exact mechanism check (no sampling noise)
HF's `_speculative_sampling` was wrapped to record the p and q it actually uses for every drafted token; expected accepted tokens
computed from HF's own p,q vs from the predictor's p,q on the same prefixes (real base policy Qwen2.5-1.5B, T=1, k=4):
- Qwen2.5-0.5B drafter: 738.97 (HF) vs 738.84 (ours), -0.02%; realized 720.
- Qwen2.5-0.5B-Instruct drafter: 676.79 vs 674.69, -0.31%; realized 678.
- Instruct drafter, separate run: 937.7 vs 939.3 raw (+0.17%); 937.9 (+0.02%) when masking <|endoftext|> on both models as HF does
  under `min_new_tokens`.

### 2.4 Choosing a drafter from rollouts (T=1, k=4, 24 GSM8K prompts x 2)
| policy <- drafter | predicted passes | real passes | error |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct <- 0.5B-Instruct | 31.90 +- 0.44 | 32.00 +- 0.43 | -0.3% |
| Qwen2.5-1.5B-Instruct <- 0.5B (base) | 33.94 +- 0.48 | 33.54 +- 0.43 | +1.2% |
| Qwen2.5-1.5B-Instruct <- Coder-0.5B-Instruct | 34.19 +- 0.53 | 34.08 +- 0.50 | +0.3% |
| Qwen2.5-Math-1.5B-Instruct <- 0.5B-Instruct | 33.92 +- 0.71 | 33.85 +- 0.60 | +0.2% |
| Qwen2.5-Math-1.5B-Instruct <- 0.5B (base) | 36.52 +- 0.73 | 36.46 +- 0.66 | +0.2% |
| Qwen2.5-Math-1.5B-Instruct <- Coder-0.5B-Instruct | 35.47 +- 0.70 | 35.38 +- 0.87 | +0.3% |

The best drafter is ranked correctly for both policies. Between the two policies the 2nd/3rd places swap (base vs Coder), and the
prediction follows the swap (for the Instruct policy that gap is about one standard error, so that half of the swap is not certain). Prediction took 2-6 s per drafter vs ~290-320 s to measure.

### 2.5 A real RL policy change: Qwen2.5-1.5B -> hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo (T=1, k=4, 20 prompts x 2)
| policy | drafter | predicted | real | error |
|---|---|---|---|---|
| before RL | 0.5B (base) | 50.69 +- 1.80 | 48.17 +- 1.75 | +5.2% |
| before RL | 0.5B-Instruct | 54.57 +- 2.01 | 56.08 +- 1.97 | -2.7% |
| before RL | Coder-0.5B-Instruct | 58.87 +- 2.32 | 61.33 +- 2.73 | -4.0% |
| after RL | 0.5B (base) | 44.57 +- 1.61 | 42.90 +- 1.56 | +3.9% |
| after RL | 0.5B-Instruct | 46.38 +- 1.89 | 48.10 +- 1.97 | -3.6% |
| after RL | Coder-0.5B-Instruct | 49.69 +- 2.20 | 49.62 +- 2.91 | +0.1% |

Here the full ranking (base < Instruct < Coder) is the SAME before and after RL, and the prediction preserves it at both real
policies; what changes is the level: RL lowered the passes needed by every drafter by 11-19%, which the prediction tracks. Every
error is within about one standard error. Interpolated-weight intermediate policies (prediction only, a proxy, not real checkpoints) move smoothly; see
`results/summary.md`. A paired per-prompt re-run (32 prompts) gave errors +2.9 +- 3.4%, -4.4 +- 3.6%, +5.5 +- 3.8%, +1.7 +- 4.4%;
no 95% interval excludes zero. Across tests the 0.5B-Instruct drafter was under-predicted three times (-2.7%, -3.6%, -4.4%), each within about one
standard error. Section 2.3 rules out a per-token p/q mismatch (expected acceptance from HF's own p,q matches ours within
0.02-0.31%), and end-of-sequence masking under `min_new_tokens` accounts for at most ~0.2%. The estimator is unbiased by
construction (the identity already includes resampling from the residual after a rejection), so we attribute the residual to
sampling noise; a larger run would settle it. Per-token end-of-sequence proposals were not logged, so they could not be analysed.

### 2.6 vLLM 0.31 conformance (Qwen2.5-1.5B-Instruct, T4, fp16)
- **Accepted-token overcount at `max_tokens` (known: vLLM issue #56101, open PRs #56137 and #56195 on terminal-step accounting).**
  Correction: v0.1.0 of this report called this previously unreported; that was wrong. vLLM 0.31's scheduler computes
  `num_accepted = len(generated_token_ids) - num_sampled` before `_update_request_with_output` trims the step's tokens at the stop
  point, so drafts verified on the final pass beyond `max_tokens` are counted as accepted and then discarded. In our 17-config sweep
  every mismatch was on a request with finish_reason=length (+1 to +3 accepted tokens per affected config). Metrics only; outputs
  are unaffected. Our replay detected it independently and matches the mechanism described in #56137.
- n-gram (CPU), 30 single requests, greedy: replaying vLLM's proposer on vLLM's outputs reproduces drafts, drafted tokens and accepted
  tokens exactly for 30/30 requests (`tests/test_spectrace.py` re-checks this without a GPU).
- 17 configurations (greedy and sampling with top-p/top-k and penalties, batching, prefix caching off, chunked long prompts, k 2/4/8,
  fp32, stop strings, log-probs): the only differences are the `max_tokens` overcount above.
- `ngram_gpu`: drafted-token counters inflated ~9-10x (e.g. 11,936 vs 1,283 drafted, 302 = 302 accepted), so reported acceptance is
  ~2.5% instead of ~23.5%. This is the issue addressed by open vLLM PR #44056 (trimmed invalid GPU drafts never reach the stats).
- `draft_model` (Qwen2.5-0.5B drafter): drafts equal the fp32 teacher-forced key (916 = 916 greedy fp32; 1,010 = 1,010 sampling);
  engine accepted exceeds the key (2,427 vs 2,390), consistent with the same `max_tokens` tail.

## 3. Why it matters for RL (indicative)
Generation is reported at 65-72% of RL step time (NeMo-RL, arXiv:2604.26779); drafters drift as the policy changes (2603.12617,
2609.07108). In the runs above a wrong or stale drafter cost 6-27% more target passes, roughly 4-20% of step time if passes ~ time.
Rollouts with chosen-token log-probabilities already exist in RL, so ranking candidate drafters costs one drafter forward pass each.

## 4. Limits
Small models (0.5B/1.5B), one GPU type (T4), HF and vLLM only. EAGLE/MTP-style drafters (which read target hidden states) are not
tested. Passes are not wall-clock time. Prediction-vs-real tests have ~1-4% sampling noise per row. Interpolated policies are a proxy.

## 5. Reproduce
`python tests/test_spectrace.py` (CPU). GPU experiments: `kaggle/<job>/run.py` (each ran as a Kaggle T4 script; `kaggle/wheelhouse`
caches vLLM wheels and models). `python results/make_results.py` regenerates every table from `results/logs`.

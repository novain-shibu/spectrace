## Greedy, draft-model drafter, HF assisted generation (diag-2)

- torch.float32: {'target_calls': 931, 'sim_pos0': 931, 'identical': '30/30'}
- torch.float16: {'target_calls': 932, 'sim_pos0': 929, 'identical': '25/30'}
- fp32_per_request_equal: 30/30

## Speculative sampling, HF real engine (T=0.8, k=4, 120 runs)

- real passes 27.28 (SE 0.47) vs predicted 27.36 (SE 0.42)

## Exact mechanism check: expected accepted tokens from HF's own p,q vs ours

- Qwen/Qwen2.5-0.5B: HF 738.97 vs ours 738.84 (-0.02%), realized 720
- Qwen/Qwen2.5-0.5B-Instruct: HF 676.79 vs ours 674.69 (-0.31%), realized 678
- Instruct drafter, p:raw|q:raw: HF 937.7 vs ours 939.3 (+0.17%)
- Instruct drafter, p:mask_endoftext|q:mask_endoftext: HF 937.7 vs ours 937.9 (+0.02%)

## Drafter selection from rollouts (rlselect-1, T=1, k=4)

| policy <- drafter | predicted | real | error |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct <- Qwen2.5-0.5B-Instruct | 31.90 +- 0.44 | 32.00 +- 0.43 | -0.3% |
| Qwen2.5-1.5B-Instruct <- Qwen2.5-0.5B | 33.94 +- 0.48 | 33.54 +- 0.43 | +1.2% |
| Qwen2.5-1.5B-Instruct <- Qwen2.5-Coder-0.5B-Instruct | 34.19 +- 0.53 | 34.08 +- 0.50 | +0.3% |
| Qwen2.5-Math-1.5B-Instruct <- Qwen2.5-0.5B-Instruct | 33.92 +- 0.71 | 33.85 +- 0.60 | +0.2% |
| Qwen2.5-Math-1.5B-Instruct <- Qwen2.5-0.5B | 36.52 +- 0.73 | 36.46 +- 0.66 | +0.2% |
| Qwen2.5-Math-1.5B-Instruct <- Qwen2.5-Coder-0.5B-Instruct | 35.47 +- 0.70 | 35.38 +- 0.87 | +0.3% |

## Real RL pair Qwen2.5-1.5B -> SimpleRL-Zoo-1.5B (rldrift-1; alpha 0.33/0.67 = weight interpolation proxy)

| alpha | drafter | predicted | real | error |
|---|---|---|---|---|
| 0.0 | Qwen2.5-0.5B | 50.69 +- 1.80 | 48.17 +- 1.75 | +5.2% |
| 0.0 | Qwen2.5-0.5B-Instruct | 54.57 +- 2.01 | 56.08 +- 1.97 | -2.7% |
| 0.0 | Qwen2.5-Coder-0.5B-Instruct | 58.87 +- 2.32 | 61.33 +- 2.73 | -4.0% |
| 0.33 | Qwen2.5-0.5B | 48.11 +- 1.69 | - | - |
| 0.33 | Qwen2.5-0.5B-Instruct | 51.61 +- 1.95 | - | - |
| 0.33 | Qwen2.5-Coder-0.5B-Instruct | 55.95 +- 2.42 | - | - |
| 0.67 | Qwen2.5-0.5B | 46.52 +- 1.90 | - | - |
| 0.67 | Qwen2.5-0.5B-Instruct | 49.61 +- 2.28 | - | - |
| 0.67 | Qwen2.5-Coder-0.5B-Instruct | 53.02 +- 2.65 | - | - |
| 1.0 | Qwen2.5-0.5B | 44.57 +- 1.61 | 42.90 +- 1.56 | +3.9% |
| 1.0 | Qwen2.5-0.5B-Instruct | 46.38 +- 1.89 | 48.10 +- 1.97 | -3.6% |
| 1.0 | Qwen2.5-Coder-0.5B-Instruct | 49.69 +- 2.20 | 49.62 +- 2.91 | +0.1% |

## Paired per-prompt check (rlpaired-1)

| alpha | drafter | predicted | real | paired rel. error +- SE |
|---|---|---|---|---|
| 0.0 | Qwen2.5-0.5B | 47.90 | 47.62 | +2.9% +- 3.4% |
| 0.0 | Qwen2.5-0.5B-Instruct | 51.53 | 55.44 | -4.4% +- 3.6% |
| 1.0 | Qwen2.5-0.5B | 43.91 | 42.34 | +5.5% +- 3.8% |
| 1.0 | Qwen2.5-0.5B-Instruct | 45.06 | 45.69 | +1.7% +- 4.4% |

## vLLM 0.31 conformance (T4, Qwen2.5-1.5B-Instruct)

| config | engine [drafts, draft tokens, accepted] | answer key | match |
|---|---|---|---|
| A_greedy_b1 | [322, 1283, 302] | [322, 1283, 302] | True |
| B_greedy_batch | [321, 1279, 298] | [321, 1279, 298] | True |
| C_greedy_noprefixcache | [317, 1263, 299] | [317, 1263, 299] | True |
| D_greedy_long_chunked | [129, 516, 355] | [129, 516, 354] | False |
| E_greedy_k8 | [307, 2393, 309] | [307, 2393, 307] | False |
| F_greedy_k2_n1_3 | [1208, 2415, 404] | [1208, 2415, 403] | False |
| G_sample_T08_b1 | [309, 1228, 269] | [309, 1228, 268] | False |
| H_sample_T1_batch | [285, 1135, 258] | [285, 1135, 258] | True |
| I_sample_topp_topk | [308, 1229, 266] | [308, 1229, 266] | True |
| J_sample_reppen | [201, 801, 118] | [201, 801, 118] | True |
| K_greedy_reppen | [230, 917, 216] | [230, 917, 214] | False |
| L_greedy_fp32 | [327, 1303, 299] | [327, 1303, 299] | True |
| M_greedy_batch_maxseqs4 | [316, 1259, 295] | [316, 1259, 295] | True |
| N_greedy_stop | [95, 380, 80] | [95, 380, 80] | True |
| O_sample_freqpen | [255, 1017, 190] | [255, 1017, 187] | False |
| P_greedy_logprobs | [322, 1283, 302] | [322, 1283, 302] | True |
| Q_long_batch_sample | [136, 544, 365] | [136, 544, 363] | False |
| G1_ngramgpu_greedy_b1 | [2984, 11936, 302] | [322, 1283, 302] | False |
| G2_ngramgpu_greedy_batch | [2991, 11964, 295] | [319, 1271, 295] | False |
| G3_ngramgpu_sample_b1 | [3069, 12276, 256] | [294, 1174, 253] | False |
| G4_ngramgpu_sample_batch | [2918, 11672, 240] | [268, 1067, 235] | False |
| G5_ngramgpu_long_chunked | [173, 692, 355] | [129, 516, 354] | False |
| G6_ngramgpu_batch_maxseqs4 | [2988, 11952, 298] | [321, 1279, 298] | False |
| G7_ngramgpu_k8_n1_3 | [2810, 22480, 478] | [1135, 8879, 476] | False |
| G8_ngramgpu_long_batch_sample | [188, 752, 338] | [126, 504, 338] | False |
| G9_ngramgpu_reppen_sample | [3075, 12300, 170] | [205, 815, 170] | False |

All CPU-ngram mismatches (D, E, F, G, K, O, Q) are on requests that stopped at max_tokens (finish_reason=length): drafts verified on the final pass beyond the budget are counted as accepted, then truncated. ngram_gpu rows: draft counts inflated ~9-10x (open vLLM PR #44056: trimmed invalid GPU drafts never reach the stats); accepted counts agree.

Draft-model drafter (vLLM draft_model, Qwen2.5-0.5B -> 1.5B): engine vs fp32 teacher-forced answer key [drafts, draft tokens, accepted]:
- D1_draft_greedy_fp32_b1: engine [916, 3664, 2427] vs key [916, 3664, 2390]
- D2_draft_greedy_fp16_b1: engine [916, 3664, 2426] vs key [917, 3668, 2388]
- D3_draft_sample_fp32_b1: engine [1010, 4040, 2243] vs key [1010, 4040, 2209]
- D4_draft_greedy_fp32_batch: engine [916, 3664, 2427] vs key [916, 3664, 2390]
Drafts (= passes) agree; the extra engine 'accepted' is consistent with the same max_tokens tail (confirmed per request for the ngram paths only).

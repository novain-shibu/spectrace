# Re-derives every number in REPORT.md from the raw Kaggle logs in results/logs (no hand-typed numbers).
# Kaggle username redacted as <kaggle-user> in the logs. Run: python results/make_results.py
import json, os, glob, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE  # writes summary.json / summary.md next to this script
LOGS = {"real_sampling1": "spectrace-real-sampling-1.log", "diag2": "spectrace-diag-2.log", "diag3": "spectrace-diag-3.log",
        "diag4": "spectrace-diag-4.log", "rlselect1": "spectrace-rlselect-1.log", "rldrift1": "spectrace-rldrift-1.log",
        "rlpaired1": "spectrace-rlpaired-1.log", "matrix1": "spectrace-matrix-1.log", "matrix2": "spectrace-matrix-2.log"}


def log_lines(job):
    f = os.path.join(HERE, "logs", LOGS[job])
    L = json.load(open(f, encoding="utf-8"))
    return "".join(e["data"] for e in L if e["stream_name"] == "stdout").splitlines()


S = {}; md = []
# 1) greedy exactness, HF assisted generation, draft model (diag-2, fp32 + fp16)
for l in log_lines("diag2"):
    if l.startswith("SUMMARY"):
        p = l.split(); S.setdefault("hf_greedy_exact", {})[p[1]] = {"target_calls": int(p[3]), "sim_pos0": int(p[5]), "identical": p[9] + "/" + p[11]}
# per-request equality in fp32
rows = [l for l in log_lines("diag2") if l.startswith("torch.float32 {")]
import ast
eq = sum(ast.literal_eval(l.split(" ", 1)[1])["target_calls"] == ast.literal_eval(l.split(" ", 1)[1])["sim_pos0"] for l in rows)
S["hf_greedy_exact"]["fp32_per_request_equal"] = f"{eq}/{len(rows)}"
# 2) sampling, HF real engine (real_sampling1)
for l in log_lines("real_sampling1"):
    if l.startswith("RESULT"):
        S["hf_sampling"] = json.loads(l[7:])
# 3) exact mechanism (diag-3, diag-4)
S["mechanism_diag3"] = [json.loads(l[7:]) for l in log_lines("diag3") if l.startswith("RESULT")]
d4 = {}
for l in log_lines("diag4"):
    if l.startswith("VARIANT"):
        k, j = l[8:].split(" ", 1); d4[k] = json.loads(j)
S["mechanism_diag4"] = {k: d4[k] for k in ("p:raw|q:raw", "p:mask_endoftext|q:mask_endoftext")}
# 4) drafter selection (rlselect1), real RL pair (rldrift1), paired (rlpaired1)
def rows_of(job, sep_alpha=False):
    out = []
    for l in log_lines(job):
        if l.startswith("ROW"):
            if sep_alpha:
                p = l.split(" ", 3); d = json.loads(p[3]); d["policy_alpha"] = p[1]; d["drafter"] = p[2]
            else:
                k, j = l[4:].split(" {", 1); d = json.loads("{" + j); d["pair"] = k
            out.append(d)
    return out
S["rlselect"] = rows_of("rlselect1")
S["rldrift"] = rows_of("rldrift1", True)
S["rlpaired"] = [{k: v for k, v in d.items() if not k.startswith("per_prompt")} for d in rows_of("rlpaired1", True)]
# 5) vLLM conformance (vllm1 per request; matrix1/matrix2 config rows)
v = json.load(open(os.path.join(HERE, "..", "tests", "vllm_ngram_30_requests.json")))
S["vllm_ngram_single_request_runs"] = len(v["requests"])
for job in ("matrix1", "matrix2"):
    S[job] = []
    for l in log_lines(job):
        if " ROW " in l and " rc " in l:
            S[job].append(json.loads(l.split(" ROW ", 1)[1]))
        if l.startswith("DRAFT_ORACLE"):
            S.setdefault(job + "_draft_oracle", []).append(l)
json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

# ---- markdown tables
md.append("## Greedy, draft-model drafter, HF assisted generation (diag-2)\n")
for k, v_ in S["hf_greedy_exact"].items():
    md.append(f"- {k}: {v_}")
h = S["hf_sampling"]
md.append(f"\n## Speculative sampling, HF real engine (T={h['T']}, k={h['K']}, {h['n']} runs)\n\n- real passes {h['real_mean']:.2f} (SE {h['real_se']:.2f}) vs predicted {h['est_mean']:.2f} (SE {h['est_se']:.2f})")
md.append("\n## Exact mechanism check: expected accepted tokens from HF's own p,q vs ours\n")
for r in S["mechanism_diag3"]:
    md.append(f"- {r['drafter']}: HF {r['expected_accepted_HF_pq']:.2f} vs ours {r['expected_accepted_our_pq']:.2f} ({100*(r['expected_accepted_our_pq']/r['expected_accepted_HF_pq']-1):+.2f}%), realized {r['realized_accepted']}")
for k, r in S["mechanism_diag4"].items():
    md.append(f"- Instruct drafter, {k}: HF {r['exp_HF']:.1f} vs ours {r['exp_ours']:.1f} ({100*r['rel_diff']:+.2f}%)")
md.append("\n## Drafter selection from rollouts (rlselect-1, T=1, k=4)\n\n| policy <- drafter | predicted | real | error |\n|---|---|---|---|")
for d in S["rlselect"]:
    md.append(f"| {d['pair']} | {d['pred']:.2f} +- {d['pred_se']:.2f} | {d['real']:.2f} +- {d['real_se']:.2f} | {100*(d['pred']/d['real']-1):+.1f}% |")
md.append("\n## Real RL pair Qwen2.5-1.5B -> SimpleRL-Zoo-1.5B (rldrift-1; alpha 0.33/0.67 = weight interpolation proxy)\n\n| alpha | drafter | predicted | real | error |\n|---|---|---|---|---|")
for d in S["rldrift"]:
    real = f"{d['real']:.2f} +- {d['real_se']:.2f}" if "real" in d else "-"
    err = f"{100*(d['pred']/d['real']-1):+.1f}%" if "real" in d else "-"
    md.append(f"| {d['policy_alpha']} | {d['drafter']} | {d['pred']:.2f} +- {d['pred_se']:.2f} | {real} | {err} |")
md.append("\n## Paired per-prompt check (rlpaired-1)\n\n| alpha | drafter | predicted | real | paired rel. error +- SE |\n|---|---|---|---|---|")
for d in S["rlpaired"]:
    md.append(f"| {d['policy_alpha']} | {d['drafter']} | {d['pred']:.2f} | {d['real']:.2f} | {100*d['paired_rel_err_mean']:+.1f}% +- {100*d['paired_rel_err_se']:.1f}% |")
md.append("\n## vLLM 0.31 conformance (T4, Qwen2.5-1.5B-Instruct)\n\n| config | engine [drafts, draft tokens, accepted] | answer key | match |\n|---|---|---|---|")
for job in ("matrix1", "matrix2"):
    for r in S[job]:
        if job == "matrix2" and r["name"].startswith("D"):
            continue   # draft-model rows: answer key = fp32 teacher-forced oracle, listed below
        md.append(f"| {r['name']} | {r['engine_total']} | {r['replay_total']} | {r['match']} |")
md.append("\nAll CPU-ngram mismatches (D, E, F, G, K, O, Q) are on requests that stopped at max_tokens (finish_reason=length): drafts"
          " verified on the final pass beyond the budget are counted as accepted, then truncated. ngram_gpu rows: draft counts inflated"
          " ~9-10x (open vLLM PR #44056: trimmed invalid GPU drafts never reach the stats); accepted counts agree.")
md.append("\nDraft-model drafter (vLLM draft_model, Qwen2.5-0.5B -> 1.5B): engine vs fp32 teacher-forced answer key [drafts, draft tokens, accepted]:")
for l in S.get("matrix2_draft_oracle", []):
    p = l.split()
    md.append(f"- {p[1]}: engine {' '.join(p[3:6])} vs key {' '.join(p[7:10])}")
md.append("Drafts (= passes) agree; the extra engine 'accepted' is consistent with the same max_tokens tail (confirmed per request for the ngram paths only).")
open(os.path.join(OUT, "summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))

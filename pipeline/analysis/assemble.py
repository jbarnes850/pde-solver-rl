"""Assemble story/story-data.json from the work/ products and campaign artifacts (read-only on campaign data)."""
import collections
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import C, CK_STEP, W, available, load  # noqa: E402

OUT = W / "story-data.json"
RUNS = [("calib-train-base-p2-anchored", 0), ("calib-train-base-p2-anchored-cont64", 32), ("calib-train-base-p2-anchored-ext96", 64),
        ("calib-train-base-p2-anchored-ext128", 96), ("calib-train-base-p2-anchored-ext160", 128), ("calib-train-base-p2-anchored-ext192", 160),
        ("calib-train-base-p2-anchored-ext224", 192), ("calib-train-base-p2-anchored-ext256", 224)]


def stability():
    rows = []
    for run, start in RUNS:
        d = C / "training-monitor" / run
        for g in sorted(d.glob("step-*-rank-0.json")):
            s = int(g.name.split("-")[1])
            x = json.loads(g.read_text())
            st = d / f"step-{s - 1:06d}.pt.stats.json"
            y = json.loads(st.read_text()) if st.exists() else {}
            rows.append({"update": s, "lr": (x.get("learning_rates") or [None])[0], "grad_l2": x.get("grad_l2"),
                         "relative_update": x.get("trainable_relative_update_l2"), "kl_k3_to_base": y.get("reference_k3_mean"),
                         "ratio_above5_fraction": y.get("ratio_above5_fraction"), "ess_fraction": y.get("clipped_ess_fraction"),
                         "action_tokens": y.get("action_tokens")})
    return rows


def ledger():
    ev = [json.loads(l) for l in (C / "logs/launch-ledger.jsonl").read_text().splitlines() if l.strip()]
    out = {}
    for e in ev:
        if e["run"].startswith("calib-train-base-p2-anchored") and "rep2" not in e["run"]:
            d = out.setdefault(e["run"], {})
            if "start_utc" in e:
                d["start_utc"] = e["start_utc"]
            if "end_utc" in e:
                d["end_utc"], d["exit"] = e["end_utc"], e.get("exit")
    return out


def darcy_bins():
    edges = [0.01, 0.02, 0.03, 0.05, 0.08, 0.15]
    lab = ["<=1%", "1-2%", "2-3%", "3-5%", "5-8%", "8-15%", ">15%"]
    out = {}
    for s, f in (("fresh", "darcy2d"), ("heldout", "darcy_reaction2d")):
        out[f] = {}
        for ck in available(s):
            rows = [r for r in load(s, ck, keep_code=False) if r["family"] == f]
            v = [r for r in rows if r["validity"] == 1.0 and r["err"] is not None]
            h = collections.Counter()
            for r in v:
                i = next((j for j, e in enumerate(edges) if r["err"] <= e), len(edges))
                h[lab[i]] += 1
            out[f][ck] = {"responses": len(rows), "valid": len(v), "error_bins": {k: h[k] for k in lab},
                          "mean_anchored_reward_3_to_8pct": round(statistics.fmean(r["anchored"] for r in v if 0.03 < r["err"] <= 0.08), 3) if any(0.03 < r["err"] <= 0.08 for r in v) else None,
                          "truncated": sum(r["finish"] == "length" for r in rows)}
    return out


def fft_axes_habit():
    out = {}
    for ck in available("heldout"):
        a1 = [r for r in load("fresh", ck) if r["family"] == "advection1d"]
        a2 = [r for r in load("heldout", ck) if r["family"] in ("advection2d", "advection_diffusion2d")]
        strip = lambda c: "\n".join(l.split("#", 1)[0] for l in c.splitlines())
        bad = [r for r in a2 if re.search(r"\b(i?fft)\([^)\n]*(axes\s*=|axis\s*=\s*\()", strip(r["code"]))]
        out[ck] = {"adv1d_fft_axis_kw": sum(bool(re.search(r"\b(i?r?fft)\([^)\n]*axis\s*=", strip(r["code"]))) for r in a1), "adv1d_n": len(a1),
                   "adv2d_families_1d_fft_with_axes": len(bad), "of_which_strict": sum(r["outcome"] == "strict" for r in bad),
                   "adv2d_families_fft2_or_fftn": sum(bool(re.search(r"fft2|fftn", strip(r["code"]))) for r in a2), "adv2d_n": len(a2),
                   "advection2d_strict": sum(r["outcome"] == "strict" for r in a2 if r["family"] == "advection2d"),
                   "advection2d_crash": sum(r["exec_error"] == "solver_exit_1" for r in a2 if r["family"] == "advection2d")}
    return out


def length_arc():
    out = {}
    for s in ("fresh", "heldout"):
        out[s] = {}
        for ck in available(s):
            rows = load(s, ck, keep_code=False)
            cs = {(c["case"], c["idx"]): c for c in map(json.loads, open(W / f"content-{s}-{ck}.jsonl"))}
            ex = [r for r in rows if r["family"] != "incompressible_ns2d"]
            st = [r for r in ex if r["outcome"] == "strict"]
            fl = [r for r in ex if r["outcome"] != "strict"]
            comp = [r for r in rows if r["finish"] != "length"]
            cc = [cs[(r["case"], r["idx"])] for r in comp]
            out[s][ck] = {"median_tokens_ex_ns2d": statistics.median(r["ntok"] for r in ex),
                          "median_tokens_strict_ex_ns2d": statistics.median(r["ntok"] for r in st),
                          "median_tokens_fail_ex_ns2d": statistics.median(r["ntok"] for r in fl),
                          "truncation_rate_ex_ns2d": round(sum(r["finish"] == "length" for r in ex) / len(ex), 4),
                          "truncation_rate_all": round(sum(r["finish"] == "length" for r in rows) / len(rows), 4),
                          "median_noncomment_code_lines_strict": statistics.median(r["noncomment_lines"] for r in st),
                          "comment_share_of_scored_code": round(statistics.fmean(r["comment_lines"] / max(1, r["code_lines"]) for r in comp), 3),
                          "comment_share_of_whole_response": round(statistics.fmean(c["comment_lines_all"] / max(1, c["lines_all"]) for c in cc), 3),
                          "self_correction_lines_per_response": round(statistics.fmean(cs[(r["case"], r["idx"])]["delib_all"] for r in rows), 2),
                          "self_correction_lines_per_truncated": round(statistics.fmean(cs[(r["case"], r["idx"])]["delib_all"] for r in rows if r["finish"] == "length"), 2),
                          "self_correction_lines_per_strict": round(statistics.fmean(cs[(r["case"], r["idx"])]["delib_all"] for r in rows if r["outcome"] == "strict"), 2),
                          "draft_then_fenced_final_share": round(sum(c["pre_has_def_solver"] for c in cc) / len(cc), 3),
                          "single_unfenced_program_share": round(sum(r["fences"] == 0 for r in comp) / len(comp), 3),
                          "median_draft_final_line_overlap": round(statistics.median(c["draft_final_jaccard"] for c in cc if c["pre_has_def_solver"] and c["draft_final_jaccard"] is not None), 2),
                          "median_scored_code_chars": statistics.median(r["code_chars"] for r in comp)}
    return out


def replay_transposed():
    out = collections.Counter()
    tot = collections.Counter()
    for l in open(C / "offline/checkpoint-replay-extracted.jsonl"):
        r = json.loads(l)
        ck = r["key"].split("|")[0]
        tot[ck] += 1
        m = re.search(r"from shape \(([\d, ]+)\) into shape \(([\d, ]+)\)", r.get("message") or "")
        if m:
            a = [int(x) for x in m.group(1).split(",") if x.strip()]
            b = [int(x) for x in m.group(2).split(",") if x.strip()]
            if sorted(a) == sorted(b) and a != b:
                out[ck] += 1
    return {ck: {"invalid_replayed": tot[ck], "transposed_output_assignment": out[ck]} for ck in tot}


def core5(pfu):
    """Training reward re-weighted to an equal mix of the families present in every phase (removes family-mix shifts).

    Phase C (updates 225-256) has no Darcy or 1D reaction-diffusion rows, so the common families are
    advection, Burgers and 2D reaction-diffusion."""
    phases = collections.defaultdict(set)
    for r in pfu:
        phases[(r["update"] - 1) // 32].add(r["family"])
    core = sorted(set.intersection(*phases.values()))
    out = {"families": core}
    for width, name in ((32, "per_phase"), (16, "per_16_updates")):
        agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0.0]))
        for r in pfu:
            b = (r["update"] - 1) // width
            a = agg[b][r["family"]]
            a[0] += r["strict"]; a[1] += r["n"]; a[2] += r["mean_anchored"] * r["n"]
        rows = []
        for b in sorted(agg):
            fam = agg[b]
            have = [f for f in core if fam[f][1]]
            rows.append({"first": width * b + 1, "last": width * b + width, "families_present": len(have),
                         "equal_mix_anchored_reward": round(statistics.fmean(fam[f][2] / fam[f][1] for f in have), 4),
                         "equal_mix_strict": round(statistics.fmean(fam[f][0] / fam[f][1] for f in have), 4),
                         "raw_anchored_reward": round(sum(v[2] for v in fam.values()) / sum(v[1] for v in fam.values()), 4),
                         "by_family_anchored": {f: round(fam[f][2] / fam[f][1], 3) for f in fam if fam[f][1]}})
        out[name] = rows
    return out


def main():
    tr = json.loads((W / "training.json").read_text())
    sv = json.loads((W / "solvable.json").read_text())
    bh = json.loads((W / "behavior.json").read_text())
    # compact behavior: per set/ck/family selected fields
    beh = {}
    for s in bh:
        beh[s] = {}
        for ck in bh[s]:
            beh[s][ck] = {f: {k: v[k] for k in ("n", "outcomes", "strict_rate", "truncation_rate", "median_tokens", "median_tokens_strict",
                                                "median_tokens_fail", "median_code_lines_strict", "method_share", "method_strict_rate",
                                                "tags", "invalid_exec_errors")} for f, v in bh[s][ck].items()}
    for s in sv:
        sv[s].pop("case_k", None)
    data = {
        "_about": {
            "generated": "2026-10-12, CPU-only read of campaign artifacts; scripts in pipeline/analysis/",
            "definitions": {
                "strict": "verifier validity == 1 and range-normalized trajectory error <= 0.01 (1%)",
                "anchored_reward": "R = validity * (0.5*s(e) + 0.5*[e<=0.01]), s = clip(log(t/e)/log(t/0.01),0,1), t = best trivial-predictor error",
                "pass1_equal_family": "mean over families of per-family strict rate (fresh: 6 families, 288 cases x 8; held-out: 10 families, 135 cases x 8 after excluding 2 trivial cases)",
                "residual_best_of_8": "per case keep the valid response with the smallest candidate_residual (PDE residual of the answer; needs no reference solution), score it strict",
                "states": "never = 0/8 strict, sometimes = 1-5/8, reliable = >=6/8",
                "method_labels": "static regex labels on comment-stripped, verifier-extracted code; spot-checked, not exhaustive",
            },
            "checkpoints_available": {"fresh": available("fresh"), "heldout": available("heldout")},
            "phase_c_training_source": "Updates 225-256 come from the run's rollout records (training-rollouts-ext256/records.jsonl); their training verifier expired before its terminal export, so validity is unavailable for that phase.",
        },
        "training": {"per_update": tr["per_update"], "phases": tr["phases"], "ledger_utc": ledger(), "family_blocks16": tr["family_blocks16"],
                     "per_family_update": tr["per_family_update"], "stability": stability()},
        "difficulty_controlled": {
            "fixed_eval_pass1": {s: {ck: {"step": CK_STEP[ck], **{k: sv[s]["selectors"][ck][k] for k in ("pass1", "pass8", "residual_best_of_8")}}
                                     for ck in sv[s]["checkpoints"]} for s in sv},
            "fixed_eval_by_family": {s: {ck: sv[s]["selectors"][ck]["by_family"] for ck in sv[s]["checkpoints"]} for s in sv},
            "same_cases_seen_twice": tr["repeat_exposure"],
            "screened_cases_vs_phase_start_screen": tr["screened_case_lift"],
            "fixed_mix_training_reward": core5(tr["per_family_update"]),
        },
        "solvable_sets": sv,
        "phase_start_screens": json.loads((W / "screens.json").read_text()),
        "behavior": {"by_set_ckpt_family": beh, "length_and_format_arc": length_arc(),
                     "transposed_output_crashes_replay": replay_transposed()},
        "darcy": darcy_bins(),
        "fft_axes_negative_transfer": fft_axes_habit(),
    }
    OUT.write_text(json.dumps(data, indent=1, default=float))
    print("wrote", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()

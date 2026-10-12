"""Per-update training signal for every phase, plus difficulty-controlled companions (read-only).

Training groups (16 responses per window case) are mapped to updates through each phase manifest:
update = start + (row - (start*8 mod 512)) // 8 + 1. Requests with id prefix "rlvp-" and anchored
reward mode are training rollouts; screening ("probe-") and parity rows are excluded.
Writes work/training.json.
"""
import collections
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import C, W  # noqa: E402

T = C / "train-inputs"
PHASES = [  # (run, start, verifier log, manifest, screen run at phase start)
    ("calib-train-base-p2-anchored", 0, "verifier-train-terminal-export", T / "train-ordered-p2.jsonl", None),
    ("calib-train-base-p2-anchored-cont64", 32, "verifier-cont-terminal-export", T / "train-ordered-p2.jsonl", None),
    ("calib-train-base-p2-anchored-ext96", 64, "verifier-cont-terminal-export", T / "train-ordered-p2.jsonl", None),
    ("calib-train-base-p2-anchored-ext128", 96, "verifier-ext128-terminal-export", T / "curriculum128/train-ordered-p2-curr128.jsonl", None),
    ("calib-train-base-p2-anchored-ext160", 128, "verifier-ext160-terminal-export", T / "curriculum160/train-ordered-p2-new160.jsonl", "calib-screen160-s128"),
    ("calib-train-base-p2-anchored-ext192", 160, "verifier-ext192-terminal-export", T / "curriculum192/train-ordered-p2-new192.jsonl", "calib-screen192-s160"),
    ("calib-train-base-p2-anchored-ext224", 192, "verifier-ext224-terminal-export", T / "curriculum224/train-ordered-p2-new224.jsonl", "calib-screen224-s192"),
    ("calib-train-base-p2-anchored-ext256", 224, "verifier-ext256-terminal-export", T / "curriculum256/train-ordered-p2-new256.jsonl", "calib-screen256-s224"),
]
# Phase C's training verifier expired before its terminal export; its attempts are read from the run's
# own rollout records instead (rollouts_to_records.py), which carry reward and strict but not validity.
ALT_LOGS = {"verifier-ext256-terminal-export": C / "training-rollouts-ext256" / "records.jsonl"}
ANNOT = {
    0: "First pass over original cases 1-256 (8 families incl. NS2D and shallow water). Starts from untuned base.",
    32: "First pass over original cases 257-512 (same 8-family mix, new cases).",
    64: "Second pass over cases 1-256 (same cases as updates 1-32, new sampling seeds).",
    96: "Second pass over cases 257-512, with 60 zero-signal slots (27 NS2D, 26 shallow water) swapped for high-signal RD2D/Darcy/advection/Burgers cases.",
    128: "224 never-seen cases weighted to hard families (RD2D 78, Darcy 56, Burgers 50, advection 32, RD1D 8, NS2D 32); 1-D candidates near-solved at step 128 (>=7/8) screened out.",
    160: "Phase A: RD2D 80, Darcy 64 (2 per update), advection 48, Burgers 48, stiff RD1D 16; NS2D, shallow water, diffusion-sorption and Burgers nu=0.001 removed; 1-D cases screened to 1-5/8 at step 160.",
    192: "Phase B: RD2D 120, advection 48 (hard initial shapes required), Burgers 48 (nu=0.01 required), Darcy cut to 32 by gate, RD1D 8; 14 confirmed-hard inputs re-exposed.",
    224: "Phase C: RD2D 130, Burgers 54, advection 38, plus three new linear families (hyperdiffusion 14, advection on non-unit domains 12, Airy 8); Darcy dropped (falsified gate).",
}


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def strict(r):
    if "strict" in r:
        return bool(r["strict"])
    e = r.get("trajectory_loss_reference_range")
    return r.get("validity") == 1.0 and e is not None and math.isfinite(e) and e <= 0.01


def screen_counts(run):
    if run is None:
        return {}
    p = C / "probe-results" / run / "groups.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.open():
        g = json.loads(line)
        out[g["physical_case_id"]] = (sum(strict(r["verifier"]) for r in g["responses"]), len(g["responses"]))
    return out


def main():
    per_update, per_fam_update, groups_all = [], [], {}
    screen_lift = []
    for run, start, vdir, manifest, screen in PHASES:
        log = C / vdir / "verifier.jsonl"
        if not log.exists() and vdir in ALT_LOGS:
            log = ALT_LOGS[vdir]
        if not log.exists():
            print("no log yet", run)
            continue
        rows = jl(manifest)
        order = {r["physical_case_id"]: i for i, r in enumerate(rows)}
        fam = {r["physical_case_id"]: r["family_id"] for r in rows}
        offset = (start * 8) % 512
        g = collections.defaultdict(list)
        for r in jl(log):
            if not str(r.get("request_id", "")).startswith("rlvp-") or r.get("reward_mode") != "anchored":
                continue
            g[r["case_id"]].append(r)
        sc = screen_counts(screen)
        upd = collections.defaultdict(list)
        for case, rs in g.items():
            gi = order.get(case)
            if gi is None:
                continue
            u = start + (gi - offset) // 8 + 1
            if not start < u <= start + 32:
                continue
            if len(rs) != 16:
                print("warn group size", run, case, len(rs))
            upd[u].append((case, fam[case], rs))
            k = sum(strict(r) for r in rs)
            ar = [float(r.get("anchored_reward") or 0) for r in rs]
            groups_all.setdefault(case, []).append({"phase_start": start, "update": u, "family": fam[case], "strict": k, "n": len(rs),
                                                    "valid": sum(r.get("validity") == 1.0 for r in rs) if all("validity" in r for r in rs) else None,
                                                    "mean_anchored": statistics.fmean(ar), "contrast": max(ar) > min(ar),
                                                    "screen": sc.get(case)})
        for u in sorted(upd):
            rs = [r for _, _, grp in upd[u] for r in grp]
            ar = [float(r.get("anchored_reward") or 0) for r in rs]
            fams = collections.Counter(f for _, f, _ in upd[u])
            useful = sum(1 for _, _, grp in upd[u] if len({float(r.get("anchored_reward") or 0) for r in grp}) > 1)
            per_update.append({"update": u, "phase_start": start, "run": run, "groups": len(upd[u]),
                               "mean_anchored_reward": round(statistics.fmean(ar), 4),
                               "strict_rate": round(statistics.fmean(1.0 if strict(r) else 0.0 for r in rs), 4),
                               "valid_rate": (round(statistics.fmean(1.0 if r.get("validity") == 1.0 else 0.0 for r in rs), 4)
                                              if all("validity" in r for r in rs) else None),
                               "useful_groups": useful, "family_mix": dict(fams),
                               "share_zero_signal_families": round(sum(fams[f] for f in ("incompressible_ns2d", "shallow_water2d")) / len(upd[u]), 3)})
            for f in fams:
                frs = [r for _, ff, grp in upd[u] if ff == f for r in grp]
                per_fam_update.append({"update": u, "family": f, "n": len(frs), "strict": sum(strict(r) for r in frs),
                                       "mean_anchored": round(statistics.fmean(float(r.get("anchored_reward") or 0) for r in frs), 4)})
            # screened-case lift: training strict on screened cases vs the phase-start screen on the same cases
            pairs = [(sum(strict(r) for r in grp) / len(grp), sc[c][0] / sc[c][1]) for c, f, grp in upd[u] if c in sc]
            if pairs:
                screen_lift.append({"update": u, "phase_start": start, "cases": len(pairs),
                                    "train_strict": round(statistics.fmean(p[0] for p in pairs), 4),
                                    "screen_strict_at_phase_start": round(statistics.fmean(p[1] for p in pairs), 4)})
    # family x 16-update blocks
    blocks = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in per_fam_update:
        b = (r["update"] - 1) // 16
        blocks[r["family"]][b][0] += r["strict"]
        blocks[r["family"]][b][1] += r["n"]
    fam_blocks = {f: {f"{16 * b + 1}-{16 * b + 16}": {"strict": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)} for b, v in sorted(d.items())}
                  for f, d in blocks.items()}
    # same cases seen twice (difficulty held fixed): phase 1 vs phase 3, phase 2 vs phase 4
    repeat = {}
    for a, b in ((0, 64), (32, 96)):
        pairs = [(gs[0], gs[1]) for gs in ([x for x in v if x["phase_start"] in (a, b)] for v in groups_all.values())
                 if len(gs) == 2 and gs[0]["phase_start"] == a and gs[1]["phase_start"] == b]
        byf = collections.defaultdict(lambda: [0, 0, 0, 0])
        for x, y in pairs:
            s = byf[x["family"]]
            s[0] += x["strict"]; s[1] += x["n"]; s[2] += y["strict"]; s[3] += y["n"]
        repeat[f"updates {a + 1}-{a + 32} vs {b + 1}-{b + 32}"] = {
            "cases": len(pairs),
            "first_exposure_strict": round(sum(x["strict"] for x, _ in pairs) / sum(x["n"] for x, _ in pairs), 4),
            "second_exposure_strict": round(sum(y["strict"] for _, y in pairs) / sum(y["n"] for _, y in pairs), 4),
            "first_mean_anchored": round(statistics.fmean(x["mean_anchored"] for x, _ in pairs), 4),
            "second_mean_anchored": round(statistics.fmean(y["mean_anchored"] for _, y in pairs), 4),
            "by_family": {f: {"first": round(v[0] / v[1], 4), "second": round(v[2] / v[3], 4), "cases": v[1] // 16} for f, v in sorted(byf.items())}}
    # phase summaries
    ph = []
    for run, start, *_ in PHASES:
        us = [r for r in per_update if r["phase_start"] == start]
        if not us:
            continue
        mix = collections.Counter()
        for r in us:
            mix.update(r["family_mix"])
        ph.append({"run": run, "first": start + 1, "last": start + 32, "updates_logged": len(us), "annotation": ANNOT[start],
                   "mean_anchored_reward": round(statistics.fmean(r["mean_anchored_reward"] for r in us), 4),
                   "strict_rate": round(statistics.fmean(r["strict_rate"] for r in us), 4),
                   "first8_reward": round(statistics.fmean(r["mean_anchored_reward"] for r in us[:8]), 4),
                   "last8_reward": round(statistics.fmean(r["mean_anchored_reward"] for r in us[-8:]), 4),
                   "family_mix_groups": dict(mix), "mean_useful_groups": round(statistics.fmean(r["useful_groups"] for r in us), 2)})
    out = {"per_update": per_update, "per_family_update": per_fam_update, "family_blocks16": fam_blocks,
           "screened_case_lift": screen_lift, "repeat_exposure": repeat, "phases": ph}
    (W / "training.json").write_text(json.dumps(out, indent=1))
    for p in ph:
        print(p["first"], p["last"], p["updates_logged"], p["mean_anchored_reward"], p["strict_rate"], p["first8_reward"], p["last8_reward"], p["mean_useful_groups"], p["family_mix_groups"])
    print(json.dumps(repeat, indent=1))
    by = collections.defaultdict(list)
    for s in screen_lift:
        by[s["phase_start"]].append(s)
    for k, v in by.items():
        print("lift", k, round(statistics.fmean(x["train_strict"] for x in v), 3), round(statistics.fmean(x["screen_strict_at_phase_start"] for x in v), 3),
              "first8", round(statistics.fmean(x["train_strict"] - x["screen_strict_at_phase_start"] for x in v[:8]), 3),
              "last8", round(statistics.fmean(x["train_strict"] - x["screen_strict_at_phase_start"] for x in v[-8:]), 3))


if __name__ == "__main__":
    main()

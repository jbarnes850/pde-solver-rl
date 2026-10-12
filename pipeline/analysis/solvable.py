"""Solvable set, reliability histograms, transitions and residual-selected best-of-8 on the fixed eval sets.

Per checkpoint and set: per-case strict count k of 8; states never (k=0), sometimes (1-5), reliable (k>=6);
first checkpoint at which each case is solved / reliable; state transitions between consecutive
checkpoints; pass@1, pass@8 and a reference-free selector (keep the valid answer with the smallest
candidate_residual) scored strict, all equal-family and pooled. Writes work/solvable.json.
"""
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CK_STEP, W, available, by_case, load  # noqa: E402


def state(k):
    return "never" if k == 0 else ("reliable" if k >= 6 else "sometimes")


def eqfam(d):
    fams = sorted(d)
    return round(100 * statistics.fmean(statistics.fmean(d[f]) for f in fams), 2)


def main():
    out = {}
    for s in ("fresh", "heldout"):
        cks = available(s)
        K, fam = {}, {}
        sel = {}
        for ck in cks:
            cases = by_case(load(s, ck, keep_code=False))
            K[ck] = {c: sum(r["outcome"] == "strict" for r in rs) for c, rs in cases.items()}
            for c, rs in cases.items():
                fam[c] = rs[0]["family"]
            # selectors
            p1, p8, rsel, rsel_any = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
            for c, rs in cases.items():
                f = rs[0]["family"]
                k = K[ck][c]
                p1[f].append(k / len(rs))
                p8[f].append(1.0 if k else 0.0)
                valid = [r for r in rs if r["validity"] == 1.0 and r["cand_res"] is not None]
                pick = min(valid, key=lambda r: r["cand_res"]) if valid else None
                rsel[f].append(1.0 if (pick and pick["outcome"] == "strict") else 0.0)
            sel[ck] = {"pass1": eqfam(p1), "pass8": eqfam(p8), "residual_best_of_8": eqfam(rsel),
                       "by_family": {f: {"pass1": round(100 * statistics.fmean(p1[f]), 1), "pass8": round(100 * statistics.fmean(p8[f]), 1),
                                         "residual_best_of_8": round(100 * statistics.fmean(rsel[f]), 1)} for f in sorted(p1)}}
        cases = sorted(K[cks[0]])
        fams = sorted(set(fam.values()))
        hist = {ck: [sum(1 for c in cases if K[ck][c] == k) for k in range(9)] for ck in cks}
        hist_fam = {ck: {f: [sum(1 for c in cases if fam[c] == f and K[ck][c] == k) for k in range(9)] for f in fams} for ck in cks}
        states = {ck: dict(collections.Counter(state(K[ck][c]) for c in cases)) for ck in cks}
        states_fam = {ck: {f: dict(collections.Counter(state(K[ck][c]) for c in cases if fam[c] == f)) for f in fams} for ck in cks}
        # transitions between consecutive checkpoints
        trans = {}
        for a, b in zip(cks, cks[1:]):
            t = collections.Counter(f"{state(K[a][c])}->{state(K[b][c])}" for c in cases)
            trans[f"{a}->{b}"] = dict(t)
        # first solved / first reliable, cumulative ever-solved, lost
        first_solved = {}
        first_rel = {}
        for c in cases:
            fs = next((ck for ck in cks if K[ck][c] > 0), None)
            fr = next((ck for ck in cks if K[ck][c] >= 6), None)
            first_solved[c], first_rel[c] = fs, fr
        ever = {ck: sum(1 for c in cases if any(K[x][c] > 0 for x in cks[:cks.index(ck) + 1])) for ck in cks}
        newly = {ck: sum(1 for c in cases if first_solved[c] == ck) for ck in cks}
        newly_rel = {ck: sum(1 for c in cases if first_rel[c] == ck) for ck in cks}
        # among cases first solved at checkpoint X: their k at that checkpoint and at the last checkpoint
        follow = {}
        for ck in cks[1:]:
            cs = [c for c in cases if first_solved[c] == ck]
            if cs:
                follow[ck] = {"n": len(cs), "k_at_first": statistics.fmean(K[ck][c] for c in cs),
                              "k_at_last": statistics.fmean(K[cks[-1]][c] for c in cs),
                              "reliable_at_last": sum(K[cks[-1]][c] >= 6 for c in cases if c in cs)}
        # regressions: cases solved earlier but 0 at last checkpoint; cases reliable earlier but not at last
        lost = [c for c in cases if any(K[x][c] > 0 for x in cks[:-1]) and K[cks[-1]][c] == 0]
        # strict count k decomposition of the pass@1 gain: from cases at k=0 at base vs cases already >0
        base, last = cks[0], cks[-1]
        gain_from_new = sum(K[last][c] - K[base][c] for c in cases if K[base][c] == 0)
        gain_from_old = sum(K[last][c] - K[base][c] for c in cases if K[base][c] > 0)
        # per-step decomposition: change in total strict between consecutive checkpoints from cases newly solved vs already solved
        decomp = {}
        for a, b in zip(cks, cks[1:]):
            d_new = sum(K[b][c] for c in cases if K[a][c] == 0)
            d_old = sum(K[b][c] - K[a][c] for c in cases if K[a][c] > 0)
            decomp[f"{a}->{b}"] = {"strict_from_previously_unsolved": d_new, "net_change_on_previously_solved": d_old,
                                   "newly_solved_cases": sum(1 for c in cases if K[a][c] == 0 and K[b][c] > 0),
                                   "newly_unsolved_cases": sum(1 for c in cases if K[a][c] > 0 and K[b][c] == 0)}
        out[s] = {"checkpoints": cks, "steps": [CK_STEP[c] for c in cks], "n_cases": len(cases), "selectors": sel,
                  "hist_k": hist, "hist_k_by_family": hist_fam, "states": states, "states_by_family": states_fam,
                  "transitions": trans, "ever_solved_cumulative": ever, "first_solved_at": newly, "first_reliable_at": newly_rel,
                  "followup_of_newly_solved": follow, "lost_at_last": {"n": len(lost), "cases": [(c, fam[c], [K[x][c] for x in cks]) for c in lost]},
                  "gain_decomposition_base_to_last": {"from_base_zero_cases": gain_from_new, "from_base_nonzero_cases": gain_from_old},
                  "step_decomposition": decomp,
                  "case_k": {c: {"family": fam[c], "k": [K[ck][c] for ck in cks]} for c in cases}}
        print("==", s, cks)
        for ck in cks:
            print(f"  {ck:5s} states {states[ck]}  hist {hist[ck]}  sel {sel[ck]['pass1']}/{sel[ck]['residual_best_of_8']}/{sel[ck]['pass8']}  ever {ever[ck]} new {newly[ck]} newrel {newly_rel[ck]}")
        print("  decomposition", json.dumps(decomp))
        print("  base->last gain from base-zero", gain_from_new, "from base-nonzero", gain_from_old, "lost", len(lost))
        print("  follow", json.dumps(follow))
    (W / "solvable.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

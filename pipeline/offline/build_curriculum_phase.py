"""Curriculum for one training phase: a new 512-row manifest for any start step.

Generalizes build_curriculum160.py to any start step S. The collector reads rows (S*8 + k) mod 512,
so the active window sits at rows 256-511 for S = 160 and 224, and at rows 0-255 for S = 192 and 256.
The unread half holds the previous phase's window (cases already trained, not read this phase).

Phase A (updates 161-192) window, 256 rows, all new cases (role rl_train_20261012c192):
  - reaction_diffusion2d 80, unscreened (regime-determined), noise initial conditions, 32x32 : 48x48 ~ 56 : 44;
  - darcy2d 64 (2 per update), unscreened, spread evenly over the 30 (grid, field kind, beta) strata, with
    the three prompt forms of the original training manifest (40 generic, 16 parameter, 8 parameter_ic) so
    the half-cell boundary is learned as a rule across inputs and phrasings, not as one prompt's template;
  - advection1d 48, burgers1d 48 (no nu = 0.001), reaction_diffusion1d 16 (stiff corner nu in {2, 5},
    nx = 128): screened with the phase's starting policy (8 samples). Keep 1/8 <= p <= 5/8 plus "alive
    zeros" (0/8 strict, >= 2 valid, mean anchored R >= 0.05); drop p >= 6/8 and dead zeros. Required strata
    first (advection: >= 16 windowed_absolute, >= 10 beta = 2; Burgers: >= 28 nu = 0.01, >= 8 nu = 1 at
    nx = 128), then the target difficulty mix (25% alive zeros, 40% 1-2/8, 35% 3-5/8), ranked by MaxRL
    weight mass within each bucket. Shortfalls go to RD2D.
  - NS2D, shallow water and diffusion-sorption: 0 rows.
Per-update composition alternates X = 2 Darcy + 3 RD2D + 1 adv + 1 Burgers + 1 RD1D and
Y = 2 Darcy + 2 RD2D + 2 adv + 2 Burgers (16 of each), with family order shuffled within an update.
Every row gets new seeds under a new namespace; occurrence carries prior exposures.
Later phases pass a spec JSON (6th argument) that overrides the quotas (Darcy set by gate G5), the
required strata, and adds re-exposure: at most `max` advection/Burgers inputs from the previous window whose
own 16-sample training group had 1-6 strict (a second exposure, never a third), ranked by MaxRL mass, which
replace new inputs of the same family one for one. Spec
phases spread each family's quota evenly over the updates instead of the X/Y pattern. Without a spec the
builder reproduces phase A exactly.
Usage: python build_curriculum_phase.py PHASE START_STEP SCREEN_GROUPS_JSONL PREV_MANIFEST PREV_PARENT_SCHEDULE [SPEC_JSON]
"""
import collections
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RLVP_ROOT", Path(__file__).resolve().parents[2]))  # repository root
sys.path[:0] = [os.environ.get("RLVP_SRC", str(ROOT / "src")), str(ROOT / "scripts/modal")]  # src/ contains the new families
from make_pilot_manifest import make_continuation_schedule, content_sha256  # noqa: E402
from rlvp.data import load_case_store  # noqa: E402
from rlvp.pdes.registry import get_environment  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
PRIVATE = Path(os.environ.get("RLVP_PRIVATE_DIR", ROOT / ".artifacts/private")) / "rlvp-calibration-20261010"
SEED = 20261012
QUOTA = {"darcy2d": 64, "reaction_diffusion2d": 80, "advection1d": 48, "burgers1d": 48, "reaction_diffusion1d": 16}
UPDATE_X = {"darcy2d": 2, "reaction_diffusion2d": 3, "advection1d": 1, "burgers1d": 1, "reaction_diffusion1d": 1}
UPDATE_Y = {"darcy2d": 2, "reaction_diffusion2d": 2, "advection1d": 2, "burgers1d": 2}
SCREENED = ("advection1d", "burgers1d", "reaction_diffusion1d")
DARCY_FORMS = ["generic"] * 40 + ["parameter"] * 16 + ["parameter_ic"] * 8
MIX = {"alive_zero": 0.25, "low": 0.40, "mid": 0.35}
TESTS = {"windowed_absolute": lambda q: q["input_kind"] == "windowed_absolute",
         "beta2": lambda q: float(q["beta"]) == 2.0,
         "nu0.01": lambda q: float(q["nu"]) == 0.01,
         "nu1_nx128": lambda q: float(q["nu"]) == 1.0 and int(q["nx"]) == 128}
REQUIRED = {"advection1d": [["windowed_absolute", 16], ["beta2", 10]], "burgers1d": [["nu0.01", 28], ["nu1_nx128", 8]]}
EXCLUDED_FAMILIES = {"incompressible_ns2d", "shallow_water2d", "diffusion_sorption1d"}
HELD_OUT = {"advection2d", "advection_diffusion1d", "advection_diffusion2d", "advection_reaction_diffusion1d",
            "allen_cahn2d", "cahn_hilliard1d", "darcy_reaction2d", "gray_scott2d", "heat1d", "kdv1d"}


def strict(v):
    return v["validity"] == 1 and v.get("trajectory_loss_reference_range") is not None \
        and v["trajectory_loss_reference_range"] <= 0.01


def mass(rewards):
    m = statistics.fmean(rewards)
    return sum(abs(x - m) for x in rewards) / (m + 0.01)


def params(case):
    p = dict(case.get("parameters") or {})
    for k in ("beta", "ic_kind", "nx", "nt"):
        if k in case:
            p[k] = case[k]
    p["nx"] = p.get("nx") or (case.get("grid_shape") or [None])[0]
    p["input_kind"] = case.get("input_kind") or case.get("ic_kind")
    return p


def bucket(s):
    if s["strict"] == 0:
        return "alive_zero" if s["valid"] >= 2 and s["mean_r"] >= 0.05 else "dead"
    if s["strict"] >= 6:
        return "solved"
    return "low" if s["strict"] <= 2 else "mid"


def pick_screened(rows, screen, store, family, k):
    """Required strata first, then the difficulty mix, mass-ranked within each bucket."""
    alive = [r for r in rows if r["physical_case_id"] in screen and bucket(screen[r["physical_case_id"]]) not in ("dead", "solved")]
    alive.sort(key=lambda r: (-screen[r["physical_case_id"]]["mass"], r["physical_case_id"]))
    p = {r["physical_case_id"]: params(store[r["physical_case_id"]]) for r in alive}
    required = [(TESTS[name], need) for name, need in REQUIRED.get(family, [])]
    chosen, shortfalls = [], {}
    for test, need in required:
        have = sum(test(p[r["physical_case_id"]]) for r in chosen)
        for r in alive:
            if have >= need or len(chosen) >= k:
                break
            if r not in chosen and test(p[r["physical_case_id"]]):
                chosen.append(r)
                have += 1
        if have < need:
            shortfalls[f"required_{need}"] = need - have
    targets = {b: round(k * f) for b, f in MIX.items()}
    for b in ("alive_zero", "low", "mid"):
        have = sum(bucket(screen[r["physical_case_id"]]) == b for r in chosen)
        for r in alive:
            if have >= targets[b] or len(chosen) >= k:
                break
            if r not in chosen and bucket(screen[r["physical_case_id"]]) == b:
                chosen.append(r)
                have += 1
    for r in alive:  # buckets short: fill by mass
        if len(chosen) >= k:
            break
        if r not in chosen:
            chosen.append(r)
    return chosen[:k], shortfalls


def even_slots(quota, updates=32, per_update=8):
    """Largest-remainder spread of each family's quota over the updates, exactly per_update rows each."""
    fams = sorted(quota)
    acc = {f: 0.0 for f in fams}
    plan = []
    for u in range(updates):
        for f in fams:
            acc[f] += quota[f] / updates
        take = {f: int(acc[f]) for f in fams}
        while sum(take.values()) < per_update:
            f = max(fams, key=lambda x: (acc[x] - take[x], x))
            take[f] += 1
        while sum(take.values()) > per_update:
            f = min((x for x in fams if take[x] > 0), key=lambda x: (acc[x] - take[x], x))
            take[f] -= 1
        for f in fams:
            acc[f] -= take[f]
        plan.append(take)
    return plan


def main():
    global QUOTA, REQUIRED, DARCY_FORMS
    phase, start = sys.argv[1], int(sys.argv[2])
    screen_path, prev_manifest, prev_parent = Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    spec = json.loads(Path(sys.argv[6]).read_text()) if len(sys.argv) > 6 else None
    if spec:
        QUOTA = dict(spec["quota"])
        REQUIRED = spec.get("required", REQUIRED)
        d = QUOTA["darcy2d"]
        DARCY_FORMS = ["generic"] * (d - d // 4 - d // 8) + ["parameter"] * (d // 4) + ["parameter_ic"] * (d // 8)
    grid = json.loads((HERE / "prompts/user-grid-lines-p2.json").read_text())
    extra = HERE / "prompts/user-grid-lines-p2-extra.json"  # new training families, after the step-224 merge
    if extra.exists():
        grid.update(json.loads(extra.read_text()))
    store = {}  # parameters for this phase's candidates and any re-exposed inputs from earlier phases
    for path in sorted(PRIVATE.glob("rl_train_20261012c*.cases.json")):
        store.update(json.loads(path.read_text())["cases"])
    cases = load_case_store(PRIVATE / f"rl_train_20261012c{phase}.cases.json")
    cand = [json.loads(l) for l in (HERE / f"train-inputs/new{phase}/new-train-candidates.jsonl").read_text().splitlines() if l.strip()]
    screen = {}
    for line in screen_path.read_text().splitlines():
        g = json.loads(line)
        vs = [r["verifier"] for r in g["responses"]]
        rs = [float(v.get("anchored_reward") or 0.0) for v in vs]
        screen[g["physical_case_id"]] = {"strict": sum(strict(v) for v in vs), "valid": sum(v["validity"] == 1 for v in vs),
                                         "n": len(vs), "mean_r": statistics.fmean(rs), "mass": mass(rs)}

    prev = [json.loads(l) for l in prev_manifest.read_text().splitlines() if l.strip()]
    prev_parent_rows = {json.loads(l)["physical_case_id"]: json.loads(l) for l in prev_parent.read_text().splitlines() if l.strip()}
    prev_offset = ((start - 32) * 8) % 512
    reexposed, reexpose_info = [], {}
    if spec and spec.get("reexpose"):
        rx = spec["reexpose"]
        prev_window = prev[prev_offset:prev_offset + 256]
        prev_ids = {r["physical_case_id"]: r for r in prev_window}
        grp = collections.defaultdict(list)
        for line in Path(rx["log"]).read_text().splitlines():
            r = json.loads(line)
            if r.get("request_id", "").startswith("parity") or r.get("reward_mode") != "anchored" or r["case_id"] not in prev_ids:
                continue
            grp[r["case_id"]].append(r)
        cands = []
        for c, rs in grp.items():
            k = sum(strict(r) for r in rs)
            fam = prev_ids[c]["family_id"]
            if fam in rx["families"] and len(rs) == 16 and rx["band"][0] <= k <= rx["band"][1] \
                    and int(prev_parent_rows.get(c, {}).get("occurrence", 0)) == 0:
                cands.append((mass([float(r.get("anchored_reward") or 0.0) for r in rs]), c, k))
        cands.sort(key=lambda x: (-x[0], x[1]))
        reexposed = [prev_ids[c] for _, c, _ in cands[:rx["max"]]]
        reexpose_info = {"candidates": len(cands), "chosen": len(reexposed),
                         "families": dict(collections.Counter(r["family_id"] for r in reexposed))}
        for r in reexposed:  # re-exposed inputs count toward their family's quota
            QUOTA[r["family_id"]] -= 1
    by_fam = collections.defaultdict(list)
    for r in cand:
        by_fam[r["family_id"]].append(r)
    chosen, receipt_pick, shortfall = {}, {}, 0
    for fam in SCREENED:
        rows, short = pick_screened(by_fam[fam], screen, store, fam, QUOTA[fam])
        chosen[fam] = rows
        shortfall += QUOTA[fam] - len(rows)
        receipt_pick[fam] = {"chosen": len(rows), "quota": QUOTA[fam], "required_shortfalls": short,
                             "buckets": dict(collections.Counter(bucket(screen[r["physical_case_id"]]) for r in rows))}
    for fam in [f for f in QUOTA if f not in SCREENED and f not in ("reaction_diffusion2d", "darcy2d")]:
        # new training families (step-224 merge onward): screened band if screened, else generator order
        rows = sorted(by_fam[fam], key=lambda r: r["family_case_ordinal"])
        if any(r["physical_case_id"] in screen for r in rows):
            rows, short = pick_screened(rows, screen, store, fam, QUOTA[fam])
        else:
            rows, short = rows[:QUOTA[fam]], {}
        chosen[fam] = rows
        shortfall += QUOTA[fam] - len(rows)
        receipt_pick[fam] = {"chosen": len(rows), "quota": QUOTA[fam], "required_shortfalls": short}
    # RD2D: 32x32 : 48x48 ~ 56 : 44, spread over (du, dv, k, time count), plus any screened shortfall.
    rd = sorted(by_fam["reaction_diffusion2d"], key=lambda r: r["family_case_ordinal"])
    want = QUOTA["reaction_diffusion2d"] + shortfall
    n32 = round(want * 0.56)
    g = lambda r: params(store[r["physical_case_id"]])["nx"]
    rd32 = [r for r in rd if g(r) == 32]
    rd48 = [r for r in rd if g(r) == 48]
    chosen["reaction_diffusion2d"] = (rd32[:n32] + rd48[:want - n32])
    if len(chosen["reaction_diffusion2d"]) < want:  # grid supply short: top up from either grid
        rest = [r for r in rd if r not in chosen["reaction_diffusion2d"]]
        chosen["reaction_diffusion2d"] += rest[:want - len(chosen["reaction_diffusion2d"])]
    assert len(chosen["reaction_diffusion2d"]) == want, ("RD2D supply", len(chosen["reaction_diffusion2d"]), want)
    # Darcy: round-robin over the 30 strata so every (grid, field kind, beta) appears about equally.
    strata = collections.defaultdict(list)
    for r in sorted(by_fam["darcy2d"], key=lambda r: r["family_case_ordinal"]):
        q = params(store[r["physical_case_id"]])
        strata[(q["nx"], q["field_kind"], float(q["beta"]))].append(r)
    keys = sorted(strata)
    darcy = []
    while len(darcy) < QUOTA["darcy2d"] and any(strata[s] for s in keys):
        for s in keys:
            if strata[s] and len(darcy) < QUOTA["darcy2d"]:
                darcy.append(strata[s].pop(0))
    assert len(darcy) == QUOTA["darcy2d"], ("Darcy supply", len(darcy))
    chosen["darcy2d"] = darcy

    rng = random.Random(SEED + start)
    forms = DARCY_FORMS[:]
    rng.shuffle(forms)
    env_d = get_environment("darcy2d")

    def as_train(r, form="generic"):
        r = dict(r)
        if form != "generic":
            r["prompt"] = env_d.prompt(cases[r["physical_case_id"]], form=form)
            r["prompt_form"] = form
        r["prompt"] = r["prompt"].rstrip("\n") + "\n\n" + grid[r["family_id"]] + "\n"
        r["prompt_sha256"] = hashlib.sha256(r["prompt"].encode()).hexdigest()
        r["role"], r["split"] = "rl_train", "train"
        r["row_id"] = "row:" + hashlib.sha256(json.dumps({k: v for k, v in r.items() if k != "row_id"}, sort_keys=True).encode()).hexdigest()
        return r

    pools = {f: [as_train(r, forms[i]) if f == "darcy2d" else as_train(r) for i, r in enumerate(rows)] for f, rows in chosen.items()}
    if reexposed:
        pools["reexposed"] = [dict(r) for r in reexposed]  # already rl_train rows with the P2 prompt
    for f in pools:
        rng.shuffle(pools[f])
    window = []
    slot_plan = None
    if spec:
        q = {f: len(v) for f, v in pools.items()}
        slot_plan = even_slots(q)
    for u in range(32):
        comp = slot_plan[u] if slot_plan else (UPDATE_X if u % 2 == 0 else UPDATE_Y)
        upd = []
        for f, n in comp.items():
            take = min(n, len(pools[f]))
            upd += [pools[f].pop() for _ in range(take)]
        while len(upd) < 8:  # a screened family ran short: its slot goes to RD2D (pool enlarged by the shortfall)
            upd.append(pools["reaction_diffusion2d"].pop())
        rng.shuffle(upd)
        window += upd
    assert len(window) == 256 and all(not v for v in pools.values()), {f: len(v) for f, v in pools.items()}

    in_window = {r["physical_case_id"] for r in window}
    # the previous phase's window (trained, not read now), minus re-exposed rows, topped up from its filler
    filler = [r for r in prev[prev_offset:prev_offset + 256] if r["physical_case_id"] not in in_window]
    filler += [r for r in prev if r["physical_case_id"] not in in_window
               and r not in filler][:256 - len(filler)]
    assert not in_window & {r["physical_case_id"] for r in filler}
    offset = (start * 8) % 512
    rows = (filler + window) if offset == 256 else (window + filler)
    assert len(rows) == 512 and len({r["physical_case_id"] for r in rows}) == 512
    act = rows[offset:offset + 256]
    assert [r["physical_case_id"] for r in act] == [r["physical_case_id"] for r in window]
    fams = collections.Counter(r["family_id"] for r in act)
    assert not (set(fams) & (EXCLUDED_FAMILIES | HELD_OUT)), fams
    assert all(r["family_id"] != "burgers1d" or float(params(store[r["physical_case_id"]])["nu"]) != 0.001 for r in act)
    if not spec:
        for u in range(32):
            assert sum(r["family_id"] == "darcy2d" for r in act[u * 8:(u + 1) * 8]) == 2
    assert sum(r["family_id"] == "darcy2d" for r in act) == QUOTA["darcy2d"]
    assert len(act) == 256

    out = Path(os.environ.get("RLVP_CURRICULUM_OUT", HERE / f"train-inputs/curriculum{phase}"))  # dry runs: scratch dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / f"train-ordered-p2-new{phase}.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows))
    mhash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    parent = []
    for k, r in enumerate(rows):
        base = prev_parent_rows.get(r["physical_case_id"])
        entry = {"family_id": r["family_id"], "physical_case_id": r["physical_case_id"], "group_index": k,
                 "occurrence": int(base["occurrence"]) + 1 if base else 0, "optimizer_update": k // 8 + 1,
                 "is_pilot": False, "is_proof": False, "prompt_sha256": r["prompt_sha256"],
                 "ordered_manifest_sha256": mhash, "response_seeds": [], "response_uids": []}
        entry["group_id"] = content_sha256({"case": r["physical_case_id"], "index": k, "manifest": mhash})
        parent.append(entry)
    parent_path = out / "parent-schedule.jsonl"
    parent_path.write_text("".join(json.dumps(p, sort_keys=True, separators=(",", ":")) + "\n" for p in parent))
    namespace = f"rlvp-calib-20261012-g16-p2-new{phase}-v1"
    sched = make_continuation_schedule(ordered_manifest=manifest, parent_schedule=parent_path,
                                       output_dir=out / "schedule", namespace=namespace)
    darcy_strata = collections.Counter((params(store[r["physical_case_id"]])["nx"], params(store[r["physical_case_id"]])["field_kind"])
                                       for r in act if r["family_id"] == "darcy2d")
    summary = {"rule": __doc__.strip().splitlines()[0], "phase": phase, "start_step": start, "window_rows": [offset, offset + 255],
               "window_family_counts": dict(fams), "screened": receipt_pick, "reexposed": reexpose_info,
               "spec": spec,
               "rd2d_grids": dict(collections.Counter(params(store[r["physical_case_id"]])["nx"] for r in act if r["family_id"] == "reaction_diffusion2d")),
               "darcy_grid_fieldkind_counts": {f"{a}-{b}": v for (a, b), v in sorted(darcy_strata.items())},
               "darcy_prompt_forms": dict(collections.Counter(r["prompt_form"] for r in act if r["family_id"] == "darcy2d")),
               "generic_share": sum(r["prompt_form"] == "generic" for r in act) / 256,
               "distinct_prompts_by_family": {f: len({r["prompt_sha256"] for r in act if r["family_id"] == f}) for f in fams},
               "filler_rows_from_previous_window": len(filler),
               "manifest_sha256": mhash, "schedule_sha256": sched["schedule_sha256"], "namespace": namespace,
               "screen_sha256": hashlib.sha256(screen_path.read_bytes()).hexdigest(),
               "candidates_sha256": hashlib.sha256((HERE / f"train-inputs/new{phase}/new-train-candidates.jsonl").read_bytes()).hexdigest()}
    (out / "curriculum-receipt.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

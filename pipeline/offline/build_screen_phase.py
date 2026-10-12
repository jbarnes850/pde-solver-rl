"""Screening probe for a training phase: the phase's starting policy, 8 samples, screened families only.

Generalizes build_screen160.py. Advection, Burgers and RD1D candidates from
train-inputs/new{PHASE}/new-train-candidates.jsonl are filtered by regime (Burgers nu = 0.001 is out:
no independent solver passes at the 1% bar; RD1D keeps only the stiff corner nu in {2, 5} at nx = 128)
and written as a prompt_dev manifest with the P2 prompt (the training prompt). RD2D and Darcy are not
screened: their per-case difficulty is set by regime (overdispersion 1.22 and 0.84 in phase 129-160).
build_curriculum_phase.py later keeps 1/8 <= p <= 5/8 plus alive zeros and drops p >= 6/8.
Usage: python build_screen_phase.py PHASE START_STEP RESUME_CHECKPOINT BUNDLE_CASES_SHA256 SOLVER_DIGEST
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RLVP_ROOT", Path(__file__).resolve().parents[2]))
PRIVATE = Path(os.environ.get("RLVP_PRIVATE_DIR", ROOT / ".artifacts/private")) / "rlvp-calibration-20261010"
SCREENED = ("advection1d", "burgers1d", "reaction_diffusion1d")
CAPS = {"advection1d": 120, "burgers1d": 160, "reaction_diffusion1d": 80}  # ~360 rows, about 70 min at concurrency 64
EXTRA = ("advection1d_domain", "airy1d", "hyperdiffusion1d")  # new families, screened when present (cap 32 each)
CAPS.update({f: 32 for f in EXTRA})


def params(case):
    p = dict(case.get("parameters") or {})
    for k in ("beta", "ic_kind", "nx", "nt"):
        if k in case:
            p[k] = case[k]
    p["nx"] = p.get("nx") or (case.get("grid_shape") or [None])[0]
    p["input_kind"] = case.get("input_kind") or case.get("ic_kind")
    return p


def eligible(family, p):
    if family in EXTRA:
        return True
    if family == "burgers1d":
        return float(p["nu"]) != 0.001
    if family == "reaction_diffusion1d":
        return float(p["nu"]) in (2.0, 5.0) and int(p["nx"]) == 128
    return family == "advection1d"


def main():
    phase, start, ckpt, cases_sha, digest = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    src = HERE / f"train-inputs/new{phase}"
    grid = json.loads((HERE / "prompts/user-grid-lines-p2.json").read_text())
    extra = HERE / "prompts/user-grid-lines-p2-extra.json"  # new training families, after the step-224 merge
    if extra.exists():
        grid.update(json.loads(extra.read_text()))
    store = json.loads((PRIVATE / f"rl_train_20261012c{phase}.cases.json").read_text())["cases"]
    rows = [json.loads(l) for l in (src / "new-train-candidates.jsonl").read_text().splitlines() if l.strip()]
    out = src / "screen-p2.jsonl"
    counts = {}
    keep = [r for r in rows if r["family_id"] in SCREENED + EXTRA and eligible(r["family_id"], params(store[r["physical_case_id"]]))]
    # Burgers nu = 0.01 (the frontier regime, 22% strict at step 160) goes first; then generator order.
    keep.sort(key=lambda r: ((SCREENED + EXTRA).index(r["family_id"]),
                             float(params(store[r["physical_case_id"]]).get("nu", 0)) != 0.01 if r["family_id"] == "burgers1d" else False,
                             r["family_case_ordinal"]))
    with out.open("w") as fh:
        for r in keep:
            f = r["family_id"]
            if counts.get(f, 0) >= CAPS[f]:
                continue
            r = dict(r)
            r["prompt"] = r["prompt"].rstrip("\n") + "\n\n" + grid[f] + "\n"
            r["prompt_sha256"] = hashlib.sha256(r["prompt"].encode()).hexdigest()
            r["role"] = "prompt_dev"
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
            counts[f] = counts.get(f, 0) + 1
    y = yaml.safe_load((HERE / "probe-inputs/configs/promptdev-p2.yaml").read_text())
    y["id"] = f"rlvp-calib-20261012-screen{phase}-p2"
    y["evaluation"]["seed_namespace"] = f"rlvp-calib-20261012-screen{phase}-v1"
    y["evaluation"].pop("seed_manifest_sha256", None)
    cfg = src / f"screen{phase}-p2.yaml"
    cfg.write_text(yaml.safe_dump(y, sort_keys=False, width=200))
    n = sum(counts.values())
    t = json.loads((HERE / "trials/calib-probe-base-p2.json").read_text())
    t.update({"run_id": f"calib-screen{phase}-s{start}", "resume_checkpoint": ckpt, "weight_version": start,
              "case_store_sha256": cases_sha, "solver_image_digest": digest,
              "evaluation_config": f"/inputs/calib-20261010/new{phase}/screen{phase}-p2.yaml",
              "evaluation_manifest": f"/inputs/calib-20261010/new{phase}/screen-p2.jsonl",
              "max_seconds": 6000, "max_estimated_dollars": 29.3})
    (HERE / f"trials/calib-screen{phase}-s{start}.json").write_text(json.dumps(t, indent=2, sort_keys=True))
    print(n, "screen rows", counts, hashlib.sha256(out.read_bytes()).hexdigest()[:16], hashlib.sha256(cfg.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()

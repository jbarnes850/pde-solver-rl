"""Generate new cases for selected seen families with per-family counts under a new role namespace.

Copy of generate_fresh_cases.py with a per-family count map (argument 3 is JSON, e.g.
'{"darcy2d": 56}'); used for the step-129-160 training cases and the fresh-v3 test set, and
(held-out families allowed since 2026-10-12) for the held-out-v2 test set.

Seeds use the existing counter-based SHA-256 derivation with a new role, so no existing
case seed can recur. Each case is fingerprinted by its complete public invocation
(arrays, scalars, expected shape) and rejected if it matches any existing train,
validation, test, seen-evaluation or SFT case. Accepted cases must pass the existing
two-route reference gates. No policy outcomes are generated here.
"""
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RLVP_ROOT", Path(__file__).resolve().parents[2]))  # repository root
PRIVATE_DIR = Path(os.environ.get("RLVP_PRIVATE_DIR", ROOT / ".artifacts/private"))  # private case stores
sys.path.insert(0, os.environ.get("RLVP_SRC", str(ROOT / "src")))  # src/ already contains the three new families
import rlvp.manifests as _manifests  # noqa: E402
from rlvp.manifests import HELD_OUT_FAMILIES, SEEN_FAMILIES, _case_id, _stable_seed, canonical_json  # noqa: E402
EXTRA_TRAIN_FAMILIES = tuple(getattr(_manifests, "EXTRA_TRAIN_FAMILIES", ()))  # present after the step-224 merge
from rlvp.pdes.registry import get_environment  # noqa: E402
from rlvp.reference_gates import REFERENCE_RANGE_NRMSE_LIMITS  # noqa: E402
from rlvp.rewards import normalized_trajectory_loss  # noqa: E402
from rlvp.data import load_case_store  # noqa: E402

ROLE, SPLIT = sys.argv[1], sys.argv[2]
COUNTS = json.loads(sys.argv[3])
PER_FAMILY = COUNTS
PRIVATE = Path(sys.argv[4])
PUBLIC = Path(sys.argv[5])
EXISTING = sorted(glob.glob(str(PRIVATE_DIR / "rlvp-source-derived-v1/manifests/*.cases.json")))
EXTRA_EXISTING = [p for p in sys.argv[6:]]
MAX_TRIES = 4000  # finite discrete case spaces (e.g. 108 sorption cases) can be exhausted


def fingerprint(env, case):
    inv = env.invocation(case)
    h = hashlib.sha256()
    h.update(case.family_id.encode())
    h.update(repr(tuple(inv.expected_shape)).encode())
    for name in inv.argument_order:
        if name in inv.arrays:
            a = np.ascontiguousarray(np.asarray(inv.arrays[name], dtype=np.float64))
            h.update(name.encode() + repr(a.shape).encode() + a.tobytes())
        else:
            h.update(name.encode() + repr(float(inv.scalars[name])).encode())
    return h.hexdigest()


existing = {}
for path in EXISTING + EXTRA_EXISTING:
    for cid, case in load_case_store(Path(path)).items():
        existing[fingerprint(get_environment(case.family_id), case)] = (Path(path).name, cid)
print("existing fingerprints", len(existing), flush=True)

public_rows, cases, rejected = [], {}, []
for family in [f for f in SEEN_FAMILIES + HELD_OUT_FAMILIES + EXTRA_TRAIN_FAMILIES if f in COUNTS]:
    env = get_environment(family)
    ordinal, accepted = 0, 0
    while accepted < COUNTS[family] and ordinal < MAX_TRIES:
        seed = _stable_seed(role=ROLE, family_id=family, ordinal=ordinal)
        case = env.make_case(seed=seed, split=SPLIT)
        fp = fingerprint(env, case)
        reason = None
        if fp in existing:
            reason = f"duplicate_of:{existing[fp]}"
        elif case.case_id in cases:
            reason = "duplicate_case_id"
        else:
            primary = np.asarray(env.reference_primary(case), dtype=np.float64)
            audit = np.asarray(env.reference_audit(case), dtype=np.float64)
            gate = normalized_trajectory_loss(primary, audit, normalization="reference_range")
            if not (np.isfinite(primary).all() and gate <= REFERENCE_RANGE_NRMSE_LIMITS[family]):
                reason = f"reference_gate_failed:{gate:.3e}"
        if reason:
            rejected.append({"family_id": family, "ordinal": ordinal, "reason": reason})
        else:
            existing[fp] = (ROLE, case.case_id)
            prompt = env.prompt(case, form="generic")
            row = {
                "assumption_ids": ["A-SEED-001", "A-EVAL-OOD-001" if family in HELD_OUT_FAMILIES else "A-NEW-20261011"],
                "benchmark_id": "rlvp-source-derived-v1",
                "case_id": _case_id(role=ROLE, family_id=family, ordinal=ordinal, case_seed=seed),
                "case_seed": seed, "family_case_ordinal": ordinal, "family_id": family,
                "generator_version": str(getattr(case, "generator_version", "advection-source-derived-v1")),
                "physical_case_id": str(case.case_id), "prompt": prompt, "prompt_form": "generic",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "role": ROLE, "split": SPLIT,
                "public_input_fingerprint": fp,
            }
            row["row_id"] = "row:" + hashlib.sha256(canonical_json(row).encode()).hexdigest()
            public_rows.append(row)
            cases[case.case_id] = {**case.to_dict(), "family_id": family}
            accepted += 1
        ordinal += 1
    print(family, "accepted", accepted, "tried", ordinal, "exhausted" if accepted < COUNTS[family] else "", flush=True)

PRIVATE.parent.mkdir(parents=True, exist_ok=True)
PRIVATE.write_text(json.dumps({"benchmark_id": "rlvp-source-derived-v1", "cases": dict(sorted(cases.items())),
                               "manifest": ROLE, "schema": "rlvp.private-case-store.v1"}, indent=2, sort_keys=True) + "\n")
os.chmod(PRIVATE, 0o600)
PUBLIC.parent.mkdir(parents=True, exist_ok=True)
with PUBLIC.open("w") as fh:
    for row in public_rows:
        fh.write(canonical_json(row) + "\n")
receipt = {"role": ROLE, "split": SPLIT, "per_family": PER_FAMILY, "accepted": len(public_rows),
           "rejected": rejected, "private_sha256": hashlib.sha256(PRIVATE.read_bytes()).hexdigest(),
           "public_sha256": hashlib.sha256(PUBLIC.read_bytes()).hexdigest(),
           "existing_stores_checked": EXISTING + EXTRA_EXISTING, "outcomes_generated": 0}
PUBLIC.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
print(json.dumps({k: v for k, v in receipt.items() if k != "rejected"} | {"rejected": len(rejected)}))

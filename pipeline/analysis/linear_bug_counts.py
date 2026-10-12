"""Recurring spectral-program bugs on the four linear unseen families, per checkpoint (read-only).

Families: advection2d, advection_diffusion1d, advection_diffusion2d, heat1d (cases with best trivial error
above 1%; 62 cases x 8 answers = 496). Detectors from bugs.py, applied to the verifier-extracted code with
comments stripped: kscale (wavenumbers built for the wrong domain size) and fft1d_axes (a 1-D FFT called on
a multi-dimensional array with axes=). Writes linear-bug-counts.json.
"""
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bugs  # noqa: E402
from common import C, W  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rlvp.prompts import extract_python_code  # noqa: E402

TRIV = json.loads((C / "offline/heldout-trivial-errors.json").read_text())
LIN = {"advection2d", "advection_diffusion1d", "advection_diffusion2d", "heat1d"}
RUNS = {"base": "calib-heldout-base", "s160": "calib-heldout-ext160", "s192": "calib-heldout-ext192",
        "s224": "calib-heldout-ext224", "s256": "calib-heldout-ext256"}


def strict(v):
    e = v.get("trajectory_loss_reference_range")
    return v.get("validity") == 1.0 and e is not None and math.isfinite(e) and e <= 0.01


def main():
    out = {}
    for ck, run in RUNS.items():
        counts, n, ok = collections.Counter(), 0, 0
        for line in open(C / "probe-results" / run / "groups.jsonl"):
            g = json.loads(line)
            if g["family_id"] not in LIN or TRIV[g["physical_case_id"]] <= 0.01:
                continue
            for r in g["responses"]:
                n += 1
                lines = bugs.strip(extract_python_code(r["content"]))
                for name in ("kscale", "fft1d_axes"):
                    try:
                        counts[name] += bool(bugs.DET[name](lines))
                    except Exception:
                        pass
                ok += strict(r["verifier"])
        out[ck] = {"answers": n, "strict": ok, **{k: counts[k] for k in ("kscale", "fft1d_axes")}}
        print(ck, out[ck])
    (W / "linear-bug-counts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

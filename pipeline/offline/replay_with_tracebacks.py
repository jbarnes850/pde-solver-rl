"""Local diagnostic replay of existing programs with stderr capture.

Uses the unchanged DockerSolverExecutor.command flags (network none, read-only, 4 GiB,
2 CPUs, 30 s timeout, unprivileged user) and the original invocation staging, but in a
locally built arm64 image of containers/solver.Dockerfile (same NumPy 2.5.2 / SciPy 1.17.0).
This is diagnosis, not scoring: the scored verifier image was amd64 on Modal.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RLVP_ROOT", Path(__file__).resolve().parents[2]))  # repository root
sys.path.insert(0, str(ROOT / "src"))
from rlvp.data import load_case_store  # noqa: E402
from rlvp.executor import DockerSolverExecutor  # noqa: E402
from rlvp.pdes.registry import environment_for_case  # noqa: E402

IMAGE = "rlvp-solver:local-diag-arm64"
CASE_STORE = Path(os.environ.get("RLVP_REPLAY_CASE_STORE", str(ROOT / ".artifacts/modal/2026-10-09/verifier-bundle-v3/cases.json")))
EXC = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Exit|Interrupt))(?::\s*(.*))?$")


def last_exception(stderr: str) -> tuple[str | None, str | None, str | None]:
    lines = [ln for ln in stderr.splitlines() if ln.strip() and not ln.startswith("WARNING: Your kernel")]
    frame = None
    for ln in lines:
        if ln.strip().startswith('File "/work/candidate.py"'):
            frame = ln.strip()
    for ln in reversed(lines):
        m = EXC.match(ln.strip())
        if m:
            return m.group(1), (m.group(2) or "")[:300], frame
    return None, (lines[-1][:300] if lines else None), frame


def run_one(executor, cases, item):
    case = cases[item["case_id"]]
    inv = environment_for_case(case).invocation(case)
    name = f"rlvp-diag-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="rlvp-diag-") as raw:
        work = Path(raw)
        os.chmod(work, 0o777)
        (work / "candidate.py").write_text(item["code"] + "\n")
        np.savez(work / "input.npz", **inv.arrays,
                 **{k: np.asarray(v) for k, v in inv.scalars.items()})
        (work / "call.json").write_text(json.dumps({
            "argument_order": inv.argument_order, "array_names": sorted(inv.arrays),
            "scalar_names": sorted(inv.scalars)}, sort_keys=True) + "\n")
        (work / "output.npy").touch()
        for f, m in (("candidate.py", 0o444), ("input.npz", 0o444), ("call.json", 0o444), ("output.npy", 0o666)):
            os.chmod(work / f, m)
        t0 = time.monotonic()
        try:
            p = subprocess.run(executor.command(work, container_name=name), capture_output=True,
                               timeout=executor.timeout_seconds, check=False)
            rc, stderr, timed_out = p.returncode, p.stderr.decode("utf-8", "replace"), False
        except subprocess.TimeoutExpired:
            DockerSolverExecutor._force_remove(name)
            rc, stderr, timed_out = None, "", True
        dur = time.monotonic() - t0
        shape_ok = finite = None
        if not timed_out and rc == 0 and (work / "output.npy").stat().st_size > 0:
            try:
                out = np.load(work / "output.npy", allow_pickle=False)
                shape_ok = tuple(out.shape) == tuple(inv.expected_shape)
                finite = bool(shape_ok and np.isfinite(out).all())
                if not shape_ok:
                    stderr += f"\n[diag] returned shape {tuple(out.shape)} expected {tuple(inv.expected_shape)}"
            except Exception as exc:  # noqa: BLE001
                stderr += f"\n[diag] output load error {type(exc).__name__}"
    etype, emsg, frame = last_exception(stderr)
    return {"key": item["key"], "returncode": rc, "timed_out": timed_out, "seconds": round(dur, 3),
            "local_shape_ok": shape_ok, "local_finite": finite, "exception": etype,
            "message": emsg, "candidate_frame": frame, "stderr_tail": stderr[-1500:],
            "expected_shape": list(inv.expected_shape)}


def main():
    corpus, out_path, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    done = set()
    if out_path.exists():
        done = {json.loads(l)["key"] for l in open(out_path)}
    items = []
    for line in open(corpus):
        r = json.loads(line)
        key = f'{r["policy"]}|{r.get("row", r.get("response_index"))}|{r["case_id"]}|{r["candidate_sha256"][:16]}'
        if key in done:
            continue
        items.append({"key": key, "case_id": r["case_id"], "code": r["code"]})
    cases = load_case_store(CASE_STORE)
    executor = DockerSolverExecutor(image=IMAGE)
    lock = threading.Lock()
    with out_path.open("a") as fh, ThreadPoolExecutor(workers) as pool:
        for res in pool.map(lambda it: run_one(executor, cases, it), items):
            with lock:
                fh.write(json.dumps(res, sort_keys=True) + "\n")
                fh.flush()
    print("replayed", len(items))


if __name__ == "__main__":
    main()

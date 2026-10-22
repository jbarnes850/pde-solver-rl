"""Run generated solver code in the RLVP Docker sandbox with an arbitrary invocation.

Same isolation as rlvp.executor.DockerSolverExecutor (no network, read-only root, dropped
capabilities, CPU/memory/pid limits, non-root user); only the environment lookup is
removed so materials invocations can be passed directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rlvp.executor import DockerSolverExecutor  # noqa: E402
from rlvp.prompts import extract_python_code  # noqa: E402

import os as _os

IMAGE = _os.environ.get("RLVPM_SANDBOX_IMAGE", "rlvp-solver:local-diag-arm64")


@dataclass
class Invocation:
    argument_order: tuple[str, ...]
    arrays: dict[str, np.ndarray]
    scalars: dict[str, float]
    expected_shape: tuple[int, ...]


@dataclass
class RunResult:
    ok: bool
    error: str | None
    output: np.ndarray | None
    seconds: float
    stderr_tail: str = ""
    meta: dict = field(default_factory=dict)


class Sandbox(DockerSolverExecutor):
    def __init__(self, timeout_seconds: float = 120.0, cpus: str = "2", memory: str = "4g"):
        super().__init__(image=IMAGE, timeout_seconds=timeout_seconds, cpus=cpus, memory=memory)

    def run(self, code: str, inv: Invocation) -> RunResult:
        started = time.monotonic()
        name = f"rlvpm-solver-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory(prefix="rlvpm-solver-") as raw:
            work = Path(raw)
            os.chmod(work, 0o777)
            (work / "candidate.py").write_text(code + "\n")
            np.savez(work / "input.npz", **inv.arrays,
                     **{k: np.asarray(v) for k, v in inv.scalars.items()})
            (work / "call.json").write_text(json.dumps({
                "argument_order": list(inv.argument_order),
                "array_names": sorted(inv.arrays), "scalar_names": sorted(inv.scalars)}) + "\n")
            (work / "output.npy").touch()
            for f, mode in (("candidate.py", 0o444), ("input.npz", 0o444), ("call.json", 0o444),
                            ("output.npy", 0o666)):
                os.chmod(work / f, mode)
            err_path = work / "stderr.log"
            try:
                with (work / "stdout.log").open("wb") as so, err_path.open("wb") as se:
                    proc = subprocess.run(self.command(work, container_name=name), stdout=so,
                                          stderr=se, timeout=self.timeout_seconds, check=False)
            except subprocess.TimeoutExpired:
                self._force_remove(name)
                return RunResult(False, "timeout", None, time.monotonic() - started)
            tail = err_path.read_text(errors="replace")[-1500:]
            if proc.returncode != 0:
                return RunResult(False, f"exit_{proc.returncode}", None,
                                 time.monotonic() - started, tail)
            try:
                out = np.load(work / "output.npy", allow_pickle=False)
            except Exception as exc:  # noqa: BLE001
                return RunResult(False, f"load_error:{type(exc).__name__}", None,
                                 time.monotonic() - started, tail)
            if out.shape != inv.expected_shape:
                return RunResult(False, f"shape {tuple(out.shape)} != {inv.expected_shape}", None,
                                 time.monotonic() - started, tail)
            if not np.isfinite(out).all():
                return RunResult(False, "nonfinite", None, time.monotonic() - started, tail)
            return RunResult(True, None, out.astype(np.float64), time.monotonic() - started, tail)


def extract(completion: str) -> str:
    return extract_python_code(completion)

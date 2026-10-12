from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

from .contracts import ExecutionResult, SolverInvocation
from .pdes.registry import environment_for_case
from .prompts import extract_python_code


class DockerSolverExecutor:
    """Run untrusted solver code without mounting hidden references.

    This is deterministic validity isolation, not a formal security proof. The Docker
    daemon, image digest, host kernel, and limits remain part of the trusted base.
    """

    def __init__(
        self,
        *,
        image: str = "rlvp-solver:py312",
        timeout_seconds: float = 30.0,
        memory: str = "4g",
        cpus: str = "2",
    ) -> None:
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory = memory
        self.cpus = cpus

    def command(self, workdir: Path, *, container_name: str | None = None) -> list[str]:
        command = [
            "docker",
            "run",
            "--rm",
        ]
        if container_name is not None:
            command.extend(["--name", container_name])
        command.extend(
            [
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "64",
                "--memory",
                self.memory,
                "--memory-swap",
                self.memory,
                "--cpus",
                self.cpus,
                "--ulimit",
                "nofile=128:128",
                "--ulimit",
                "fsize=134217728:134217728",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=256m",
                "--mount",
                f"type=bind,src={workdir / 'candidate.py'},dst=/work/candidate.py,readonly",
                "--mount",
                f"type=bind,src={workdir / 'input.npz'},dst=/work/input.npz,readonly",
                "--mount",
                f"type=bind,src={workdir / 'call.json'},dst=/work/call.json,readonly",
                "--mount",
                f"type=bind,src={workdir / 'output.npy'},dst=/work/output.npy",
                "--user",
                "65532:65532",
                self.image,
                "python",
                "-B",
                "/opt/rlvp/execute_candidate.py",
            ]
        )
        return command

    @staticmethod
    def _force_remove(container_name: str) -> bool:
        try:
            removed = subprocess.run(
                ["docker", "rm", "--force", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if removed.returncode == 0:
            return True
        try:
            inspected = subprocess.run(
                ["docker", "inspect", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return inspected.returncode != 0

    def execute(
        self,
        completion: str,
        case: object,
        *,
        invocation_override: SolverInvocation | None = None,
    ) -> ExecutionResult:
        started = time.monotonic()
        container_name = f"rlvp-solver-{uuid.uuid4().hex}"
        environment = environment_for_case(case)
        invocation = invocation_override or environment.invocation(case)
        with tempfile.TemporaryDirectory(prefix="rlvp-solver-") as raw:
            work = Path(raw)
            os.chmod(work, 0o777)
            (work / "candidate.py").write_text(extract_python_code(completion) + "\n")
            np.savez(
                work / "input.npz",
                **invocation.arrays,
                **{name: np.asarray(value) for name, value in invocation.scalars.items()},
            )
            import json

            (work / "call.json").write_text(
                json.dumps(
                    {
                        "argument_order": invocation.argument_order,
                        "array_names": sorted(invocation.arrays),
                        "scalar_names": sorted(invocation.scalars),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            (work / "output.npy").touch()
            os.chmod(work / "candidate.py", 0o444)
            os.chmod(work / "input.npz", 0o444)
            os.chmod(work / "call.json", 0o444)
            os.chmod(work / "output.npy", 0o666)
            stdout_path = work / "stdout.log"
            stderr_path = work / "stderr.log"
            try:
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    proc = subprocess.run(
                        self.command(work, container_name=container_name),
                        stdout=stdout,
                        stderr=stderr,
                        timeout=self.timeout_seconds,
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                cleanup_ok = self._force_remove(container_name)
                return ExecutionResult(
                    executed=False,
                    shape_valid=False,
                    finite=False,
                    error="solver_timeout" if cleanup_ok else "solver_timeout_cleanup_failed",
                    duration_seconds=time.monotonic() - started,
                    metadata={"timeout_cleanup_ok": cleanup_ok},
                )
            output_path = work / "output.npy"
            if proc.returncode != 0 or not output_path.is_file():
                return ExecutionResult(
                    executed=False,
                    shape_valid=False,
                    finite=False,
                    error=f"solver_exit_{proc.returncode}",
                    duration_seconds=time.monotonic() - started,
                )
            if output_path.stat().st_size > 134_217_728:
                return ExecutionResult(
                    executed=True,
                    shape_valid=False,
                    finite=False,
                    error="solver_output_too_large",
                    duration_seconds=time.monotonic() - started,
                )
            try:
                output = np.load(output_path, allow_pickle=False)
            except (EOFError, OSError, ValueError):
                return ExecutionResult(
                    executed=True,
                    shape_valid=False,
                    finite=False,
                    error="solver_output_load_error",
                    duration_seconds=time.monotonic() - started,
                )
            shape_valid = output.shape == invocation.expected_shape
            finite = bool(shape_valid and np.isfinite(output).all())
            return ExecutionResult(
                executed=True,
                shape_valid=shape_valid,
                finite=finite,
                output=output if shape_valid else None,
                error=None if shape_valid and finite else "invalid_shape_or_nonfinite",
                duration_seconds=time.monotonic() - started,
                metadata={"image": self.image, "returncode": proc.returncode},
            )

"""CPU-VM entrypoint; runs the original Docker-isolated verifier unchanged."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

STATE = Path("/rlvp-state")
EXPORT = Path("/rlvp-export")
RECEIPTS = (
    "startup.json",
    "bundle-manifest.json",
    "verifier.jsonl",
    "dockerd.log",
    "solver-build.log",
    "verifier-service.log",
    "persistence-probe.txt",
    "failure.json",
    "docker-preflight.json",
)


def failure_receipt(error: BaseException) -> None:
    """Keep startup evidence available even if the controller disconnects."""
    STATE.mkdir(mode=0o700, exist_ok=True)
    record = {
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_unix": time.time(),
        "logs": {},
    }
    for name in ("solver-build.log", "dockerd.log", "verifier-service.log"):
        path = STATE / name
        if path.is_file():
            with path.open("rb") as handle:
                handle.seek(max(path.stat().st_size - 32768, 0))
                record["logs"][name] = handle.read().decode("utf-8", errors="replace")
    (STATE / "failure.json").write_text(json.dumps(record, indent=2))
    # Modal retains entrypoint logs after a VM exits; these contain no credentials
    # or case-store data. Do not print configuration or verifier source manifests.
    print("RLVP_VERIFIER_FAILURE " + json.dumps(record, sort_keys=True), flush=True)
    export()


def export() -> None:
    EXPORT.mkdir(mode=0o700, exist_ok=True)
    files = []
    for name in RECEIPTS:
        source = STATE / name
        if not source.is_file():
            continue
        # Capture a finite prefix: append-only live logs may grow during the copy.
        size = source.stat().st_size
        with source.open("rb") as handle:
            blob = handle.read(size)
        target = EXPORT / name
        with target.open("wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        target.chmod(0o600)
        files.append(
            {"name": name, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        )
    (EXPORT / "manifest.json").write_text(
        json.dumps({"exported_unix": time.time(), "files": files}, indent=2)
    )


def serve() -> None:
    os.umask(0o077)
    STATE.mkdir(mode=0o700, exist_ok=True)
    deadline = time.monotonic() + 300
    while not (STATE / "config.ready").is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("controller did not upload private verifier config")
        time.sleep(0.2)
    config = json.loads((STATE / "config.json").read_text())
    blob = (STATE / "cases.json").read_bytes()
    if hashlib.sha256(blob).hexdigest() != config["cases_sha256"]:
        raise ValueError("private case-store hash mismatch")
    (STATE / "cases.json").chmod(0o600)
    if platform.machine() != "x86_64":
        raise RuntimeError("Modal verifier port requires recorded amd64 architecture")
    docker_log = (STATE / "dockerd.log").open("ab", buffering=0)
    daemon = subprocess.Popen(["dockerd"], stdout=docker_log, stderr=subprocess.STDOUT)
    service = None
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if daemon.poll() is not None:
                raise RuntimeError("dockerd exited")
            info = subprocess.run(
                ["docker", "info", "--format", "{{json .}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if info.returncode == 0:
                break
            time.sleep(0.5)
        else:
            raise TimeoutError("dockerd did not become ready")
        preflight = {
            "docker_info": json.loads(info.stdout),
            "docker_buildkit_env": os.environ.get("DOCKER_BUILDKIT"),
            "buildx_builder_env": os.environ.get("BUILDX_BUILDER"),
        }
        for label, command in (
            ("version", ["docker", "version"]),
            ("buildx", ["docker", "buildx", "version"]),
        ):
            checked = subprocess.run(
                command, capture_output=True, text=True, timeout=15, check=False
            )
            preflight[label] = {
                "returncode": checked.returncode,
                "stdout": checked.stdout,
                "stderr": checked.stderr,
            }
        (STATE / "docker-preflight.json").write_text(json.dumps(preflight, indent=2))
        print("RLVP_DOCKER_PREFLIGHT " + json.dumps(preflight, sort_keys=True), flush=True)
        # Deliberately narrow build context: neither cases nor reference code enters it.
        context = Path("/tmp/rlvp-solver-build")
        (context / "src/rlvp/runner").mkdir(parents=True)
        shutil.copy2("/opt/rlvp/containers/solver.Dockerfile", context / "Dockerfile")
        shutil.copy2(
            "/opt/rlvp/src/rlvp/runner/execute_candidate.py",
            context / "src/rlvp/runner/execute_candidate.py",
        )
        with (STATE / "solver-build.log").open("ab", buffering=0) as log:
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--platform",
                    "linux/amd64",
                    "-t",
                    config["solver_image"],
                    str(context),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
        image = subprocess.run(
            ["docker", "image", "inspect", config["solver_image"]],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        receipt = {
            "run_id": config["run_id"],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "docker_info": json.loads(info.stdout),
            "solver_image": json.loads(image.stdout),
            "cases_sha256": config["cases_sha256"],
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "fastapi", "uvicorn", "pydantic", "pyyaml")
            },
            "source_manifest": json.loads((STATE / "bundle-manifest.json").read_text()),
            "started_unix": time.time(),
        }
        (STATE / "startup.json").write_text(json.dumps(receipt, indent=2))
        with (STATE / "persistence-probe.txt").open("w") as handle:
            handle.write(config["run_id"] + ":cpu-verifier-controller-export\n")
            handle.flush()
            os.fsync(handle.fileno())
        with (STATE / "verifier-service.log").open("ab", buffering=0) as log:
            service = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "rlvp.verifier_service",
                    "--case-store",
                    str(STATE / "cases.json"),
                    "--solver-image",
                    config["solver_image"],
                    "--log",
                    str(STATE / "verifier.jsonl"),
                    "--port",
                    str(config["port"]),
                    "--max-concurrency",
                    str(config["max_concurrency"]),
                    "--normalization",
                    "reference_range",
                    "--trajectory-temperature",
                    "0.05",
                    "--residual-temperature",
                    "2.0",
                    "--epsilon",
                    "1e-12",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            code = service.wait()
            raise SystemExit(code)
    finally:
        if service is not None and service.poll() is None:
            service.terminate()
        daemon.terminate()
        export()


if __name__ == "__main__":
    if sys.argv[1:] == ["export"]:
        export()
    elif sys.argv[1:] == ["serve"]:
        try:
            serve()
        except BaseException as exc:
            failure_receipt(exc)
            # The controller detects failure.json, exports receipts, and terminates.
            # This bounded window prevents immediate VM exit from destroying logs.
            time.sleep(300)
            raise
    else:
        raise SystemExit("expected serve or export")

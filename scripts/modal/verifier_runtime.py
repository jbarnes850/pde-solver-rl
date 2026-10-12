"""Controller for the unchanged RLVP verifier in a CPU Modal VM Sandbox.

Import, ``check`` and ``prepare`` are local-only. ``launch``, ``export`` and
``stop`` are explicit external operations owned by the campaign controller.
No private case store is baked into an image or put on a GPU-mounted Volume.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SDK_PIN = "1.6.1"
PORT = 19240
SOLVER_IMAGE = "rlvp-solver:modal-20261009"
VERIFIER_PINS = (
    "numpy==2.5.2",
    "scipy==1.17.1",
    "fastapi==0.141.1",
    "uvicorn==0.52.4",
    "pydantic==2.13.5",
    "pyyaml==6.0.3",
)
SOURCES = {
    "runtime": "https://modal.com/docs/guide/sandboxes#runtimes",
    "authentication": "https://modal.com/docs/guide/sandbox-networking",
    "sdk_minimum": "https://modal.com/docs/sdk/py/releases",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def private_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def sdk_check() -> dict[str, Any]:
    import modal
    from modal.sandbox_fs import SandboxFilesystem

    supported = "runtime" in inspect.signature(modal.Sandbox.create).parameters
    methods = {}
    for name in ("read_bytes", "read_text", "write_bytes", "write_text"):
        method = getattr(SandboxFilesystem, name, None)
        methods[name] = str(inspect.signature(method)) if method is not None else None
    return {
        "modal_version": modal.__version__,
        "vm_runtime_supported": supported,
        "filesystem_api_supported": all(value is not None for value in methods.values()),
        "filesystem_signatures": methods,
        "minimum_version": "1.6.0",
        "pinned_version": SDK_PIN,
        "sources": SOURCES,
    }


def source_manifest() -> dict[str, Any]:
    paths = sorted((ROOT / "src/rlvp").rglob("*.py")) + [
        ROOT / "containers/solver.Dockerfile",
        ROOT / "scripts/modal/verifier_vm.py",
    ]
    files = {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in paths}
    return {
        "files": files,
        "tree_sha256": sha256(json.dumps(files, sort_keys=True).encode()),
        "trusted_verifier_pins": VERIFIER_PINS,
        "sources": SOURCES,
    }


def prepare(case_stores: list[Path], output: Path) -> dict[str, Any]:
    """Make a CPU-only bundle; reject duplicate/different cases and sealed test input."""
    merged: dict[str, Any] = {}
    inputs = []
    for path in case_stores:
        if "test" in path.name or "held_out" in path.name:
            raise ValueError("sealed test stores are not allowed in qualification")
        blob = path.read_bytes()
        raw = json.loads(blob)
        cases = raw.get("cases", raw)
        for case_id, case in cases.items():
            if str(case.get("split", "")).lower() in {"test", "rl_test", "held_out_evaluation"}:
                raise ValueError("sealed test cases are not allowed in qualification")
            if case_id in merged and merged[case_id] != case:
                raise ValueError(f"conflicting private case {case_id}")
            merged[case_id] = case
        inputs.append(
            {"path": str(path.resolve()), "sha256": sha256(blob), "case_count": len(cases)}
        )
    if not merged:
        raise ValueError("no cases supplied")
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    private_json(output / "cases.json", {"cases": merged})
    manifest = {
        "schema": "rlvp-modal-private-verifier-bundle-v1",
        "case_count": len(merged),
        "inputs": inputs,
        "cases_sha256": sha256((output / "cases.json").read_bytes()),
        "source": source_manifest(),
    }
    private_json(output / "bundle-manifest.json", manifest)
    return manifest


def verifier_image():
    import modal

    dockerfile = (ROOT / "containers/solver.Dockerfile").read_text()
    base = dockerfile.splitlines()[0].removeprefix("FROM ")
    return (
        modal.Image.from_registry(base)
        .apt_install("docker.io")
        .pip_install(*VERIFIER_PINS)
        .env({"PYTHONPATH": "/opt/rlvp/src", "PYTHONUNBUFFERED": "1"})
        .run_commands("mkdir -p /rlvp-state /rlvp-export /opt/rlvp/containers")
        .add_local_dir(
            ROOT / "src/rlvp",
            "/opt/rlvp/src/rlvp",
            ignore=["**/__pycache__/**", "**/*.pyc"],
            copy=True,
        )
        .add_local_file(
            ROOT / "containers/solver.Dockerfile",
            "/opt/rlvp/containers/solver.Dockerfile",
            copy=True,
        )
        .add_local_file(
            ROOT / "scripts/modal/verifier_vm.py", "/opt/rlvp-modal/verifier_vm.py", copy=True
        )
    )


def request(credentials: dict[str, Any], route: str, payload: dict | None = None) -> dict:
    if not route.startswith("/") or route.startswith("//"):
        raise ValueError("route must be an absolute service path")
    headers = {"Authorization": f"Bearer {credentials['token']}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    req = urllib.request.Request(
        credentials["url"].rstrip("/") + route, data=data, headers=headers
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def export_receipts(sandbox: Any, destination: Path) -> dict[str, Any]:
    """Snapshot the finite receipt files, then verify the copied bytes locally."""
    process = sandbox.exec("python", "/opt/rlvp-modal/verifier_vm.py", "export", timeout=60)
    process.wait()
    if process.returncode:
        raise RuntimeError("verifier receipt snapshot failed: " + process.stderr.read())
    manifest = json.loads(sandbox.filesystem.read_text("/rlvp-export/manifest.json"))
    destination.mkdir(parents=True, exist_ok=True)
    destination.chmod(0o700)
    for entry in manifest["files"]:
        name = entry["name"]
        if Path(name).name != name:
            raise ValueError("invalid receipt path")
        blob = sandbox.filesystem.read_bytes("/rlvp-export/" + name)
        if sha256(blob) != entry["sha256"] or len(blob) != entry["bytes"]:
            raise RuntimeError("verifier receipt export hash mismatch")
        target = destination / name
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
    private_json(destination / "manifest.json", manifest)
    return manifest


def stop_verifier(sandbox: Any, destination: Path) -> dict[str, Any]:
    """Export best-effort, but never retain paid compute because export failed."""
    try:
        return export_receipts(sandbox, destination)
    finally:
        sandbox.terminate()


def launch(
    *,
    bundle: Path,
    state_dir: Path,
    run_id: str,
    timeout: int = 7200,
    cpu: int = 16,
    memory: int = 65536,
    max_concurrency: int = 8,
) -> dict[str, Any]:
    """Paid operation: starts a billed CPU VM Sandbox."""
    import modal

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", run_id):
        raise ValueError("invalid run id")
    sdk = sdk_check()
    if not sdk["vm_runtime_supported"] or not sdk["filesystem_api_supported"]:
        raise RuntimeError("VM runtime requires Modal >=1.6.0; use isolated modal==1.6.1")
    if not 60 <= timeout <= 36000 or not 1 <= max_concurrency <= 8:
        raise ValueError("verifier duration/concurrency exceeds the bounded contract")
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    case_blob = (bundle / "cases.json").read_bytes()
    if sha256(case_blob) != manifest["cases_sha256"]:
        raise ValueError("private case bundle changed")
    # Tuples in the current manifest serialize as lists; compare canonical serialization.
    if json.dumps(source_manifest(), sort_keys=True) != json.dumps(
        manifest["source"], sort_keys=True
    ):
        raise ValueError("verifier source changed after bundle preparation")
    app = modal.App.lookup(f"{run_id}-verifier", create_if_missing=True)
    sandbox = modal.Sandbox.create(
        "python",
        "/opt/rlvp-modal/verifier_vm.py",
        "serve",
        app=app,
        name=f"{run_id}-verifier",
        image=verifier_image(),
        runtime="vm",
        cpu=(cpu, cpu),
        memory=(memory, memory),
        timeout=timeout,
        # No tunneled ports: all inbound HTTP uses the authenticated Connect endpoint.
        encrypted_ports=[],
        unencrypted_ports=[],
        include_oidc_identity_token=False,
        tags={"campaign": run_id, "role": "cpu-verifier"},
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    credentials = {
        "sandbox_id": sandbox.object_id,
        "run_id": run_id,
        "app_name": f"{run_id}-verifier",
        "started_unix": time.time(),
        "timeout_seconds": timeout,
        "cpu": cpu,
        "memory_mib": memory,
    }
    private_json(state_dir / "verifier-connection.json", credentials)
    try:
        for name, blob in [
            ("cases.json", case_blob),
            ("bundle-manifest.json", (bundle / "bundle-manifest.json").read_bytes()),
        ]:
            sandbox.filesystem.write_bytes(blob, "/rlvp-state/" + name)
        config = {
            "run_id": run_id,
            "solver_image": SOLVER_IMAGE,
            "max_concurrency": max_concurrency,
            "port": PORT,
            "case_count": manifest["case_count"],
            "cases_sha256": manifest["cases_sha256"],
        }
        sandbox.filesystem.write_text(json.dumps(config), "/rlvp-state/config.json")
        sandbox.filesystem.write_text("uploaded\n", "/rlvp-state/config.ready")
        creds = sandbox.create_connect_token(user_metadata={"campaign": run_id}, port=PORT)
        credentials.update({"url": creds.url, "token": creds.token})
        private_json(state_dir / "verifier-connection.json", credentials)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if sandbox.poll() is not None:
                raise RuntimeError("verifier VM exited before readiness")
            try:
                failure = json.loads(sandbox.filesystem.read_text("/rlvp-state/failure.json"))
            except modal.exception.SandboxFilesystemNotFoundError:
                pass
            else:
                private_json(state_dir / "verifier-startup-failure.json", failure)
                raise RuntimeError("verifier VM startup failed; see verifier-startup-failure.json")
            try:
                health = request(credentials, "/health")
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(3)
        else:
            raise TimeoutError("verifier readiness deadline exceeded")
        if health["cases"] != manifest["case_count"]:
            raise RuntimeError("verifier case count mismatch")
        # The URL must not expose a bypass around Modal's Connect Token check.
        try:
            urllib.request.urlopen(credentials["url"].rstrip("/") + "/health", timeout=15)
        except urllib.error.HTTPError as exc:
            if exc.code not in (401, 403):
                raise RuntimeError("unexpected unauthenticated verifier response") from exc
        else:
            raise RuntimeError("verifier accepted an unauthenticated request")
        private_json(state_dir / "verifier-health.json", health)
        exported = export_receipts(sandbox, state_dir / "verifier-receipts")
        if "persistence-probe.txt" not in {row["name"] for row in exported["files"]}:
            raise RuntimeError("controller persistence probe did not survive export")
        private_json(
            state_dir / "verifier-launch.json",
            {
                **{k: v for k, v in credentials.items() if k != "token"},
                "health": health,
                "sdk": sdk_check(),
                "source": manifest["source"],
                "persistence": "controller-copy-sha256-verified; periodically export before termination",
            },
        )
        return credentials
    except BaseException:
        try:
            export_receipts(sandbox, state_dir / "verifier-failure-receipts")
        except Exception as exc:  # noqa: BLE001 - preserve original failure and terminate compute
            private_json(state_dir / "verifier-export-failure.json", {"error": str(exc)})
        finally:
            sandbox.terminate()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check")
    prep = sub.add_parser("prepare")
    prep.add_argument("--case-store", type=Path, action="append", required=True)
    prep.add_argument("--output", type=Path, required=True)
    start = sub.add_parser("launch")
    start.add_argument("--bundle", type=Path, required=True)
    start.add_argument("--state-dir", type=Path, required=True)
    start.add_argument("--run-id", required=True)
    start.add_argument("--timeout", type=int, default=7200)
    for command in ("export", "stop"):
        child = sub.add_parser(command)
        child.add_argument("--connection", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "check":
        print(json.dumps(sdk_check(), indent=2))
    elif args.action == "prepare":
        result = prepare(args.case_store, args.output)
        print(
            json.dumps(
                {
                    "case_count": result["case_count"],
                    "cases_sha256": result["cases_sha256"],
                    "output": str(args.output),
                }
            )
        )
    elif args.action == "launch":
        result = launch(
            bundle=args.bundle, state_dir=args.state_dir, run_id=args.run_id, timeout=args.timeout
        )
        print(json.dumps({k: v for k, v in result.items() if k != "token"}))
    else:
        import modal

        credentials = json.loads(args.connection.read_text())
        sandbox = modal.Sandbox.from_id(credentials["sandbox_id"])
        if args.action == "stop":
            result = stop_verifier(sandbox, args.output)
        else:
            result = export_receipts(sandbox, args.output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

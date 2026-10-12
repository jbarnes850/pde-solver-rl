"""Bounded H200:2 Function; parent owns launches and cumulative cost ledger.

  RLVP_BRIDGE_DIR=/path/to/patched-bridge-tree .venv-modal/bin/modal run scripts/modal/gpu_app.py \
      --config /absolute/path/trial.json \
      --verifier-connection /private/path/verifier-connection.json

No paid action occurs from importing this module. Use `modal run` once for a
bounded invocation; sequential trial invocations are permitted. Do not launch
overlapping ephemeral apps with this same name.

Configuration (environment, read by the local controller):
  MODAL_PROFILE        Modal profile to use (optional; Modal's active profile otherwise).
  RLVP_BRIDGE_DIR      patched NeMo-RL/Polar bridge tree (src/nemo_polar_bridge, scripts/patch); required.
  RLVP_MODAL_APP_NAME  app name (default rlvp-20261009-qwen35-9b).
  RLVP_INPUTS_VOLUME   inputs Volume (default rlvp-modal-20261009-inputs).
  RLVP_RUNS_VOLUME     runs Volume (default rlvp-modal-20261009-runs).
"""

import json
import os
import sys
from pathlib import Path

import modal

# A separately named app may run a qualification alongside a training run without the two
# ephemeral apps interfering. The default name is unchanged.
app = modal.App(os.environ.get("RLVP_MODAL_APP_NAME", "rlvp-20261009-qwen35-9b"))
inputs = modal.Volume.from_name(os.environ.get("RLVP_INPUTS_VOLUME", "rlvp-modal-20261009-inputs"))
runs = modal.Volume.from_name(os.environ.get("RLVP_RUNS_VOLUME", "rlvp-modal-20261009-runs"), create_if_missing=True)

# Modal imports this entry module at /root/gpu_app.py. Only the local controller
# has the development tree; image construction and local uploads belong there.
# The remote Function already has its resolved image and source directories.
if modal.is_local():
    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))
    from gpu_image import gpu_image

    ROOT = HERE.parents[1]
    if not os.environ.get("RLVP_BRIDGE_DIR"):
        raise RuntimeError("set RLVP_BRIDGE_DIR to the patched NeMo-RL/Polar bridge tree (see README)")
    BRIDGE = Path(os.environ["RLVP_BRIDGE_DIR"])
    image = (
        gpu_image
        .add_local_dir(str(BRIDGE / "src"), "/opt/rlvp-bridge/src", copy=True,
                       ignore=["**/__pycache__/**"])
        .add_local_dir(str(BRIDGE / "scripts/patch"), "/opt/rlvp-bridge/scripts/patch", copy=True,
                       ignore=["**/__pycache__/**"])
        .add_local_dir(str(ROOT / "src"), "/opt/rlvp-src", copy=True,
                       ignore=["**/__pycache__/**"])
        .add_local_dir(str(HERE), "/opt/rlvp-modal", copy=True,
                       ignore=["upstream/**", "**/__pycache__/**", "test_*.py"])
    )
else:
    sys.path.insert(0, "/opt/rlvp-modal")
    image = None


@app.function(
    image=image, gpu="H200:2", cpu=16, memory=131072,
    timeout=36300, startup_timeout=1800, retries=0,
    min_containers=0, max_containers=1, scaledown_window=2,
    volumes={"/inputs": inputs, "/runs": runs},
)
def experiment(config: dict, verifier: dict):
    import sys

    sys.path.insert(0, "/opt/rlvp-modal")
    from gpu_runtime import run

    return run(config, verifier, runs)


@app.function(image=image, cpu=1, memory=8192, timeout=300, retries=0)
def import_probe():
    """Run on CPU in the real Modal import context before a GPU retry."""
    import importlib
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    sys.path[:0] = ["/opt/rlvp-modal", "/opt/rlvp-bridge/src", "/opt/rlvp-src", "/opt/nemo-rl"]
    modules = ["gpu_runtime", "trusted_runtime", "verifier_proxy", "patch_transport", "modal_refit_probe",
               "nemo_polar_bridge.datasets.prepare_dataset", "nemo_polar_bridge.collector"]
    locations = {name: importlib.import_module(name).__file__ for name in modules}
    assert not modal.is_local(), "probe must execute inside an actual Modal Function"
    assert image is None, "remote import unexpectedly constructed the local image"
    assert all(Path(locations[name]).is_relative_to("/opt/rlvp-modal") for name in modules[:5])
    assert all(Path(locations[name]).is_relative_to("/opt/rlvp-bridge/src") for name in modules[5:])
    # Check exact runtime patch composition against image bytes in a disposable
    # copy, before GPU allocation. Do not mutate the image's imported modules.
    with tempfile.TemporaryDirectory(prefix="rlvp-cpu-preflight-") as directory:
        staging = Path(directory)
        nemo_root, bridge_root = staging / "nemo", staging / "bridge"
        source_pairs = {
            "/opt/nemo-rl/nemo_rl/algorithms/loss/loss_functions.py": nemo_root / "nemo_rl/algorithms/loss/loss_functions.py",
            "/opt/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py": nemo_root / "nemo_rl/models/generation/vllm/vllm_worker_async.py",
            "/opt/rlvp-bridge/scripts/patch/nemo_rl_fp8_broadcast/vllm_backend.py": nemo_root / "nemo_rl/models/generation/vllm/vllm_backend.py",
            "/opt/rlvp-bridge/scripts/patch/nemo_rl_memory/model_utils.py": nemo_root / "nemo_rl/distributed/model_utils.py",
            "/opt/rlvp-bridge/scripts/patch/nemo_rl_memory/advantage_estimator.py": nemo_root / "nemo_rl/algorithms/advantage_estimator.py",
            "/opt/rlvp-bridge/scripts/patch/nemo_rl_memory/grpo.py": nemo_root / "nemo_rl/algorithms/grpo.py",
            "/opt/rlvp-bridge/scripts/patch/nemo_rl_fp8_broadcast/dtensor_policy_worker_v2.py": nemo_root / "nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py",
            "/opt/rlvp-bridge/src/nemo_polar_bridge/collector.py": bridge_root / "src/nemo_polar_bridge/collector.py",
        }
        for source, destination in source_pairs.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        sys.path.insert(0, "/opt/rlvp-modal/patches")
        audit = importlib.import_module("apply_audit_patches").apply(nemo_root)
        transport = importlib.import_module("patch_transport").apply(nemo_root, bridge_root)
        maxrl_advantage = importlib.import_module("patch_maxrl_advantage").apply(nemo_root)
        maxrl_group_refit = importlib.import_module("patch_group_refit").apply(bridge_root)
        maxrl_training = importlib.import_module("patch_maxrl_training").apply(nemo_root)
    return {"actual_modal_function_context": True, "entry_file": __file__,
            "gpu_requested": False, "local_uploads_constructed": False, "module_locations": locations,
            "audit_patch_manifest": audit, "transport_patch_manifest": transport,
            "maxrl_advantage_patch_manifest": maxrl_advantage,
            "maxrl_group_refit_patch_manifest": maxrl_group_refit,
            "maxrl_training_patch_manifest": maxrl_training,
            "disposable_patch_composition_passed": True}


@app.local_entrypoint()
def main(config: str, verifier_connection: str):
    settings = json.loads(Path(config).read_text())
    verifier = json.loads(Path(verifier_connection).read_text())
    print(json.dumps(experiment.remote(settings, verifier), indent=2))

"""Pinned amd64 NeMo runtime. Defining this image does not build or allocate it.

Build-only check (CPU, no GPU):
  .venv-modal/bin/modal run scripts/modal/gpu_image.py   # set MODAL_PROFILE to choose a Modal profile
"""

import modal

NEMO_REF = "c236061b250e97638722292ab8a54d5eb47ae00f"
NGC_IMAGE = (
    "nvcr.io/nvidia/nemo-rl@sha256:"
    "9392cc2cc07140f6a4cbd30f31ebffd0a54de3ded25c5a3b49505407a779e4f9"
)

# v0.7.0 amd64 supplies compiled CUDA 13.2 / torch 2.11 / vLLM 0.20
# dependencies. The SOURCE and actor environments below are restored to the
# campaign's existing pin; this is not an upgrade to v0.7.0 algorithm code.
gpu_image = (
    modal.Image.from_registry(NGC_IMAGE)
    .env({
        "NRL_CONTAINER": "1",
        "NEMO_RL_COMMIT": NEMO_REF,
        "RAY_USAGE_STATS_ENABLED": "0",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "NEMO_RL_VENV_DIR": "/opt/ray_venvs",
        "UV_PROJECT_ENVIRONMENT": "/opt/nemo_rl_venv",
        "UV_LINK_MODE": "symlink",
        "UV_NO_CACHE": "false",
        "UV_CACHE_DIR": "/root/.cache/uv",
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "MAX_JOBS": "8",
        "NVTE_BUILD_THREADS_PER_JOB": "1",
    })
    .run_commands(
        "cd /opt/nemo-rl && git fetch --depth 1 origin " + NEMO_REF
        + " && git checkout --detach " + NEMO_REF
        + " && git submodule update --init --recursive",
        "cd /opt/nemo-rl && UV_NO_CACHE=false uv sync --locked --no-install-project",
        "export UV_NO_CACHE=false; cd /opt/nemo-rl && python - <<'PY'\n"
        "from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY\n"
        "from nemo_rl.utils.venvs import create_local_venv\n"
        "actors=('DTensorPolicyWorkerV2','VllmAsyncGenerationWorker','VllmGenerationWorker',"
        "'AsyncTrajectoryCollector','ReplayBuffer','MathEnvironment')\n"
        "for name, executable in ACTOR_ENVIRONMENT_REGISTRY.items():\n"
        "    if name.rsplit('.',1)[-1] in actors and executable.startswith('uv'):\n"
        "        print('Reconcile pinned actor:',name,flush=True)\n"
        "        create_local_venv(executable,name,force_rebuild=True)\n"
        "PY",
        "cd /opt/nemo-rl && python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint",
    )
)

app = modal.App("rlvp-20261009-image-check")


@app.function(image=gpu_image, cpu=2, memory=4096, timeout=300, retries=0)
def check_image():
    import json
    import platform
    import subprocess

    versions = {}
    actors = [
        "/opt/nemo_rl_venv/bin/python",
        "/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python",
        "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python",
    ]
    for executable in actors:
        command = (
            "import json,importlib.metadata as m; "
            "print(json.dumps({n:m.version(n) for n in ['torch','transformers','ray','tensordict']}))"
        )
        versions[executable] = json.loads(subprocess.check_output([executable, "-c", command]))
    receipt = {
        "architecture": platform.machine(),
        "nemo_commit": subprocess.check_output(
            ["git", "-C", "/opt/nemo-rl", "rev-parse", "HEAD"], text=True
        ).strip(),
        "actors": versions,
        "gpu_requested": False,
    }
    assert receipt["architecture"] == "x86_64", receipt
    assert receipt["nemo_commit"] == NEMO_REF, receipt
    return receipt


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(check_image.remote(), indent=2))

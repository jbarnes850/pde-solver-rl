"""Install opt-in hooks after the existing audit and transport patches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _replace(source, before, after):
    count = source.count(before)
    if count != 1:
        raise RuntimeError(f"MaxRL training anchor expected once, found {count}: {before[:100]!r}")
    return source.replace(before, after)


def patch_grpo(source):
    if "from nemo_rl.utils.maxrl_training import" in source:
        raise RuntimeError("MaxRL training hooks are already installed")
    source = _replace(source,
        "from nemo_rl.utils.rlvp_modal_audit import archive_batch\n",
        "from nemo_rl.utils.rlvp_modal_audit import archive_batch\n"
        "from nemo_rl.utils.maxrl_training import after_train_step, after_refit_step, raise_pending_failure\n")
    source = _replace(source,
        '                print("🔄 Synchronizing policy weights to trajectory collector…")\n',
        '                maxrl_step_state = after_train_step(policy, train_data, timer, step, master_config)\n\n'
        '                print("🔄 Synchronizing policy weights to trajectory collector…")\n')
    source = _replace(source,
        '                # Clear logger metrics after each refit (weight sync), starting a new logging cycle\n',
        '                after_refit_step(maxrl_step_state, bool(NEED_REFIT and post_refit_error is None))\n\n'
        '                # Clear logger metrics after each refit (weight sync), starting a new logging cycle\n')
    return _replace(source,
        '                if post_refit_error is not None:\n'
        '                    raise RuntimeError("post-refit proof failed; native checkpoint preserved") from post_refit_error\n',
        '                raise_pending_failure(maxrl_step_state)\n'
        '                if post_refit_error is not None:\n'
        '                    raise RuntimeError("post-refit proof failed; native checkpoint preserved") from post_refit_error\n')


def patch_worker(source):
    if "from nemo_rl.utils.maxrl_training import" in source:
        raise RuntimeError("MaxRL training worker hook is already installed")
    source = _replace(source,
        '_ACTIVE_ONLY = os.getenv("NRL_ACTIVE_ONLY") == "1"\n',
        '_ACTIVE_ONLY = os.getenv("NRL_ACTIVE_ONLY") == "1"\n'
        'from nemo_rl.utils.maxrl_training import prepare_worker_train\n')
    return _replace(source,
        '        local_gbs = gbs // self.dp_size\n',
        '        prepare_worker_train(self, data, gbs, mbs, eval_mode)\n'
        '        local_gbs = gbs // self.dp_size\n')


def apply(nemo_root):
    root = Path(nemo_root)
    pending = {}
    for relative, patch in {
        "nemo_rl/algorithms/grpo.py": patch_grpo,
        "nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py": patch_worker,
    }.items():
        path = root / relative
        before = path.read_bytes()
        after = patch(before.decode()).encode()
        compile(after, str(path), "exec")
        pending[path] = (before, after)
    for name in ("maxrl_training.py", "maxrl_lr_gate.py"):
        path = root / "nemo_rl/utils" / name
        if path.exists():
            raise FileExistsError(path)
        after = Path(__file__).with_name(name).read_bytes()
        compile(after, str(path), "exec")
        pending[path] = (b"", after)
    manifest = {}
    # All syntax, anchors and required helper files are checked before mutation.
    for path, (before, after) in pending.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(after)
        manifest[str(path.relative_to(root))] = {
            "before_sha256": hashlib.sha256(before).hexdigest() if before else None,
            "after_sha256": hashlib.sha256(after).hexdigest(),
        }
    (root / "rlvp-maxrl-training-patches.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-root", required=True)
    args = parser.parse_args()
    print(json.dumps(apply(args.nemo_root), indent=2))

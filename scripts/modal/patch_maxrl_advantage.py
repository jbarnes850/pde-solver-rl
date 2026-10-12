"""Install the optional MaxRL branch AFTER apply_audit_patches.py; no launch."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

RELATIVE = "nemo_rl/algorithms/advantage_estimator.py"
HELPER = "nemo_rl/utils/rlvp_maxrl_advantage.py"


def _replace(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f"MaxRL patch expected one anchor, got {source.count(before)}: {before!r}")
    return source.replace(before, after, 1)


def patch_advantage(source):
    """Change only GRPO estimator's explicitly enabled branch; fail on drift/reapply."""
    before_import = "from nemo_rl.utils.rlvp_modal_audit import group_statistics_fp32\n"
    if "from nemo_rl.utils.rlvp_maxrl_advantage import" in source:
        raise RuntimeError("MaxRL patch already applied")
    source = _replace(source, before_import, before_import + (
        "from nemo_rl.utils.rlvp_maxrl_advantage import (\n"
        "    maxrl_settings, compute_maxrl_advantage, record_maxrl_receipt,\n)\n"
    ))
    start = source.index("class GRPOAdvantageEstimator:")
    end = source.index("\nclass GDPOAdvantageEstimator:", start)
    part = source[start:end]
    part = _replace(part, '        self.normalize_rewards = estimator_config["normalize_rewards"]\n',
                    '        self.normalize_rewards = estimator_config["normalize_rewards"]\n'
                    '        self._rlvp_maxrl_options = maxrl_settings(estimator_config, loss_config)\n')
    anchor = "        # Reward/statistic/advantage math stays FP32 regardless of model dtype.\n"
    part = _replace(part, anchor,
                    "        if self._rlvp_maxrl_options is not None:\n"
                    "            advantages, receipt = compute_maxrl_advantage(\n"
                    "                prompt_ids, rewards, mask, **self._rlvp_maxrl_options,\n"
                    "            )\n"
                    "            record_maxrl_receipt(receipt)\n"
                    "            return advantages\n\n" + anchor)
    result = source[:start] + part + source[end:]
    compile(result, RELATIVE, "exec")
    return result


def apply(nemo_root):
    root = Path(nemo_root)
    target = root / RELATIVE
    source = target.read_text()
    result = patch_advantage(source)
    helper_source = Path(__file__).with_name("maxrl_advantage.py").read_text()
    compile(helper_source, HELPER, "exec")
    helper_target = root / HELPER
    manifest_path = root / "rlvp-maxrl-advantage-patch.json"
    diff_path = root / "rlvp-maxrl-advantage.patch"
    if any(p.exists() for p in (helper_target, manifest_path, diff_path)):
        raise FileExistsError("refusing to replace an existing MaxRL patch artifact")
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    manifest = {
        "patch_order": "after apply_audit_patches.py", "activation": "RLVP_ADVANTAGE_NORMALIZER=mean_plus_epsilon",
        RELATIVE: {"before_sha256": sha(source), "after_sha256": sha(result)},
        HELPER: {"after_sha256": sha(helper_source)},
    }
    diff = "".join(difflib.unified_diff(source.splitlines(True), result.splitlines(True),
                                         fromfile=RELATIVE + ".before", tofile=RELATIVE))
    helper_target.parent.mkdir(parents=True, exist_ok=True)
    helper_target.write_text(helper_source)
    target.write_text(result)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    diff_path.write_text(diff)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", required=True)
    args = parser.parse_args()
    print(json.dumps(apply(args.nemo_root), indent=2, sort_keys=True))

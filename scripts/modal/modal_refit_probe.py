"""Read-only fixed-prefix FP8 diagnostics, separate from training sampling."""

import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path


def record_fp8_probe(base_url, stage, weight_version):
    fixed = json.loads(Path(os.environ["NRL_FIXED_LOGPROB_PROBE_PATH"]).read_text())
    request = urllib.request.Request(
        base_url.rstrip("/").removesuffix("/v1") + "/rlvp/prompt-logprobs",
        data=json.dumps({"token_ids": fixed["token_ids"]}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    if result["token_ids"] != fixed["token_ids"]:
        raise RuntimeError("FP8 diagnostic changed the frozen input tokens")
    values = result["next_token_logprobs"]
    if len(values) != len(fixed["token_ids"]) - 1 or not all(math.isfinite(v) for v in values):
        raise RuntimeError("FP8 fixed-prefix likelihood is incomplete or nonfinite")
    result.update({"event": "fp8_fixed_prefix_logprobs", "stage": stage,
                   "weight_version": int(weight_version), "training_sample": False,
                   "diagnostic_temperature": 1.0})
    target = Path(os.environ["NRL_AUDIT_DIR"]) / "fixed-logprob-probe" / f"fp8-{stage}-v{weight_version:06d}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.write_text(payload)
    receipt = {"event": result["event"], "stage": stage, "weight_version": weight_version,
               "path": str(target), "sha256": hashlib.sha256(payload.encode()).hexdigest(),
               "token_count": len(fixed["token_ids"]), "diagnostic_temperature": 1.0}
    print("[rlvp_receipt] " + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt

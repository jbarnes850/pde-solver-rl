#!/usr/bin/env bash
# Export one completed probe arm from the runs volume and verify its internal hashes.
# Usage: export_probe_arm.sh RUN_ID
set -euo pipefail
RUN="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${RLVP_ROOT:-$(cd "$HERE/../.." && pwd)}"
S="${RLVP_CAMPAIGN_DIR:-$(cd "$HERE/.." && pwd)}"
M="${MODAL_BIN:-$ROOT/.venv-modal/bin/modal}"
RUNS_VOLUME="${RLVP_RUNS_VOLUME:-rlvp-modal-20261009-runs}"
OUT=$S/probe-results/$RUN; mkdir -p $OUT/receipts $OUT/logs
for f in groups.jsonl summary.json probe_config.json seed_table.jsonl policy-artifact-receipt.json; do
  "$M" volume get --force "$RUNS_VOLUME" /$RUN/evaluation/$f $OUT/$f >/dev/null
done
for f in run-exit.json policy-artifact.json verified-inputs.json resume-source.json verifier-health.json sealed-policy-identity.json; do
  "$M" volume get --force "$RUNS_VOLUME" /$RUN/receipts/$f $OUT/receipts/$f >/dev/null 2>&1 || true
done
for f in evaluation.log refit-only-ready.json; do
  "$M" volume get --force "$RUNS_VOLUME" /$RUN/logs/$f $OUT/logs/$f >/dev/null 2>&1 || true
done
python3 - "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1])
cfg = json.loads((out / "probe_config.json").read_text())
groups = [json.loads(l) for l in (out / "groups.jsonl").read_text().splitlines() if l.strip()]
seed_sha = hashlib.sha256((out / "seed_table.jsonl").read_bytes()).hexdigest()
bad = 0
for g in groups:
    assert g["config_sha256"] == cfg["config_sha256"]
    for r in g["responses"]:
        bad += hashlib.sha256(r["content"].encode()).hexdigest() != r["content_sha256"]
        bad += r.get("model_retries", 0) != 0 or r.get("verifier_retries", 0) != 0
print(json.dumps({"run": out.name, "groups": len(groups), "responses": sum(len(g["responses"]) for g in groups),
                  "bad_or_retried": bad, "seed_table_matches_config": seed_sha == cfg["seed_table_sha256"],
                  "system_prompt_sha256": cfg["system_prompt_sha256"], "manifest_sha256": cfg["manifest_sha256"],
                  "seed_manifest_sha256": cfg.get("seed_manifest_sha256"), "case_concurrency": cfg.get("case_concurrency"),
                  "policy_checkpoint_tree": json.loads((out / "policy-artifact-receipt.json").read_text()).get("checkpoint_tree_sha256")}))
PY

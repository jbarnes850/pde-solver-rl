"""Local post-freeze preparation and pre-sampling identity check; no launches."""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


# v1: the original 128-case rl_test pair (SFT baseline). v2: a declared fresh sealed role whose
# baseline is the training run's own starting checkpoint.
SEALED_SPLITS = {"rl_test": {"test", "rl_test"}, "rl_fresh_test_20261010": {"fresh_test"},
                 "rl_fresh_test_20261011v2": {"fresh_test"},
                 "rl_fresh_test_20261011v3": {"fresh_test"},
                 "rl_fresh_test_20261012v4": {"fresh_test"},
                 "held_out_evaluation_20261012v2": {"held_out_evaluation"},
                 "held_out_evaluation": {"held_out_evaluation"}}


def load_freeze(path, expected_sha256):
    blob = Path(path).read_bytes()
    if digest(blob) != expected_sha256:
        raise ValueError("selection freeze bytes differ from the declared hash")
    freeze = json.loads(blob)
    schema = freeze.get("schema")
    if schema == "rlvp.selected-policy-freeze.v1":
        freeze = {**freeze, "sealed_role": "rl_test", "sealed_case_count": 128,
                  "baseline_checkpoint_tree_sha256": freeze.get("sft_checkpoint_tree_sha256")}
    elif schema == "rlvp.selected-policy-freeze.v2":
        if freeze.get("sealed_role") not in SEALED_SPLITS or freeze.get("sealed_role") == "rl_test":
            raise ValueError("v2 freeze must declare a fresh sealed role")
        if not isinstance(freeze.get("sealed_case_count"), int) or freeze["sealed_case_count"] <= 0:
            raise ValueError("v2 freeze must declare the sealed case count")
    else:
        raise ValueError("missing selected-policy freeze schema")
    if freeze.get("test_unopened_at_freeze") is not True:
        raise ValueError("freeze must declare test remained unopened during selection")
    if not 0 < freeze.get("frozen_unix", 0) <= time.time():
        raise ValueError("freeze timestamp must precede access")
    for key in ("selected_checkpoint_tree_sha256", "baseline_checkpoint_tree_sha256",
                "selection_evidence_sha256", "evaluation_contract_sha256",
                "verifier_source_tree_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(freeze.get(key, ""))):
            raise ValueError(f"missing or invalid frozen {key}")
    return freeze, blob


def prepare(freeze_path, freeze_sha256, public_manifest, case_store, evaluation_config, output):
    # No source test path is opened/stat'ed/hashed before all preconditions pass.
    import yaml
    from verifier_runtime import private_json, source_manifest
    from rlvp.variance_probe import load_locked_rows

    freeze, freeze_blob = load_freeze(freeze_path, freeze_sha256)
    contract_blob = Path(evaluation_config).read_bytes()
    contract = yaml.safe_load(contract_blob)
    if digest(contract_blob) != freeze["evaluation_contract_sha256"]:
        raise ValueError("evaluation contract differs from freeze")
    role, count = freeze["sealed_role"], freeze["sealed_case_count"]
    if contract.get("evaluation", {}).get("manifest_role") != role:
        raise ValueError("sealed evaluation requires the frozen sealed role")
    source = source_manifest()
    if source["tree_sha256"] != freeze["verifier_source_tree_sha256"]:
        raise ValueError("verifier source differs from frozen qualified source")
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in ("public", "private"):
        (output / name).mkdir(mode=0o700)

    def persist(path, blob):
        with path.open("xb") as handle:
            os.chmod(path, 0o600)
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())

    persist(output / "public/selection-freeze.json", freeze_blob)
    persist(output / "public/evaluation-contract.yaml", contract_blob)
    access = {"schema": "rlvp.sealed-test-access.v1", "freeze_sha256": freeze_sha256,
              "frozen_unix": freeze["frozen_unix"], "access_begins_unix": time.time(),
              "public_manifest": str(public_manifest), "private_case_store": str(case_store)}
    private_json(output / "access-start.json", access)
    # Persist directory entries before the first sealed-source read.
    fd = os.open(output, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    public_blob = Path(public_manifest).read_bytes()
    case_blob = Path(case_store).read_bytes()
    public_rows = [json.loads(line) for line in public_blob.splitlines() if line.strip()]
    # Preserve exact public bytes; reject anything outside the existing public schema.
    allowed = {"assumption_ids", "benchmark_id", "case_id", "case_seed", "family_case_ordinal",
               "family_id", "prompt_form", "prompt_id", "role", "schema_version", "split",
               "row_id", "generator_version", "physical_case_id", "prompt", "prompt_sha256",
               "reference_operator_id", "residual_operator_id", "public_input_fingerprint"}
    if any(set(row) - allowed for row in public_rows):
        raise ValueError("unexpected field in public test manifest; do not upload")
    persist(output / "public/rl-test.jsonl", public_blob)
    rows = load_locked_rows(output / "public/rl-test.jsonl", expected_role=role)
    raw = json.loads(case_blob)
    cases = raw.get("cases", raw)
    if len(rows) != count or len(cases) != count:
        raise ValueError(f"sealed paired evaluation requires exactly {count} public/private cases")
    if {row["physical_case_id"] for row in rows} != set(cases):
        raise ValueError("public/private test identities differ")
    for row in rows:
        case = cases[row["physical_case_id"]]
        if case.get("split") not in SEALED_SPLITS[role] or row.get("split") not in SEALED_SPLITS[role]:
            raise ValueError("mixed or non-test split in sealed bundle")
        if case.get("case_id") != row["physical_case_id"] or case.get("family_id") != row["family_id"]:
            raise ValueError("private case identity/family differs from public manifest")
    persist(output / "private/cases.json", case_blob)
    manifest = {"schema": "rlvp-modal-private-verifier-bundle-v1", "case_count": count,
                "cases_sha256": digest(case_blob), "source": source,
                "inputs": [{"path": str(case_store), "sha256": digest(case_blob), "case_count": count}],
                "selection_freeze_sha256": freeze_sha256}
    private_json(output / "private/bundle-manifest.json", manifest)
    private_json(output / "access-complete.json", {**access, "completed_unix": time.time(),
                 "manifest_sha256": digest(public_blob), "case_store_sha256": digest(case_blob),
                 "case_count": count})
    return {"output": str(output), "freeze_sha256": freeze_sha256,
            "manifest_sha256": digest(public_blob), "case_store_sha256": digest(case_blob)}


def assert_frozen_policy(config, evaluation, policy_artifact, ready, health):
    role = evaluation.get("evaluation", {}).get("manifest_role")
    policy_role = config.get("sealed_test_policy")
    if role not in SEALED_SPLITS and policy_role is None:
        return None  # Existing training and validation path is unchanged.
    if role not in SEALED_SPLITS or policy_role not in {"baseline", "selected"}:
        raise ValueError("sealed evaluation needs a sealed role and baseline/selected identity")
    freeze, _ = load_freeze(config["selection_freeze_path"], config["selection_freeze_sha256"])
    if role != freeze["sealed_role"]:
        raise ValueError("GPU evaluation role differs from the frozen sealed role")
    if digest(Path(config["evaluation_config"]).read_bytes()) != freeze["evaluation_contract_sha256"]:
        raise ValueError("GPU evaluation contract differs from freeze")
    key = "baseline_checkpoint_tree_sha256" if policy_role == "baseline" else "selected_checkpoint_tree_sha256"
    expected = freeze[key]
    if policy_artifact["checkpoint_tree_sha256"] != expected or ready.get("weights_tree_sha256") != expected:
        raise ValueError("actual checkpoint/refit identity differs from frozen policy")
    if (ready.get("optimizer_steps") != 0 or ready.get("refit_complete") is not True
            or ready.get("collector_started") is not False or ready.get("replay_started") is not False):
        raise ValueError("sealed evaluation requires completed refit with no optimization/collection")
    if health.get("cases") != freeze["sealed_case_count"]:
        raise ValueError("sealed verifier must contain exactly the frozen sealed case count")
    return {"selection_freeze_sha256": config["selection_freeze_sha256"],
            "policy_role": policy_role, "checkpoint_tree_sha256": expected,
            "refit_weights_tree_sha256": ready["weights_tree_sha256"], "manifest_role": role}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "public-manifest", "case-store", "evaluation-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.freeze, args.freeze_sha256, args.public_manifest,
                             args.case_store, args.evaluation_config, args.output), indent=2))

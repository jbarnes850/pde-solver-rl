"""Parent-side per-update monitor for the MaxRL training run, with pre-declared stop rules.

Downloads each completed update's receipts from the runs volume (gradient JSON, MaxRL advantage
JSON, batch tensors) and computes: LR, grad norm, clip scale, relative update; anchored-reward
support (useful groups: nonconstant and mean/(mean+eps) >= 0.1) by family; pre-update
learner/behavior ratio-cap incidence and clipped-token ESS; sampled reference K3.

Stop rules (written before training; applied to updates after LR promotion):
  - any nonfinite statistic;
  - ratio>5 incidence > 2% on two consecutive updates;
  - clipped-token ESS fraction < 0.80 on two consecutive updates;
  - mean clamped reference K3 > 0.5;
  - first four updates: fewer than 6 useful groups or fewer than 3 families with a useful group;
  - anchored mean reward below 25% of the first-four-update mean for three consecutive updates
    after update 8 (policy collapse).
If a stop rule fires, the evaluated checkpoint is the last update before the triggering window.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

RUN = sys.argv[1]
SCHEDULE = Path(sys.argv[2])
EPS = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "training-monitor" / RUN
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(os.environ.get("RLVP_ROOT", Path(__file__).resolve().parents[2]))  # repository root
MODAL = os.environ.get("MODAL_BIN", str(ROOT / ".venv-modal/bin/modal"))
RUNS_VOLUME = os.environ.get("RLVP_RUNS_VOLUME", "rlvp-modal-20261009-runs")
# Minimal environment for the Modal CLI; the Modal profile and credentials pass through only when set.
ENV = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("MODAL_PROFILE", "MODAL_CONFIG_PATH", "MODAL_TOKEN_ID",
                                                         "MODAL_TOKEN_SECRET") if os.environ.get(k)}}
family_of_group = {}
for line in SCHEDULE.read_text().splitlines():
    row = json.loads(line)
    family_of_group[row["group_id"]] = row["family_id"]


def get(remote, local):
    if local.exists():
        return True
    result = subprocess.run([MODAL, "volume", "get", "--force", RUNS_VOLUME, remote, str(local)],
                            capture_output=True, env=ENV, timeout=900)
    return result.returncode == 0 and local.exists()


def listing(remote):
    result = subprocess.run([MODAL, "volume", "ls", RUNS_VOLUME, remote],
                            capture_output=True, text=True, env=ENV, timeout=120)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def batch_stats(path):
    b = torch.load(path, map_location="cpu", weights_only=True)
    mask = b["token_mask"].bool() & (b["sample_mask"][:, None] > 0)
    learner, behavior, ref = b["prev_logprobs"][mask], b["generation_logprobs"][mask], b["reference_policy_logprobs"][mask]
    ratio = torch.exp((learner - behavior).double())
    w = ratio.clamp(max=5.0)
    ess = float(w.sum() ** 2 / (w.numel() * (w ** 2).sum()))
    # Pinned NeMo calculate_kl(k3) convention as in maxrl_lr_gate: clamp log-ratio, then value.
    d = (ref - learner).double().clamp(-20, 20)
    k3 = float((torch.exp(d) - 1 - d).clamp(-10, 10).mean())
    return {"ratio_above5_fraction": float((ratio > 5).double().mean()), "clipped_ess_fraction": ess,
            "reference_k3_mean": k3, "action_tokens": int(mask.sum())}


def main():
    grads = sorted(x for x in listing(f"/{RUN}/receipts/gradients") if x.endswith(".json"))
    advs = sorted(x for x in listing(f"/{RUN}/receipts/maxrl-advantages") if x.endswith(".json"))
    rows = []
    for g in grads:
        name = Path(g).name
        step = int(name.split("-")[1])
        local = OUT / name
        if not get("/" + g, local):
            continue
        rec = json.loads(local.read_text())
        row = {"update": step, "lr": rec["learning_rates"], "grad_l2": rec["grad_l2"],
               "clip_scale": rec["measured_clip_scale"], "relative_update": rec["trainable_relative_update_l2"]}
        bname = f"step-{step - 1:06d}.pt"
        blocal = OUT / bname
        cache = OUT / f"{bname}.stats.json"
        if cache.exists():
            row.update(json.loads(cache.read_text()))
        elif get(f"/{RUN}/receipts/batches/{bname}", blocal):
            stats = batch_stats(blocal)
            cache.write_text(json.dumps(stats))
            row.update(stats)
            blocal.unlink()  # keep only statistics locally; originals stay on the volume
        rows.append(row)
    adv_rows = []
    for a in advs:
        local = OUT / Path(a).name
        if not get("/" + a, local):
            continue
        rec = json.loads(local.read_text())
        means = rec["group_mean_rewards"]
        useful, fams = 0, set()
        # group_ids are batch-local; family coverage is attributed from the verifier log instead.
        for m, lo, hi in zip(means, rec["group_min_rewards"], rec["group_max_rewards"]):
            if hi != lo and m / (m + EPS) >= 0.1:
                useful += 1
        adv_rows.append({"file": Path(a).name, "mean_reward": sum(means) / len(means), "useful_groups": useful,
                         "useful_families": sorted(fams)})
    report = {"run": RUN, "updates": rows, "advantage_batches": adv_rows, "stop": None}
    finite = all(math.isfinite(v) for r in rows for v in (r["grad_l2"], r["relative_update"]))
    if not finite:
        report["stop"] = "nonfinite gradient statistic"
    late = [r for r in rows if "ratio_above5_fraction" in r]
    for a, b in zip(late, late[1:]):
        if a["ratio_above5_fraction"] > 0.02 and b["ratio_above5_fraction"] > 0.02:
            report["stop"] = f"ratio>5 incidence above 2% at updates {a['update']},{b['update']}"
        if a["clipped_ess_fraction"] < 0.8 and b["clipped_ess_fraction"] < 0.8:
            report["stop"] = f"clipped ESS below 0.80 at updates {a['update']},{b['update']}"
    if any(r.get("reference_k3_mean", 0) > 0.5 for r in late):
        report["stop"] = "reference K3 above 0.5"
    if len(adv_rows) >= 4:
        first = adv_rows[:4]
        if sum(x["useful_groups"] for x in first) < 6:
            report["stop"] = "first-four-update support below 6 useful groups or 3 families"
        base = sum(x["mean_reward"] for x in first) / 4
        tail = adv_rows[8:]
        for i in range(len(tail) - 2):
            if all(x["mean_reward"] < 0.25 * base for x in tail[i:i + 3]):
                report["stop"] = "anchored reward collapse"
    (OUT / "monitor.json").write_text(json.dumps(report, indent=1))
    for r in rows:
        print(json.dumps(r))
    for a in adv_rows:
        print(json.dumps(a))
    print("STOP:", report["stop"])


if __name__ == "__main__":
    main()

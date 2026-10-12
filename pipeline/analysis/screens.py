"""Phase-start screening pools: fresh candidate draws scored at the phase's starting checkpoint (8 samples, P2 prompt)."""
import json, math, collections
from pathlib import Path
C = Path(os.environ.get("RLVP_CAMPAIGN_DIR", Path(__file__).resolve().parents[1]))
W = Path(__file__).resolve().parent
out = {}
for run, ck in (("calib-screen160-s128", "s128"), ("calib-screen192-s160", "s160"), ("calib-screen224-s192", "s192"), ("calib-screen256-s224", "s224")):
    fam = collections.defaultdict(lambda: [0, 0, 0])
    for l in open(C / "probe-results" / run / "groups.jsonl"):
        g = json.loads(l)
        for r in g["responses"]:
            v = r["verifier"]; e = v.get("trajectory_loss_reference_range")
            a = fam[g["family_id"]]
            a[0] += v.get("validity") == 1.0 and e is not None and math.isfinite(e) and e <= 0.01
            a[1] += 1
        a[2] += 1
    out[ck] = {"run": run, "families": {f: {"cases": v[2], "responses": v[1], "strict_rate": round(v[0] / v[1], 4)} for f, v in fam.items()}}
    print(ck, {f: (v["cases"], v["strict_rate"]) for f, v in out[ck]["families"].items()})
(W / "screens.json").write_text(json.dumps(out, indent=1))

"""Shared loaders for the story analysis (read-only on campaign data)."""
import os
import collections
import json
from pathlib import Path

C = Path(os.environ.get("RLVP_CAMPAIGN_DIR", Path(__file__).resolve().parents[1]))
W = Path(__file__).resolve().parent
CK_ORDER = ["base", "s32", "s64", "s96", "s128", "s160", "s192", "s224", "s256"]
CK_STEP = {"base": 0, "s32": 32, "s64": 64, "s96": 96, "s128": 128, "s160": 160, "s192": 192, "s224": 224, "s256": 256}
FRESH6 = ["advection1d", "burgers1d", "reaction_diffusion1d", "reaction_diffusion2d", "darcy2d", "incompressible_ns2d"]
TRIV = json.loads((C / "offline/heldout-trivial-errors.json").read_text())


def available(setname):
    return [ck for ck in CK_ORDER if (W / f"rows-{setname}-{ck}.jsonl").exists()]


def load(setname, ck, keep_code=True):
    rows = []
    for line in open(W / f"rows-{setname}-{ck}.jsonl"):
        r = json.loads(line)
        if setname == "heldout" and TRIV.get(r["case"], 1.0) <= 0.01:
            continue
        if setname == "fresh" and r["family"] not in FRESH6:
            continue
        if not keep_code:
            r.pop("code", None)
        rows.append(r)
    return rows


def by_case(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["case"]].append(r)
    for v in d.values():
        v.sort(key=lambda r: r["idx"])
    return d


def eq_family(values_by_family):
    """Equal-family mean of per-family means."""
    fams = [f for f in values_by_family if values_by_family[f]]
    return sum(sum(v) / len(v) for f, v in values_by_family.items() if v) / len(fams)

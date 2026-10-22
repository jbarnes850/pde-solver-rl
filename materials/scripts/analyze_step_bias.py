"""Paired step-size check for the decision map: uniform h = 0.25 against the map's h = 1 field stage.

Same mobility x0.5 specimens (701, 702, 801, 802) and three Curie offsets, run once with the map's
two-stage step cap and once at uniform h = 0.25 (onset_route_map.py --h-max 0.25 --h-aging 0.25
--t-starts 929 for the fast ramp and the 929 K start). Reports the final-utility difference, uniform
minus map, with its standard error over the 12 paired members. The fast ramp's +1.5 is the bias that
the decision analysis corrects for with --ramp-adjust 1.5 and 3.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map', type=Path, required=True, help='results.json of the map batch (mobility x0.5)')
    ap.add_argument('--uniform', type=Path, required=True, help='results.json of the uniform h = 0.25 rerun')
    args = ap.parse_args()
    key = lambda r: (r['dTc_K'], r['kinetic_multiplier'], r['seed'], r['route'], r['law'])  # noqa: E731
    base = {key(r): r for r in json.loads(args.map.read_text()) if r['final']}
    fine = [r for r in json.loads(args.uniform.read_text()) if r['final'] and r['law'] == 'full']
    report = {}
    for route in sorted({r['route'] for r in fine}):
        d = np.array([r['BHmax_kJ_m3'] - base[key(r)]['BHmax_kJ_m3'] for r in fine
                      if r['route'] == route and key(r) in base])
        report[route] = {'uniform_minus_map': float(d.mean()), 'se': float(d.std(ddof=1)/np.sqrt(len(d))), 'n': len(d)}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

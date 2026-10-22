"""Chord geometry of the final onset-study fields (independent of the spectral descriptor).

For each member: directional chord statistics of the magnetic alpha1 phase
(c < 0.5, Fe,Co-rich) along the field (x) and across it (y). The uncensored chord
ratio equals the geometric aspect for isolated aligned ellipses (control checked
in magnet.chords); for connected structures it is an orientation descriptor.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magnet.chords import controls, directional_chords  # noqa: E402
from magnet.physics import MaterialParams  # noqa: E402


def batch(folder):
    plan = json.loads((folder/'plan.json').read_text())
    g = plan['grid']
    dx, dy = g['Lx']/g['nx'], g['Ly']/g['ny']
    fields = np.load(folder/'fields-final.npz')['fields']
    assert len(fields) == len(plan['labels'])
    rows = []
    for label, params, c in zip(plan['labels'], plan['params'], fields):
        ell = MaterialParams(**params).ell*1e9
        r = directional_chords(c < .5, (dx*ell, dy*ell))
        along, across = r['directions']
        rows.append({**label, 'chord_ratio': r['uncensored_chord_ratio'],
                     'finite_chord_ratio': (along['mean_finite_chord_nm']/across['mean_finite_chord_nm']
                                                if along['mean_finite_chord_nm'] and across['mean_finite_chord_nm'] else None),
                     'chord_along_nm': along['mean_finite_chord_nm'], 'chord_across_nm': across['mean_finite_chord_nm'],
                     'censored_fraction': max(along['phase_fraction_on_infinite_lines'],
                                              across['phase_fraction_on_infinite_lines'])})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folders', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    controls()
    rows = [r for f in args.folders for r in batch(f)]
    args.out.write_text(json.dumps({'scope': __doc__, 'rows': rows}, indent=1))
    ok = [r['chord_ratio'] for r in rows if r['chord_ratio'] is not None]
    print(len(rows), 'members;', len(ok), 'uncensored; max chord ratio', max(ok) if ok else None)


if __name__ == '__main__':
    main()

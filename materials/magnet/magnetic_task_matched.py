"""Magnetic chemical potential posed in the RLVP training format (matched comparison).

Same physics, cases and variational check as magnetic_task.py. What changes is the
presentation, to match how the step-256 adapter was trained: the training system
prompt byte for byte, a `def solver(...)` contract with batched array input and an
exact output shape, the training code extractor, and no instruction against drafts.
The check calls `solver` directly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .magnetic_task import CONTROL as _CONTROL, cases, energy

_TRAIN_CONFIG = Path(__file__).resolve().parents[2]/'pipeline/trials/calib-train-base-p2-anchored-ext256.json'
if _TRAIN_CONFIG.exists():
    SYSTEM = json.loads(_TRAIN_CONFIG.read_text())['system_prompt']
else:  # staged copies carry the prompt beside this module
    SYSTEM = (Path(__file__).resolve().parent/'training_system_prompt.txt').read_text()
SYSTEM_SHA256 = hashlib.sha256(SYSTEM.encode()).hexdigest()

STATEMENT = """Compute the magnetic chemical potential of periodic two-dimensional composition fields.

For a composition field c(x, y) on a periodic grid with spacings dx and dy, the magnetic
free energy per grid cell is

    F_mag[c] = beta/2 * mean(g * A[g]) - h * mean(g),

where g = g(c, T) is the magnetization

    Tc(c) = Tc_a + (Tc_b - Tc_a) * c
    q = 1 - T / Tc(c)
    s = (q + sqrt(q * q + width * width)) / 2
    g = (1 - c) * sqrt(s)

and A is the linear operator with Fourier multiplier kx**2 / (kx**2 + ky**2), set to zero
at the zero wavevector, with kx = 2*pi*fftfreq(Nx, dx) and ky = 2*pi*fftfreq(Ny, dy).
Return the chemical potential mu defined by delta F_mag = mean(mu * delta c), including
the complete chain rule through g.

Write this function:

    def solver(c_batch, T, beta, h, Tc_a, Tc_b, width, dx, dy):
        ...

c_batch has shape [B, Nx, Ny] with 0 < c < 1; each member is an independent field.
T, beta, h, Tc_a, Tc_b, width, dx and dy are scalars. Return a finite NumPy array with
shape [B, Nx, Ny]. Do not hard-code a solution array.
"""

ARGS = ('c_batch', 'T', 'beta', 'h', 'Tc_a', 'Tc_b', 'width', 'dx', 'dy')
CONTROL = _CONTROL.replace('def magnetic_mu(c, T,', 'def solver(c, T,')


def extract(completion: str) -> str:
    """The RLVP training extractor: first fenced block if any, otherwise the whole text."""
    import re
    text = str(completion or '').strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def check(code, sandbox, additional=False):
    from .sandbox import Invocation

    rows = []
    for case in cases(additional):
        c, p = case['c'], case['scalars']
        r = sandbox.run(code, Invocation(ARGS, {'c_batch': c}, p, c.shape))
        row = {'case': case['name'], 'ok': r.ok, 'error': r.error, 'stderr': r.stderr_tail, 'seconds': r.seconds}
        if r.ok:
            errors = []
            for v in case['directions']:
                eps = 5e-6
                gradient = (energy(c+eps*v, p)-energy(c-eps*v, p))/(2*eps)
                observed = float(np.mean(r.output*v))
                errors.append(abs(observed-gradient)/max(1e-7, abs(gradient)))
            row.update({'max_relative_directional_error': float(max(errors)), 'passed': bool(max(errors) < 1e-4)})
            if case['name'] == 'zero_couplings':
                row['passed'] &= bool(abs(r.output).max() < 1e-12)
            if case['name'] == 'uniform':
                row['passed'] &= bool(r.output.std(axis=(-2, -1)).max() < 1e-12)
        else:
            row['passed'] = False
        rows.append(row)
    return {'passed': all(r['passed'] for r in rows), 'n_passed': sum(r['passed'] for r in rows),
            'cases': rows, 'feedback': '\n'.join(str(r) for r in rows if not r['passed']) or 'All checks passed.'}

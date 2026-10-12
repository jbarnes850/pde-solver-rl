"""Clean-room renderers for the seven paper-listed shallow-water methods.

The paper does not state its CFL, positivity floor, wet/dry treatment, HLL wave-speed
estimate, Roe entropy-fix value, or HLLC closure. The local renderers state these choices in
the generated code. The ``weno5`` codename is rendered as MUSCL-MC because that is the
spatial reconstruction printed in the paper table; it is an explicit unresolved label clash.
"""

from __future__ import annotations

from textwrap import dedent

METHOD_SPECS = {
    "fv_muscl_hll_ssprk3": ("hll", 3),
    "fv_muscl_rusanov_ssprk2": ("rusanov", 2),
    "fv_muscl_rusanov_ssprk3": ("rusanov", 3),
    "kt_central_upwind_muscl_ssprk3": ("kt", 3),
    # Paper table says MUSCL (MC), despite the WENO5 codename.
    "weno5_rusanov_ssprk3_pos": ("rusanov", 3),
    "roe_entropy_fix_muscl_ssprk3": ("roe", 3),
    "hllc_muscl_ssprk3_pos": ("hllc", 3),
}

METHODS = tuple(METHOD_SPECS)

AMBIGUITIES = {
    "all": "The author CFL, positivity, wet/dry, and Riemann-solver details are not public.",
    "weno5_rusanov_ssprk3_pos": (
        "The codename says WENO5, but the paper table says MUSCL-MC and cites no WENO "
        "reference. This renderer follows the table description."
    ),
    "kt_central_upwind_muscl_ssprk3": (
        "The local KT central-upwind flux uses one-sided HLL-form speeds."
    ),
}


_TEMPLATE = """
import numpy as np

_FLUX_KIND = "__FLUX_KIND__"
_RK_ORDER = __RK_ORDER__
_H_FLOOR = 1.0e-8       # Local choice; the paper does not publish its floor.
_CFL = 0.25             # Local choice; the paper does not publish its CFL.


def _fix(q):
    q = np.asarray(q, dtype=np.float64).copy()
    dry = q[..., 0] < _H_FLOOR
    q[..., 0] = np.maximum(q[..., 0], _H_FLOOR)
    q[..., 1] = np.where(dry, 0.0, q[..., 1])
    q[..., 2] = np.where(dry, 0.0, q[..., 2])
    return q


def _physical(q, grav, axis):
    h = np.maximum(q[..., 0], _H_FLOOR)
    hu, hv = q[..., 1], q[..., 2]
    u, v = hu / h, hv / h
    if axis == 1:
        return np.stack((hu, hu * u + 0.5 * grav * h * h, hu * v), axis=-1)
    return np.stack((hv, hv * u, hv * v + 0.5 * grav * h * h), axis=-1)


def _mc(dm, dc, dp):
    same = (np.sign(dm) == np.sign(dc)) & (np.sign(dc) == np.sign(dp))
    return np.where(same, np.sign(dc) * np.minimum(np.minimum(abs(dm), abs(dc)), abs(dp)), 0.0)


def _faces(q, axis):
    pads = ((0, 0), (1, 1), (1, 1), (0, 0))
    ext = np.pad(q, pads, mode="edge")
    slope = np.zeros_like(ext)
    if axis == 1:
        dm = 2.0 * (ext[:, 1:-1, :, :] - ext[:, :-2, :, :])
        dc = 0.5 * (ext[:, 2:, :, :] - ext[:, :-2, :, :])
        dp = 2.0 * (ext[:, 2:, :, :] - ext[:, 1:-1, :, :])
        slope[:, 1:-1, :, :] = _mc(dm, dc, dp)
        return ext[:, :-1, 1:-1, :] + 0.5 * slope[:, :-1, 1:-1, :], ext[:, 1:, 1:-1, :] - 0.5 * slope[:, 1:, 1:-1, :]
    dm = 2.0 * (ext[:, :, 1:-1, :] - ext[:, :, :-2, :])
    dc = 0.5 * (ext[:, :, 2:, :] - ext[:, :, :-2, :])
    dp = 2.0 * (ext[:, :, 2:, :] - ext[:, :, 1:-1, :])
    slope[:, :, 1:-1, :] = _mc(dm, dc, dp)
    return ext[:, 1:-1, :-1, :] + 0.5 * slope[:, 1:-1, :-1, :], ext[:, 1:-1, 1:, :] - 0.5 * slope[:, 1:-1, 1:, :]


def _normal(q, axis):
    h = np.maximum(q[..., 0], _H_FLOOR)
    mn = q[..., axis]
    mt = q[..., 3 - axis]
    return h, mn, mt, mn / h, mt / h


def _riemann(left, right, grav, axis):
    left, right = _fix(left), _fix(right)
    fl, fr = _physical(left, grav, axis), _physical(right, grav, axis)
    hl, mnl, mtl, unl, utl = _normal(left, axis)
    hr, mnr, mtr, unr, utr = _normal(right, axis)
    cl, cr = np.sqrt(grav * hl), np.sqrt(grav * hr)
    if _FLUX_KIND == "rusanov":
        speed = np.maximum(abs(unl) + cl, abs(unr) + cr)
        return 0.5 * (fl + fr) - 0.5 * speed[..., None] * (right - left)
    if _FLUX_KIND in ("hll", "kt"):
        minus = np.minimum(np.minimum(unl - cl, unr - cr), 0.0)
        plus = np.maximum(np.maximum(unl + cl, unr + cr), 0.0)
        width = np.maximum(plus - minus, 1.0e-14)
        return (plus[..., None] * fl - minus[..., None] * fr + (plus * minus)[..., None] * (right - left)) / width[..., None]
    if _FLUX_KIND == "hllc":
        sl, sr = np.minimum(unl - cl, unr - cr), np.maximum(unl + cl, unr + cr)
        pl, pr = 0.5 * grav * hl * hl, 0.5 * grav * hr * hr
        denominator = hl * (sl - unl) - hr * (sr - unr)
        ss = (pr - pl + hl * unl * (sl - unl) - hr * unr * (sr - unr)) / np.where(abs(denominator) > 1e-14, denominator, 1e-14)
        def star(q, h, ut, un, wave):
            hs = h * (wave - un) / np.where(abs(wave - ss) > 1e-14, wave - ss, 1e-14)
            normal, tangent = hs * ss, hs * ut
            return np.stack((hs, normal, tangent), axis=-1) if axis == 1 else np.stack((hs, tangent, normal), axis=-1)
        qsl, qsr = star(left, hl, utl, unl, sl), star(right, hr, utr, unr, sr)
        return np.where((sl >= 0)[..., None], fl, np.where((ss >= 0)[..., None], fl + sl[..., None] * (qsl - left), np.where((sr > 0)[..., None], fr + sr[..., None] * (qsr - right), fr)))
    # Roe flux with a local Harten entropy fix delta=0.1*c_bar.
    root_l, root_r = np.sqrt(hl), np.sqrt(hr)
    denom = root_l + root_r
    un = (root_l * unl + root_r * unr) / denom
    ut = (root_l * utl + root_r * utr) / denom
    c = np.sqrt(0.5 * grav * (hl + hr))
    dh, dmn, dmt = hr - hl, mnr - mnl, mtr - mtl
    a1 = ((un + c) * dh - dmn) / (2.0 * c)
    a3 = (dmn - (un - c) * dh) / (2.0 * c)
    a2 = dmt - ut * dh
    lam = [un - c, un, un + c]
    delta = np.maximum(0.1 * c, 1e-12)
    lam = [np.where(abs(z) < delta, 0.5 * (z*z / delta + delta), abs(z)) for z in lam]
    d0 = lam[0] * a1 + lam[2] * a3
    dn = lam[0] * a1 * (un - c) + lam[2] * a3 * (un + c)
    dt = lam[0] * a1 * ut + lam[1] * a2 + lam[2] * a3 * ut
    diss = np.stack((d0, dn, dt), axis=-1) if axis == 1 else np.stack((d0, dt, dn), axis=-1)
    return 0.5 * (fl + fr - diss)


def _rhs(q, grav, dx, dy):
    lx, rx = _faces(q, 1); ly, ry = _faces(q, 2)
    fx, fy = _riemann(lx, rx, grav, 1), _riemann(ly, ry, grav, 2)
    return -(fx[:, 1:] - fx[:, :-1]) / dx - (fy[:, :, 1:] - fy[:, :, :-1]) / dy


def _step(q, dt, grav, dx, dy):
    if _RK_ORDER == 2:
        q1 = _fix(q + dt * _rhs(q, grav, dx, dy))
        return _fix(0.5 * q + 0.5 * (q1 + dt * _rhs(q1, grav, dx, dy)))
    q1 = _fix(q + dt * _rhs(q, grav, dx, dy))
    q2 = _fix(0.75 * q + 0.25 * (q1 + dt * _rhs(q1, grav, dx, dy)))
    return _fix(q / 3.0 + 2.0 * (q2 + dt * _rhs(q2, grav, dx, dy)) / 3.0)


def solver(h0_batch, hu0_batch, hv0_batch, t_coordinate, grav):
    h, hu, hv = (np.asarray(x, dtype=np.float64) for x in (h0_batch, hu0_batch, hv0_batch))
    times = np.asarray(t_coordinate, dtype=np.float64)
    if h.shape != hu.shape or h.shape != hv.shape or h.ndim != 3:
        raise ValueError("initial fields must have the same [B,Nx,Ny] shape")
    if times.ndim != 1 or np.any(np.diff(times) < 0.0):
        raise ValueError("t_coordinate must be one-dimensional and nondecreasing")
    q = _fix(np.stack((h, hu, hv), axis=-1))
    nx, ny = h.shape[1:]
    dx, dy = 5.0 / (nx - 1), 5.0 / (ny - 1)
    out = np.empty((h.shape[0], times.size, nx, ny, 3), dtype=np.float64)
    current = 0.0
    for index, target in enumerate(times):
        while current < target - 1e-14:
            speed = max(float(np.max(abs(q[..., 1] / q[..., 0]) + np.sqrt(float(grav) * q[..., 0]))), float(np.max(abs(q[..., 2] / q[..., 0]) + np.sqrt(float(grav) * q[..., 0]))), 1e-12)
            dt = min(target - current, _CFL * min(dx, dy) / speed)
            q = _step(q, dt, float(grav), dx, dy)
            current += dt
        out[:, index] = q
    return out
"""


def render(codename: str) -> str:
    """Return standalone solver code for one exact paper codename."""

    try:
        flux_kind, rk_order = METHOD_SPECS[codename]
    except KeyError as exc:
        raise KeyError(f"unknown shallow_water2d solver-bank codename: {codename}") from exc
    header = ""
    if codename == "weno5_rusanov_ssprk3_pos":
        header = (
            "# The paper codename says WENO5, but its table specifies MUSCL-MC.\n"
            "# This source-derived renderer follows the published table description.\n"
        )
    return (
        header
        + dedent(_TEMPLATE)
        .replace("__FLUX_KIND__", flux_kind)
        .replace("__RK_ORDER__", str(rk_order))
        .lstrip()
    )


__all__ = ["AMBIGUITIES", "METHODS", "METHOD_SPECS", "render"]

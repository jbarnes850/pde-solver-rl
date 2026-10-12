"""Clean-room renderers for the ten paper-listed incompressible-flow methods.

The paper specifies method families but does not release its projection operator, pressure
boundary convention, DST normalization, time-step rule, or tolerances. These renderers use
a collocated endpoint grid, centered differences, zero pressure on the DST/Jacobi boundary,
and explicit local constants. The paper calls two methods MAC aliases while describing DST
projection, so those rows are intentionally rendered as the stated DST aliases rather than
as unsubstantiated staggered-grid methods.
"""

from __future__ import annotations

from textwrap import dedent

METHOD_SPECS = {
    "chorin_centered_euler": ("centered", "euler", "jacobi"),
    "chorin_centered_rk2": ("centered", "rk2", "jacobi"),
    "chorin_skewsym_euler": ("skew", "euler", "jacobi"),
    "semiimplicit_centered_euler": ("centered", "imex_euler", "jacobi"),
    "imex_cnab2_skewsym": ("skew", "cnab2", "jacobi"),
    # The paper does not select centered or skew advection for this row; local choice: skew.
    "dst_projection_cnab2_runtime": ("skew", "cnab2", "dst"),
    "imex_cnab2_skewsym_spectral": ("skew", "cnab2", "dst"),
    "semiimplicit_centered_spectral": ("centered", "imex_euler", "dst"),
    # Paper table describes both MAC codenames as DST-projection aliases.
    "mac_projection_centered": ("centered", "imex_euler", "dst"),
    "mac_projection_rk2": ("skew", "cnab2", "dst"),
}

METHODS = tuple(METHOD_SPECS)

AMBIGUITIES = {
    "all": (
        "The pressure boundary condition, collocated operators, DST convention, CFL, and "
        "linear-solver tolerances are local because the author code is not public."
    ),
    "dst_projection_cnab2_runtime": (
        "The paper says centered/skew-symmetric but does not select one. This renderer uses "
        "skew-symmetric advection."
    ),
    "mac_projection_centered": (
        "The paper calls this a MAC method but describes a DST-projection alias and does not "
        "define a staggered grid. This renderer follows the DST description."
    ),
    "mac_projection_rk2": (
        "The paper calls this a MAC method but describes a DST-projection alias and does not "
        "define a staggered grid. This renderer follows the DST description."
    ),
}


_TEMPLATE = """
import numpy as np
from scipy.fft import dstn, idstn

_ADVECTION = "__ADVECTION__"
_INTEGRATOR = "__INTEGRATOR__"
_PRESSURE = "__PRESSURE__"
_CFL = 0.20  # Local choice; the paper does not publish its step rule.


def _boundary(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[:, 0, :, :] = q[:, -1, :, :] = 0.0
    q[:, :, 0, :] = q[:, :, -1, :] = 0.0
    return q


def _lap(q, dx, dy):
    out = np.zeros_like(q)
    out[:, 1:-1, 1:-1] = ((q[:, 2:, 1:-1] - 2*q[:, 1:-1, 1:-1] + q[:, :-2, 1:-1]) / dx**2 + (q[:, 1:-1, 2:] - 2*q[:, 1:-1, 1:-1] + q[:, 1:-1, :-2]) / dy**2)
    return out


def _gradient(s, dx, dy):
    out = np.zeros(s.shape + (2,), dtype=np.float64)
    out[:, 1:-1, 1:-1, 0] = (s[:, 2:, 1:-1] - s[:, :-2, 1:-1]) / (2*dx)
    out[:, 1:-1, 1:-1, 1] = (s[:, 1:-1, 2:] - s[:, 1:-1, :-2]) / (2*dy)
    return out


def _divergence(q, dx, dy):
    out = np.zeros(q.shape[:-1], dtype=np.float64)
    out[:, 1:-1, 1:-1] = ((q[:, 2:, 1:-1, 0] - q[:, :-2, 1:-1, 0]) / (2*dx) + (q[:, 1:-1, 2:, 1] - q[:, 1:-1, :-2, 1]) / (2*dy))
    return out


def _advection(q, dx, dy):
    out = np.zeros_like(q)
    u, v = q[..., 0], q[..., 1]
    for component in range(2):
        w = q[..., component]
        wx = (w[:, 2:, 1:-1] - w[:, :-2, 1:-1]) / (2*dx)
        wy = (w[:, 1:-1, 2:] - w[:, 1:-1, :-2]) / (2*dy)
        advective = u[:, 1:-1, 1:-1] * wx + v[:, 1:-1, 1:-1] * wy
        if _ADVECTION == "centered":
            out[:, 1:-1, 1:-1, component] = advective
        else:
            uw, vw = u*w, v*w
            conservative = ((uw[:, 2:, 1:-1] - uw[:, :-2, 1:-1]) / (2*dx) + (vw[:, 1:-1, 2:] - vw[:, 1:-1, :-2]) / (2*dy))
            out[:, 1:-1, 1:-1, component] = 0.5 * (advective + conservative)
    return out


def _dst_solve(rhs, dx, dy, alpha, poisson=False):
    nx, ny = rhs.shape[1:3]
    ix = np.arange(1, nx - 1)
    iy = np.arange(1, ny - 1)
    mu = (4*np.sin(np.pi*ix/(2*(nx-1)))**2 / dx**2)[:, None] + (4*np.sin(np.pi*iy/(2*(ny-1)))**2 / dy**2)[None, :]
    axes = (-2, -1) if rhs.ndim == 3 else (-3, -2)
    interior = rhs[:, 1:-1, 1:-1] if rhs.ndim == 3 else rhs[:, 1:-1, 1:-1, :]
    transformed = dstn(interior, type=1, axes=axes, norm="ortho")
    denominator = mu[None] if rhs.ndim == 3 else mu[None, :, :, None]
    solved = -transformed / denominator if poisson else transformed / (1.0 + alpha*denominator)
    values = idstn(solved, type=1, axes=axes, norm="ortho")
    out = np.zeros_like(rhs); out[:, 1:-1, 1:-1] = values
    return out


def _jacobi_poisson(rhs, dx, dy):
    p = np.zeros_like(rhs)
    scale = 2.0*(dx*dx + dy*dy)
    for _ in range(1200):
        old = p
        p = np.zeros_like(rhs)
        p[:, 1:-1, 1:-1] = ((old[:, 2:, 1:-1] + old[:, :-2, 1:-1])*dy*dy + (old[:, 1:-1, 2:] + old[:, 1:-1, :-2])*dx*dx - rhs[:, 1:-1, 1:-1]*dx*dx*dy*dy) / scale
        if np.max(abs(p-old)) < 1e-10:
            break
    return p


def _helmholtz(rhs, alpha, dx, dy):
    if _PRESSURE == "dst":
        return _dst_solve(rhs, dx, dy, alpha)
    q = np.zeros_like(rhs)
    diagonal = 1.0 + 2.0*alpha*(1.0/dx**2 + 1.0/dy**2)
    for _ in range(1200):
        old = q
        q = np.zeros_like(rhs)
        q[:, 1:-1, 1:-1] = (rhs[:, 1:-1, 1:-1] + alpha*((old[:, 2:, 1:-1] + old[:, :-2, 1:-1])/dx**2 + (old[:, 1:-1, 2:] + old[:, 1:-1, :-2])/dy**2)) / diagonal
        if np.max(abs(q-old)) < 1e-10:
            break
    return q


def _project(q, dt, dx, dy):
    divergence = _divergence(q, dx, dy) / dt
    pressure = _dst_solve(divergence, dx, dy, 0.0, poisson=True) if _PRESSURE == "dst" else _jacobi_poisson(divergence, dx, dy)
    return _boundary(q - dt * _gradient(pressure, dx, dy))


def _advance(q, force, dt, nu, dx, dy, previous):
    nonlinear = -_advection(q, dx, dy) + force
    if _INTEGRATOR == "euler":
        return _project(_boundary(q + dt*(nonlinear + nu*_lap(q, dx, dy))), dt, dx, dy), nonlinear
    if _INTEGRATOR == "rk2":
        first = nonlinear + nu*_lap(q, dx, dy)
        stage = _project(_boundary(q + dt*first), dt, dx, dy)
        second = -_advection(stage, dx, dy) + force + nu*_lap(stage, dx, dy)
        return _project(_boundary(q + 0.5*dt*(first + second)), dt, dx, dy), nonlinear
    if _INTEGRATOR == "imex_euler":
        tentative = _helmholtz(_boundary(q + dt*nonlinear), dt*nu, dx, dy)
        return _project(tentative, dt, dx, dy), nonlinear
    explicit = nonlinear if previous is None else 1.5*nonlinear - 0.5*previous
    rhs = _boundary(q + 0.5*dt*nu*_lap(q, dx, dy) + dt*explicit)
    tentative = _helmholtz(rhs, 0.5*dt*nu, dx, dy)
    return _project(tentative, dt, dx, dy), nonlinear


def solver(velocity0_batch, forcing_batch, t_coordinate, nu):
    q = np.asarray(velocity0_batch, dtype=np.float64)
    force = np.asarray(forcing_batch, dtype=np.float64)
    times = np.asarray(t_coordinate, dtype=np.float64)
    if q.ndim != 4 or q.shape[-1] != 2 or force.shape != q.shape:
        raise ValueError("velocity and forcing must have shape [B,Nx,Ny,2]")
    if times.ndim != 1 or np.any(np.diff(times) < 0.0):
        raise ValueError("t_coordinate must be one-dimensional and nondecreasing")
    q, force = _boundary(q), _boundary(force)
    nx, ny = q.shape[1:3]; dx, dy = 1.0/(nx-1), 1.0/(ny-1)
    out = np.empty((q.shape[0], times.size, nx, ny, 2), dtype=np.float64)
    current, previous = 0.0, None
    for index, target in enumerate(times):
        while current < target - 1e-14:
            speed = max(float(np.max(np.linalg.norm(q, axis=-1))), 1e-12)
            dt = min(target-current, _CFL*min(dx,dy)/speed)
            if _INTEGRATOR in ("euler", "rk2") and float(nu) > 0:
                dt = min(dt, _CFL/(float(nu)*(1.0/dx**2 + 1.0/dy**2)))
            q, previous = _advance(q, force, dt, float(nu), dx, dy, previous)
            current += dt
        out[:, index] = q
    return out
"""


def render(codename: str) -> str:
    """Return standalone solver code for one exact paper codename."""

    try:
        advection, integrator, pressure = METHOD_SPECS[codename]
    except KeyError as exc:
        raise KeyError(f"unknown incompressible_ns2d solver-bank codename: {codename}") from exc
    header = ""
    if codename.startswith("mac_projection"):
        header = (
            "# The paper describes this MAC codename as a DST projection alias.\n"
            "# No staggered-grid contract is published; this renderer follows that alias.\n"
        )
    if codename == "dst_projection_cnab2_runtime":
        header = (
            "# The paper leaves centered versus skew advection open for this row.\n"
            "# This source-derived renderer selects skew-symmetric advection.\n"
        )
    return (
        header
        + dedent(_TEMPLATE)
        .replace("__ADVECTION__", advection)
        .replace("__INTEGRATOR__", integrator)
        .replace("__PRESSURE__", pressure)
        .lstrip()
    )


__all__ = ["AMBIGUITIES", "METHODS", "METHOD_SPECS", "render"]

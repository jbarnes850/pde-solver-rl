from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.sparse import eye, lil_matrix
from scipy.sparse.linalg import splu

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import periodic_grid, periodic_second, time_grid

# The paper names heat1d as held out and shows nu=0.1 with a single-sine input. It does not
# publish a complete case distribution. The remaining values below are source-derived choices.
SOURCE_DERIVED_NUS = (0.02, 0.05, 0.1, 0.2)
SOURCE_DERIVED_GRID_SIZES = (64, 128)
SOURCE_DERIVED_TIME_COUNTS = (51, 101)
SOURCE_DERIVED_FINAL_TIME = 1.0
IC_KINDS = ("single_sine", "two_mode", "multimode")


def _float(case: PDECase, name: str) -> float:
    return float(case.parameters[name])


def x_coordinate(case: PDECase) -> np.ndarray:
    return periodic_grid(case.grid_shape[0])


def t_coordinate(case: PDECase) -> np.ndarray:
    if case.time_count is None or case.final_time is None:
        raise ValueError("heat1d requires time_count and final_time")
    return time_grid(case.time_count, case.final_time)


def _components(case: PDECase) -> tuple[tuple[float, int, str, float], ...]:
    amplitude = _float(case, "amplitude")
    phase = _float(case, "phase")
    if case.input_kind == "single_sine":
        return ((amplitude, 1, "sin", phase),)
    if case.input_kind == "two_mode":
        return (
            (amplitude, 1, "sin", phase),
            (amplitude * _float(case, "mix"), 2, "sin", -0.5 * phase),
        )
    if case.input_kind == "multimode":
        return (
            (amplitude, 1, "sin", phase),
            (0.30 * amplitude, 3, "sin", -phase),
            (0.15 * amplitude, 5, "cos", 0.25 * phase),
        )
    raise ValueError(f"unknown heat1d input kind: {case.input_kind}")


def initial_condition(case: PDECase) -> np.ndarray:
    x = x_coordinate(case)
    values = np.full_like(x, _float(case, "offset"))
    for amplitude, mode, kind, phase in _components(case):
        angle = 2.0 * np.pi * mode * x + phase
        values += amplitude * (np.sin(angle) if kind == "sin" else np.cos(angle))
    return values


def make_case(
    *, seed: int, split: str = "heldout_eval", stratum: dict[str, Any] | None = None
) -> PDECase:
    rng = np.random.default_rng(seed)
    selected: dict[str, Any] = {
        "nu": float(rng.choice(SOURCE_DERIVED_NUS)),
        "nx": int(rng.choice(SOURCE_DERIVED_GRID_SIZES)),
        "nt": int(rng.choice(SOURCE_DERIVED_TIME_COUNTS)),
        "final_time": SOURCE_DERIVED_FINAL_TIME,
        "input_kind": str(rng.choice(IC_KINDS)),
    }
    if stratum:
        selected.update(stratum)
    parameters: dict[str, int | float | str] = {
        "nu": float(selected["nu"]),
        "phase": float(rng.uniform(-np.pi, np.pi)),
        "amplitude": float(rng.uniform(0.4, 1.2)),
        "offset": float(rng.uniform(-0.2, 0.2)),
        "mix": float(rng.uniform(0.2, 0.6)),
    }
    return make_case_record(
        family_id="heat1d",
        split=split,
        seed=seed,
        grid_shape=(int(selected["nx"]),),
        time_count=int(selected["nt"]),
        final_time=float(selected["final_time"]),
        input_kind=str(selected["input_kind"]),
        parameters=parameters,
    )


def prompt(case: PDECase, *, form: str = "generic") -> str:
    if form not in {"generic", "parameter", "parameter_ic"}:
        raise ValueError(f"unknown prompt form: {form}")
    conditioning = ""
    if form in {"parameter", "parameter_ic"}:
        conditioning += f"\nFor this task, nu={_float(case, 'nu')}."
    if form == "parameter_ic":
        conditioning += f"\nThe initial-condition family is {case.input_kind}."
    return f"""Define exactly:
    def solver(u0_batch, t_coordinate, nu):

Solve the periodic one-dimensional heat equation

    u_t = nu * u_xx,  x in [0, 1).

u0_batch has shape [B, N]. t_coordinate has shape [T]. Return a finite NumPy
array with shape [B, T, N]. Returned slice n must correspond to
t_coordinate[n]. Internal substeps are allowed. Use a stable numerical method.
Do not hard-code a solution array.{conditioning}
"""


def invocation(case: PDECase) -> SolverInvocation:
    u0 = initial_condition(case)[None, :]
    times = t_coordinate(case)
    return SolverInvocation(
        argument_order=("u0_batch", "t_coordinate", "nu"),
        arrays={"u0_batch": u0, "t_coordinate": times},
        scalars={"nu": _float(case, "nu")},
        expected_shape=(1, len(times), case.grid_shape[0]),
    )


def reference_primary(case: PDECase) -> np.ndarray:
    """Analytical decay of the finite Fourier series used by the case generator."""
    x = x_coordinate(case)
    times = t_coordinate(case)
    nu = _float(case, "nu")
    output = np.full((1, len(times), len(x)), _float(case, "offset"), dtype=np.float64)
    for amplitude, mode, kind, phase in _components(case):
        angle = 2.0 * np.pi * mode * x[None, :] + phase
        spatial = np.sin(angle) if kind == "sin" else np.cos(angle)
        decay = np.exp(-nu * (2.0 * np.pi * mode) ** 2 * times[:, None])
        output[0] += amplitude * decay * spatial
    return output


def reference_audit(case: PDECase) -> np.ndarray:
    """Independent periodic finite-difference Crank-Nicolson reference."""
    nx = case.grid_shape[0]
    spacing = 1.0 / nx
    times = t_coordinate(case)
    nu = _float(case, "nu")
    laplacian = lil_matrix((nx, nx), dtype=np.float64)
    rows = np.arange(nx)
    laplacian[rows, rows] = -2.0 / spacing**2
    laplacian[rows, (rows - 1) % nx] = 1.0 / spacing**2
    laplacian[rows, (rows + 1) % nx] = 1.0 / spacing**2
    laplacian = laplacian.tocsc()
    identity = eye(nx, format="csc")
    state = initial_condition(case).astype(np.float64, copy=True)
    output = np.empty((1, len(times), nx), dtype=np.float64)
    output[0, 0] = state
    for index, (start, target) in enumerate(pairwise(times), start=1):
        interval = float(target - start)
        steps = max(1, math.ceil(interval / 0.01))
        step = interval / steps
        left = identity - 0.5 * step * nu * laplacian
        right = identity + 0.5 * step * nu * laplacian
        factor = splu(left)
        for _ in range(steps):
            state = factor.solve(right @ state)
        output[0, index] = state
    return output


def residual_norm(field: np.ndarray, case: PDECase) -> float:
    values = np.asarray(field, dtype=np.float64)
    times = t_coordinate(case)
    if values.shape != (1, len(times), case.grid_shape[0]):
        raise ValueError("heat1d residual received an invalid shape")
    du_dt = np.gradient(values, times, axis=1, edge_order=2)
    laplacian = periodic_second(values, 1.0 / case.grid_shape[0], axis=2)
    residual = du_dt - _float(case, "nu") * laplacian
    return float(np.sqrt(np.mean(np.square(residual))))


class Heat1DEnvironment:
    family_id = "heat1d"
    seen_in_training = False
    method_ids: tuple[str, ...] = ()

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        if case.family_id != "heat1d":
            raise ValueError(f"wrong family for heat1d: {case.family_id}")
        return case

    make_case = staticmethod(make_case)
    prompt = staticmethod(prompt)
    invocation = staticmethod(invocation)
    reference_primary = staticmethod(reference_primary)
    reference_audit = staticmethod(reference_audit)
    residual_norm = staticmethod(residual_norm)

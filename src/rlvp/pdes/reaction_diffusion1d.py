from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import periodic_grid, periodic_second, time_grid

PAPER_NUS = (0.5, 1.0, 2.0, 5.0)
PAPER_RHOS = (1.0, 2.0, 5.0, 10.0)
PAPER_GRID_SIZES = (64, 128)
PAPER_TIME_COUNTS = (51, 101)
PAPER_FINAL_TIME = 1.0
IC_KINDS = ("single_sine", "multimode", "front_like")

REACTION_DIFFUSION_METHOD_IDS = (
    "fd_explicit_euler",
    "fd_explicit_rk2",
    "fd_explicit_rk4",
    "split_strang_fd_rk2_pes",
    "spectral_strang_exact_reaction",
    "spectral_imex_exact_diffusion",
)


def _float(case: PDECase, name: str) -> float:
    return float(case.parameters[name])


def x_coordinate(case: PDECase) -> np.ndarray:
    return periodic_grid(case.grid_shape[0])


def t_coordinate(case: PDECase) -> np.ndarray:
    if case.time_count is None or case.final_time is None:
        raise ValueError("reaction_diffusion1d requires time_count and final_time")
    return time_grid(case.time_count, case.final_time)


def initial_condition(case: PDECase) -> np.ndarray:
    x = x_coordinate(case)
    phase = _float(case, "phase")
    baseline = _float(case, "baseline")
    amplitude = _float(case, "amplitude")
    if case.input_kind == "single_sine":
        values = baseline + amplitude * np.sin(2.0 * np.pi * x + phase)
    elif case.input_kind == "multimode":
        mix = _float(case, "mix")
        values = baseline + amplitude * (
            np.sin(2.0 * np.pi * x + phase) + mix * np.sin(6.0 * np.pi * x - 0.5 * phase)
        )
    elif case.input_kind == "front_like":
        center = _float(case, "center")
        width = _float(case, "width")
        steepness = _float(case, "steepness")
        distance = np.minimum(np.abs(x - center), 1.0 - np.abs(x - center))
        values = baseline + amplitude * 0.5 * (1.0 - np.tanh(steepness * (distance - width)))
    else:
        raise ValueError(f"unknown reaction_diffusion1d input kind: {case.input_kind}")
    return np.clip(values, 0.02, 0.98)


def make_case(
    *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
) -> PDECase:
    rng = np.random.default_rng(seed)
    selected: dict[str, Any] = {
        "nu": float(rng.choice(PAPER_NUS)),
        "rho": float(rng.choice(PAPER_RHOS)),
        "nx": int(rng.choice(PAPER_GRID_SIZES)),
        "nt": int(rng.choice(PAPER_TIME_COUNTS)),
        "final_time": PAPER_FINAL_TIME,
        "input_kind": str(rng.choice(IC_KINDS)),
    }
    if stratum:
        selected.update(stratum)
    parameters: dict[str, int | float | str] = {
        "nu": float(selected["nu"]),
        "rho": float(selected["rho"]),
        "phase": float(rng.uniform(-np.pi, np.pi)),
        "baseline": float(rng.uniform(0.35, 0.65)),
        "amplitude": float(rng.uniform(0.08, 0.22)),
        "mix": float(rng.uniform(0.15, 0.35)),
        "center": float(rng.uniform(0.25, 0.75)),
        "width": float(rng.uniform(0.12, 0.25)),
        "steepness": float(rng.uniform(20.0, 35.0)),
    }
    return make_case_record(
        family_id="reaction_diffusion1d",
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
        conditioning += f"\nFor this task, nu={_float(case, 'nu')} and rho={_float(case, 'rho')}."
    if form == "parameter_ic":
        conditioning += f"\nThe initial-condition family is {case.input_kind}."
    return f"""Define exactly:
    def solver(u0_batch, t_coordinate, nu, rho):

Solve the periodic one-dimensional Fisher-KPP reaction-diffusion equation

    u_t = nu * u_xx + rho * u * (1 - u),  x in [0, 1).

u0_batch has shape [B, N]. t_coordinate has shape [T]. Return a finite NumPy
array with shape [B, T, N]. Returned slice n must correspond to
t_coordinate[n]. Internal substeps are allowed. Use a stable numerical method.
Do not hard-code a solution array.{conditioning}
"""


def invocation(case: PDECase) -> SolverInvocation:
    u0 = initial_condition(case)[None, :]
    times = t_coordinate(case)
    return SolverInvocation(
        argument_order=("u0_batch", "t_coordinate", "nu", "rho"),
        arrays={"u0_batch": u0, "t_coordinate": times},
        scalars={"nu": _float(case, "nu"), "rho": _float(case, "rho")},
        expected_shape=(1, len(times), case.grid_shape[0]),
    )


def _logistic_step(values: np.ndarray, rho: float, step: float) -> np.ndarray:
    growth = np.exp(rho * step)
    return values * growth / (1.0 + values * (growth - 1.0))


def reference_primary(case: PDECase) -> np.ndarray:
    """Spectral Strang split reference with exact diffusion and reaction flows."""
    nx = case.grid_shape[0]
    times = t_coordinate(case)
    nu = _float(case, "nu")
    rho = _float(case, "rho")
    wave = 2.0 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)
    state = initial_condition(case).astype(np.float64, copy=True)
    output = np.empty((1, len(times), nx), dtype=np.float64)
    output[0, 0] = state
    now = float(times[0])
    max_step = min(0.002, 0.05 / rho)
    for index, target in enumerate(times[1:], start=1):
        while now < float(target) - 1e-14:
            step = min(max_step, float(target) - now)
            state = _logistic_step(state, rho, 0.5 * step)
            modes = np.fft.fft(state)
            state = np.fft.ifft(modes * np.exp(-nu * wave**2 * step)).real
            state = _logistic_step(state, rho, 0.5 * step)
            now += step
        output[0, index] = state
    return output


def reference_audit(case: PDECase) -> np.ndarray:
    """Independent periodic finite-difference method-of-lines BDF reference."""
    nx = case.grid_shape[0]
    spacing = 1.0 / nx
    times = t_coordinate(case)
    nu = _float(case, "nu")
    rho = _float(case, "rho")

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        return nu * periodic_second(state, spacing) + rho * state * (1.0 - state)

    sparsity = lil_matrix((nx, nx), dtype=np.int8)
    rows = np.arange(nx)
    sparsity[rows, rows] = 1
    sparsity[rows, (rows - 1) % nx] = 1
    sparsity[rows, (rows + 1) % nx] = 1
    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        initial_condition(case),
        method="BDF",
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
        jac_sparsity=sparsity.tocsr(),
        max_step=0.02,
    )
    if not solution.success:
        raise RuntimeError(f"reaction_diffusion1d audit reference failed: {solution.message}")
    return solution.y.T[None, :, :]


def residual_norm(field: np.ndarray, case: PDECase) -> float:
    values = np.asarray(field, dtype=np.float64)
    times = t_coordinate(case)
    if values.shape != (1, len(times), case.grid_shape[0]):
        raise ValueError("reaction_diffusion1d residual received an invalid shape")
    du_dt = np.gradient(values, times, axis=1, edge_order=2)
    laplacian = periodic_second(values, 1.0 / case.grid_shape[0], axis=2)
    residual = du_dt - _float(case, "nu") * laplacian
    residual -= _float(case, "rho") * values * (1.0 - values)
    return float(np.sqrt(np.mean(np.square(residual))))


class ReactionDiffusion1DEnvironment:
    family_id = "reaction_diffusion1d"
    seen_in_training = True
    method_ids = REACTION_DIFFUSION_METHOD_IDS

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        if case.family_id != "reaction_diffusion1d":
            raise ValueError(f"wrong family for reaction_diffusion1d: {case.family_id}")
        return case

    make_case = staticmethod(make_case)
    prompt = staticmethod(prompt)
    invocation = staticmethod(invocation)
    reference_primary = staticmethod(reference_primary)
    reference_audit = staticmethod(reference_audit)
    residual_norm = staticmethod(residual_norm)

from __future__ import annotations

"""Source-derived 2D two-species reaction--diffusion environment.

The equation, Neumann boundary condition, domain, parameter values, grid sizes, output
times, final time, initial-family names, output shape, and method names come from
arXiv:2607.10474v1. The paper does not report the initial-field formulas, their sampling
weights, the solver call signature, reference tolerances, or residual stencil. Those items
are explicit source-derived choices in this module.
"""

from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import integrate_to_times, neumann_second, rk4_step, time_grid

PAPER_DU = (0.001, 0.002)
PAPER_DV = (0.005,)
PAPER_K = (0.005,)
PAPER_GRID_SIZES = (32, 48)
PAPER_TIME_COUNTS = (31, 51)
PAPER_FINAL_TIME = 5.0
PAPER_INPUT_KINDS = ("gaussian_noise", "blob_pair", "striped_noise")

REACTION_DIFFUSION_2D_METHOD_IDS = (
    "fd_explicit_euler",
    "fd_explicit_rk2",
    "fd_explicit_rk4",
    "strang_rk4_reaction_cn_sparse",
    "dct_neumann_strang_split",
    "dct_neumann_imex_rk2",
)

# Source-derived numerical choices because arXiv v1 does not report them.
SOURCE_DERIVED_BATCH_SIZE = 1
SOURCE_DERIVED_PRIMARY_MAX_STEP = 0.01
SOURCE_DERIVED_AUDIT_MAX_STEP = 0.005


def _required_time_data(case: PDECase) -> tuple[int, float]:
    if case.time_count is None or case.final_time is None:
        raise ValueError("reaction_diffusion2d requires time_count and final_time")
    return case.time_count, case.final_time


def coordinates(case: PDECase) -> np.ndarray:
    return np.linspace(-1.0, 1.0, case.grid_shape[0], dtype=np.float64)


def times(case: PDECase) -> np.ndarray:
    count, final_time = _required_time_data(case)
    return time_grid(count, final_time)


def initial_state(case: PDECase) -> np.ndarray:
    """Generate one deterministic hidden initial field in paper output channel order (u, v)."""
    n = case.grid_shape[0]
    x = coordinates(case)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rng = np.random.default_rng(case.seed)

    if case.input_kind == "gaussian_noise":
        u = -0.45 + 0.12 * rng.standard_normal((n, n))
        v = -0.50 + 0.08 * rng.standard_normal((n, n))
    elif case.input_kind == "blob_pair":
        first = np.exp(-((xx + 0.38) ** 2 + (yy + 0.16) ** 2) / 0.055)
        second = np.exp(-((xx - 0.30) ** 2 + (yy - 0.24) ** 2) / 0.075)
        u = -0.65 + 1.05 * first - 0.75 * second
        v = -0.45 + 0.45 * second + 0.08 * first
    elif case.input_kind == "striped_noise":
        phase = float(rng.uniform(-np.pi, np.pi))
        u = -0.50 + 0.28 * np.sin(3.0 * np.pi * xx + phase)
        u += 0.06 * rng.standard_normal((n, n))
        v = -0.48 + 0.18 * np.cos(2.0 * np.pi * yy - 0.5 * phase)
        v += 0.04 * rng.standard_normal((n, n))
    else:
        raise ValueError(f"unknown reaction_diffusion2d input kind: {case.input_kind}")
    return np.stack((u, v), axis=-1)[None, ...]


def _reaction(state: np.ndarray, k: float) -> np.ndarray:
    u = state[..., 0]
    v = state[..., 1]
    return np.stack((u - u**3 - k - v, u - v), axis=-1)


def _fd_rhs(state: np.ndarray, *, du: float, dv: float, k: float, dx: float) -> np.ndarray:
    laplacian = neumann_second(state, dx, axis=0) + neumann_second(state, dx, axis=1)
    diffusion = laplacian * np.array((du, dv), dtype=np.float64)
    return diffusion + _reaction(state, k)


def _primary_reference(case: PDECase, max_step: float) -> np.ndarray:
    state = initial_state(case)[0]
    dx = 2.0 / (case.grid_shape[0] - 1)
    du = float(case.parameters["du"])
    dv = float(case.parameters["dv"])
    k = float(case.parameters["k"])
    stable_step = min(max_step, 0.15 * dx**2 / max(du, dv))
    result = integrate_to_times(
        state,
        times(case),
        lambda value: _fd_rhs(value, du=du, dv=dv, k=k, dx=dx),
        stable_step,
    )
    return result[None, ...]


def _neumann_eigendecomposition(n: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize the symmetric edge-padded Neumann difference operator."""
    operator = np.zeros((n, n), dtype=np.float64)
    diagonal = np.full(n, -2.0)
    diagonal[[0, -1]] = -1.0
    np.fill_diagonal(operator, diagonal)
    indices = np.arange(n - 1)
    operator[indices, indices + 1] = 1.0
    operator[indices + 1, indices] = 1.0
    return np.linalg.eigh(operator / dx**2)


def _exact_neumann_diffusion(
    state: np.ndarray,
    step: float,
    diffusivities: tuple[float, float],
    eigenvalues: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(state)
    laplacian_eigenvalues = eigenvalues[:, None] + eigenvalues[None, :]
    for channel, diffusivity in enumerate(diffusivities):
        coefficients = basis.T @ state[..., channel] @ basis
        coefficients *= np.exp(diffusivity * step * laplacian_eigenvalues)
        output[..., channel] = basis @ coefficients @ basis.T
    return output


def _audit_reference(case: PDECase, max_step: float) -> np.ndarray:
    """Independent Strang path: RK4 reaction and eigenspace-exact Neumann diffusion."""
    state = initial_state(case)[0].copy()
    output_times = times(case)
    output = np.empty((len(output_times), *state.shape), dtype=np.float64)
    output[0] = state
    dx = 2.0 / (case.grid_shape[0] - 1)
    eigenvalues, basis = _neumann_eigendecomposition(case.grid_shape[0], dx)
    diffusivities = (float(case.parameters["du"]), float(case.parameters["dv"]))
    k = float(case.parameters["k"])
    now = float(output_times[0])
    for output_index, target in enumerate(output_times[1:], start=1):
        while now < float(target) - 1e-14:
            step = min(max_step, float(target) - now)
            state = rk4_step(state, 0.5 * step, lambda value: _reaction(value, k))
            state = _exact_neumann_diffusion(state, step, diffusivities, eigenvalues, basis)
            state = rk4_step(state, 0.5 * step, lambda value: _reaction(value, k))
            now += step
        output[output_index] = state
    return output[None, ...]


def make_case(
    *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
) -> PDECase:
    rng = np.random.default_rng(seed)
    values: dict[str, Any] = {
        "du": float(rng.choice(PAPER_DU)),
        "dv": float(rng.choice(PAPER_DV)),
        "k": float(rng.choice(PAPER_K)),
        "n": int(rng.choice(PAPER_GRID_SIZES)),
        "time_count": int(rng.choice(PAPER_TIME_COUNTS)),
        "final_time": PAPER_FINAL_TIME,
        "input_kind": str(rng.choice(PAPER_INPUT_KINDS)),
    }
    if stratum:
        unknown = set(stratum) - set(values)
        if unknown:
            raise ValueError(f"unknown reaction_diffusion2d stratum keys: {sorted(unknown)}")
        values.update(stratum)
    return make_case_record(
        family_id="reaction_diffusion2d",
        split=split,
        seed=seed,
        grid_shape=(int(values["n"]), int(values["n"])),
        time_count=int(values["time_count"]),
        final_time=float(values["final_time"]),
        input_kind=str(values["input_kind"]),
        parameters={
            "du": float(values["du"]),
            "dv": float(values["dv"]),
            "k": float(values["k"]),
        },
    )


class ReactionDiffusion2DEnvironment:
    family_id = "reaction_diffusion2d"
    seen_in_training = True
    method_ids = REACTION_DIFFUSION_2D_METHOD_IDS

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        return PDECase.from_dict(value)

    @staticmethod
    def make_case(
        *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
    ) -> PDECase:
        return make_case(seed=seed, split=split, stratum=stratum)

    @staticmethod
    def prompt(case: PDECase, *, form: str = "generic") -> str:
        conditioning = ""
        if form in {"parameter", "parameter_ic"}:
            conditioning += (
                f"\nFor this task, du={case.parameters['du']}, dv={case.parameters['dv']}, "
                f"and k={case.parameters['k']}."
            )
        if form == "parameter_ic":
            conditioning += f"\nThe initial-field family is {case.input_kind}."
        if form not in {"generic", "parameter", "parameter_ic"}:
            raise ValueError(f"unknown prompt form: {form}")
        return f"""Solve the two-dimensional two-species reaction--diffusion system

    u_t = Du * Laplacian(u) + u - u**3 - k - v
    v_t = Dv * Laplacian(v) + u - v

on (x, y) in [-1, 1]^2 with homogeneous Neumann boundary conditions.

Write this function:

    def solver(uv0_batch, t_coordinate, Du, Dv, k):
        ...

uv0_batch has shape [B, Nx, Ny, 2], with channels (u, v). t_coordinate has
shape [T]. Return a finite NumPy array with shape [B, T, Nx, Ny, 2]. Each
returned time slice must match the corresponding requested time. Internal
substeps are allowed. Do not hard-code a solution array.{conditioning}
"""

    @staticmethod
    def invocation(case: PDECase) -> SolverInvocation:
        return SolverInvocation(
            argument_order=("uv0_batch", "t_coordinate", "Du", "Dv", "k"),
            arrays={"uv0_batch": initial_state(case), "t_coordinate": times(case)},
            scalars={
                "Du": float(case.parameters["du"]),
                "Dv": float(case.parameters["dv"]),
                "k": float(case.parameters["k"]),
            },
            expected_shape=(
                SOURCE_DERIVED_BATCH_SIZE,
                len(times(case)),
                case.grid_shape[0],
                case.grid_shape[1],
                2,
            ),
        )

    @staticmethod
    def reference_primary(case: PDECase) -> np.ndarray:
        return _primary_reference(case, SOURCE_DERIVED_PRIMARY_MAX_STEP)

    @staticmethod
    def reference_audit(case: PDECase) -> np.ndarray:
        return _audit_reference(case, SOURCE_DERIVED_AUDIT_MAX_STEP)

    @staticmethod
    def residual_norm(field: np.ndarray, case: PDECase) -> float:
        values = np.asarray(field, dtype=np.float64)
        output_times = times(case)
        edge_order = 2 if len(output_times) >= 3 else 1
        derivative = np.gradient(values, output_times, axis=1, edge_order=edge_order)
        dx = 2.0 / (case.grid_shape[0] - 1)
        laplacian = neumann_second(values, dx, axis=2) + neumann_second(values, dx, axis=3)
        u = values[..., 0]
        v = values[..., 1]
        residual_u = derivative[..., 0] - (
            float(case.parameters["du"]) * laplacian[..., 0]
            + u
            - u**3
            - float(case.parameters["k"])
            - v
        )
        residual_v = derivative[..., 1] - (
            float(case.parameters["dv"]) * laplacian[..., 1] + u - v
        )
        return float(np.sqrt(np.mean(residual_u**2) + np.mean(residual_v**2)))

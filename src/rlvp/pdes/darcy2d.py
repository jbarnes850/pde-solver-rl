from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation
from rlvp.pdes.base import PDECase, make_case_record

PAPER_BETAS = (1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
PAPER_GRID_SIZES = (32, 48)
PAPER_FIELD_KINDS = ("random_blobs", "checkerboard", "channelized")

DARCY_METHOD_IDS = (
    "steady_state_cg_fv_arithmetic",
    "steady_state_jacobi_fv_arithmetic",
    "steady_state_weighted_jacobi_fv_arithmetic",
    "steady_state_redblack_gauss_seidel_fv_arithmetic",
    "steady_state_sor_fv_arithmetic",
    "steady_state_pcg_jacobi_fv_arithmetic",
    "geometric_multigrid_vcycle_fv",
    "pcg_ilu_fv",
)


def make_case(
    *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
) -> PDECase:
    """Build one paper-shaped Darcy case.

    The paper names the coefficient-field families but does not publish their generators. The
    deterministic generators below are therefore source-derived and not author-artifact exact.
    """
    rng = np.random.default_rng(seed)
    selected = dict(stratum or {})
    grid_size = int(selected.get("grid_size", rng.choice(PAPER_GRID_SIZES)))
    beta = float(selected.get("beta", rng.choice(PAPER_BETAS)))
    field_kind = str(selected.get("field_kind", rng.choice(PAPER_FIELD_KINDS)))
    if grid_size not in PAPER_GRID_SIZES:
        raise ValueError(f"grid_size must be one of {PAPER_GRID_SIZES}")
    if beta not in PAPER_BETAS:
        raise ValueError(f"beta must be one of {PAPER_BETAS}")
    if field_kind not in PAPER_FIELD_KINDS:
        raise ValueError(f"field_kind must be one of {PAPER_FIELD_KINDS}")
    return make_case_record(
        family_id="darcy2d",
        split=split,
        seed=seed,
        grid_shape=(grid_size, grid_size),
        time_count=None,
        final_time=None,
        input_kind=field_kind,
        parameters={"beta": beta, "field_kind": field_kind},
    )


def permeability_field(case: PDECase) -> np.ndarray:
    """Regenerate one positive cell-centered permeability field from the case seed."""
    nx, ny = case.grid_shape
    if nx != ny:
        raise ValueError("the source-derived Darcy environment requires a square grid")
    rng = np.random.default_rng(case.seed)
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    kind = str(case.parameters["field_kind"])

    if kind == "random_blobs":
        log_field = np.full((nx, ny), -0.15, dtype=np.float64)
        for _ in range(5):
            center_x, center_y = rng.uniform(0.08, 0.92, size=2)
            width = float(rng.uniform(0.055, 0.16))
            strength = float(rng.uniform(-0.9, 1.1))
            radius_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2
            log_field += strength * np.exp(-0.5 * radius_squared / width**2)
        field = np.exp(log_field)
    elif kind == "checkerboard":
        cells = int(rng.integers(3, 7))
        shift_x, shift_y = rng.integers(0, cells, size=2)
        index_x = (np.floor(xx * cells).astype(int) + int(shift_x)) % cells
        index_y = (np.floor(yy * cells).astype(int) + int(shift_y)) % cells
        low, high = rng.uniform((0.22, 1.35), (0.55, 2.1))
        field = np.where((index_x + index_y) % 2 == 0, low, high)
    elif kind == "channelized":
        center = float(rng.uniform(0.35, 0.65))
        amplitude = float(rng.uniform(0.08, 0.18))
        frequency = int(rng.integers(1, 3))
        phase = float(rng.uniform(-np.pi, np.pi))
        width = float(rng.uniform(0.045, 0.09))
        centerline = center + amplitude * np.sin(2.0 * np.pi * frequency * xx + phase)
        distance = np.abs(yy - centerline)
        field = 0.30 + 1.65 * np.exp(-0.5 * (distance / width) ** 2)
        field += 0.12 * (1.0 + np.sin(2.0 * np.pi * xx + 0.5 * phase))
    else:
        raise ValueError(f"unknown Darcy permeability family: {kind}")

    return np.clip(np.asarray(field, dtype=np.float64), 0.08, 3.0)


def _primary_coefficients(
    permeability: np.ndarray, reaction_rate: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build arithmetic-face FV coefficients by vectorized face construction."""
    nx, _ = permeability.shape
    scale = float(nx * nx)
    west = np.empty_like(permeability)
    east = np.empty_like(permeability)
    south = np.empty_like(permeability)
    north = np.empty_like(permeability)
    west[1:, :] = 0.5 * (permeability[1:, :] + permeability[:-1, :]) * scale
    west[0, :] = 2.0 * permeability[0, :] * scale
    east[:-1, :] = 0.5 * (permeability[:-1, :] + permeability[1:, :]) * scale
    east[-1, :] = 2.0 * permeability[-1, :] * scale
    south[:, 1:] = 0.5 * (permeability[:, 1:] + permeability[:, :-1]) * scale
    south[:, 0] = 2.0 * permeability[:, 0] * scale
    north[:, :-1] = 0.5 * (permeability[:, :-1] + permeability[:, 1:]) * scale
    north[:, -1] = 2.0 * permeability[:, -1] * scale
    diagonal = west + east + south + north + float(reaction_rate)
    return west, east, south, north, diagonal


def _primary_matvec(
    values: np.ndarray,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    west, east, south, north, diagonal = coefficients
    result = diagonal * values
    result[1:, :] -= west[1:, :] * values[:-1, :]
    result[:-1, :] -= east[:-1, :] * values[1:, :]
    result[:, 1:] -= south[:, 1:] * values[:, :-1]
    result[:, :-1] -= north[:, :-1] * values[:, 1:]
    return result


def _audit_coefficients(
    permeability: np.ndarray, reaction_rate: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independently assemble the same FV rows one cell and one face at a time."""
    nx, ny = permeability.shape
    inverse_h_squared = float(nx**2)
    left = np.zeros_like(permeability)
    right = np.zeros_like(permeability)
    down = np.zeros_like(permeability)
    up = np.zeros_like(permeability)
    diagonal = np.full_like(permeability, float(reaction_rate))
    for i in range(nx):
        for j in range(ny):
            center = float(permeability[i, j])
            if i == 0:
                diagonal[i, j] += 2.0 * center * inverse_h_squared
            else:
                face = 0.5 * (center + float(permeability[i - 1, j])) * inverse_h_squared
                left[i, j] = -face
                diagonal[i, j] += face
            if i == nx - 1:
                diagonal[i, j] += 2.0 * center * inverse_h_squared
            else:
                face = 0.5 * (center + float(permeability[i + 1, j])) * inverse_h_squared
                right[i, j] = -face
                diagonal[i, j] += face
            if j == 0:
                diagonal[i, j] += 2.0 * center * inverse_h_squared
            else:
                face = 0.5 * (center + float(permeability[i, j - 1])) * inverse_h_squared
                down[i, j] = -face
                diagonal[i, j] += face
            if j == ny - 1:
                diagonal[i, j] += 2.0 * center * inverse_h_squared
            else:
                face = 0.5 * (center + float(permeability[i, j + 1])) * inverse_h_squared
                up[i, j] = -face
                diagonal[i, j] += face
    return left, right, down, up, diagonal


def _audit_matvec(
    values: np.ndarray,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    left, right, down, up, diagonal = coefficients
    result = diagonal * values
    result[1:, :] += left[1:, :] * values[:-1, :]
    result[:-1, :] += right[:-1, :] * values[1:, :]
    result[:, 1:] += down[:, 1:] * values[:, :-1]
    result[:, :-1] += up[:, :-1] * values[:, 1:]
    return result


def _pcg_primary(
    matvec: Callable[[np.ndarray], np.ndarray],
    right_hand_side: np.ndarray,
    diagonal: np.ndarray,
    *,
    tolerance: float = 1.0e-11,
    maximum_iterations: int = 6000,
) -> np.ndarray:
    solution = np.zeros_like(right_hand_side)
    residual = right_hand_side - matvec(solution)
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    inner_product = float(np.vdot(residual, preconditioned))
    target = tolerance * max(float(np.linalg.norm(right_hand_side)), 1.0)
    for _ in range(maximum_iterations):
        image = matvec(direction)
        denominator = float(np.vdot(direction, image))
        if denominator <= 0.0:
            raise RuntimeError("Darcy primary PCG encountered a non-positive operator")
        step = inner_product / denominator
        solution += step * direction
        residual -= step * image
        if float(np.linalg.norm(residual)) <= target:
            return solution
        preconditioned = residual / diagonal
        updated_inner_product = float(np.vdot(residual, preconditioned))
        direction = preconditioned + (updated_inner_product / inner_product) * direction
        inner_product = updated_inner_product
    raise RuntimeError("Darcy primary PCG did not converge")


def _pcg_audit(
    matvec: Callable[[np.ndarray], np.ndarray],
    right_hand_side: np.ndarray,
    diagonal: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-11,
    iteration_limit: int = 6000,
) -> np.ndarray:
    estimate = np.zeros_like(right_hand_side)
    error = right_hand_side.copy()
    scaled_error = error / diagonal
    search = scaled_error.copy()
    old_energy = float(np.sum(error * scaled_error))
    threshold_squared = (relative_tolerance * max(np.linalg.norm(right_hand_side), 1.0)) ** 2
    for _iteration in range(iteration_limit):
        operator_search = matvec(search)
        alpha = old_energy / float(np.sum(search * operator_search))
        estimate = estimate + alpha * search
        error = error - alpha * operator_search
        if float(np.sum(error * error)) <= threshold_squared:
            return estimate
        scaled_error = error / diagonal
        new_energy = float(np.sum(error * scaled_error))
        search = scaled_error + (new_energy / old_energy) * search
        old_energy = new_energy
    raise RuntimeError("Darcy audit PCG did not converge")


def solve_primary(case: PDECase, *, reaction_rate: float = 0.0) -> np.ndarray:
    permeability = permeability_field(case)
    coefficients = _primary_coefficients(permeability, reaction_rate)
    right_hand_side = np.full(case.grid_shape, float(case.parameters["beta"]), dtype=np.float64)
    solution = _pcg_primary(
        lambda values: _primary_matvec(values, coefficients),
        right_hand_side,
        coefficients[-1],
    )
    return solution[None, :, :]


def solve_audit(case: PDECase, *, reaction_rate: float = 0.0) -> np.ndarray:
    permeability = permeability_field(case)
    coefficients = _audit_coefficients(permeability, reaction_rate)
    right_hand_side = np.full(case.grid_shape, float(case.parameters["beta"]), dtype=np.float64)
    solution = _pcg_audit(
        lambda values: _audit_matvec(values, coefficients),
        right_hand_side,
        coefficients[-1],
    )
    return solution[None, :, :]


def residual_norm(field: np.ndarray, case: PDECase, *, reaction_rate: float = 0.0) -> float:
    values = np.asarray(field, dtype=np.float64)
    expected_shape = (1, *case.grid_shape)
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        return float("inf")
    coefficients = _primary_coefficients(permeability_field(case), reaction_rate)
    right_hand_side = np.full(case.grid_shape, float(case.parameters["beta"]), dtype=np.float64)
    defect = _primary_matvec(values[0], coefficients) - right_hand_side
    denominator = max(float(np.sqrt(np.mean(right_hand_side**2))), 1.0e-12)
    return float(np.sqrt(np.mean(defect**2)) / denominator)


def _prompt(case: PDECase, *, form: str) -> str:
    if form not in {"generic", "parameter", "parameter_ic"}:
        raise ValueError(f"unsupported prompt form: {form}")
    details = ""
    if form in {"parameter", "parameter_ic"}:
        details += f"\nFor this task:\n- beta = {float(case.parameters['beta']):g}"
    if form == "parameter_ic":
        label = str(case.parameters["field_kind"]).replace("_", " ")
        details += f"\n- the permeability fields come from a {label} family"
    return f"""Write Python code only.

Define exactly:
    def solver(diffusion_batch, beta):

Solve the PDE
    -div(a(x,y) grad u) = beta
on (x,y) in [0,1]^2 with zero Dirichlet boundary conditions.{details}

Requirements:
- You may use NumPy and/or SciPy
- do not print anything
- do not access files or the network
- output only Python code
- Return a NumPy array of shape [B, Nx, Ny] containing the steady-state solution field.
- diffusion_batch has shape [B, Nx, Ny] and contains heterogeneous permeability fields a(x,y).
- beta is the scalar forcing term.
"""


class Darcy2DEnvironment:
    family_id = "darcy2d"
    seen_in_training = True
    method_ids = DARCY_METHOD_IDS

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        if case.family_id != "darcy2d":
            raise ValueError("case does not belong to darcy2d")
        return case

    @staticmethod
    def make_case(
        *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
    ) -> PDECase:
        return make_case(seed=seed, split=split, stratum=stratum)

    @staticmethod
    def prompt(case: PDECase, *, form: str = "generic") -> str:
        return _prompt(case, form=form)

    @staticmethod
    def invocation(case: PDECase) -> SolverInvocation:
        return SolverInvocation(
            argument_order=("diffusion_batch", "beta"),
            arrays={"diffusion_batch": permeability_field(case)[None, :, :]},
            scalars={"beta": float(case.parameters["beta"])},
            expected_shape=(1, *case.grid_shape),
        )

    @staticmethod
    def reference_primary(case: PDECase) -> np.ndarray:
        return solve_primary(case)

    @staticmethod
    def reference_audit(case: PDECase) -> np.ndarray:
        return solve_audit(case)

    @staticmethod
    def residual_norm(field: np.ndarray, case: PDECase) -> float:
        return residual_norm(field, case)

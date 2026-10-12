from __future__ import annotations

from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import periodic_grid, periodic_second, time_grid

PAPER_NUS = (1e-3, 1e-2, 1e-1, 1.0)
PAPER_GRID_SIZES = (64, 128)
PAPER_TIME_COUNTS = (51, 101)
PAPER_FINAL_TIME = 1.0
IC_KINDS = ("single_sine", "two_mode", "steeper_multimode")

BURGERS_METHOD_IDS = (
    "godunov_euler",
    "godunov_ssprk2",
    "godunov_ssprk3",
    "lax_friedrichs_euler",
    "lax_friedrichs_ssprk2",
    "lax_friedrichs_ssprk3",
    "rusanov_euler",
    "rusanov_ssprk2",
    "rusanov_ssprk3",
    "muscl_rusanov_euler",
    "muscl_rusanov_rk2",
    "muscl_rusanov_ssprk2",
    "muscl_rusanov_ssprk3",
    "lax_wendroff",
    "maccormack",
    "imex_cn_fixed",
    "spectral_rk4_dealias",
    "spectral_if_rk4_dealias",
)

_PROMPT_FORMS = frozenset({"generic", "parameter", "parameter_ic"})


def _nx(case: PDECase) -> int:
    if case.family_id != Burgers1DEnvironment.family_id:
        raise ValueError(f"expected a burgers1d case, got {case.family_id}")
    if len(case.grid_shape) != 1:
        raise ValueError("burgers1d requires a one-dimensional grid")
    return case.grid_shape[0]


def _nt(case: PDECase) -> int:
    if case.time_count is None:
        raise ValueError("burgers1d requires a time grid")
    return case.time_count


def _final_time(case: PDECase) -> float:
    if case.final_time is None:
        raise ValueError("burgers1d requires a final time")
    return case.final_time


def _nu(case: PDECase) -> float:
    return float(case.parameters["nu"])


def coordinates(case: PDECase) -> tuple[np.ndarray, np.ndarray]:
    """Return the local uniform periodic grid and uniform output-time grid.

    The paper gives grid counts, time counts, and final time. It does not give coordinate
    construction. Endpoint-free space and an inclusive uniform time grid are local choices.
    """

    return periodic_grid(_nx(case)), time_grid(_nt(case), _final_time(case))


def initial_condition(
    x: np.ndarray, kind: str, parameters: dict[str, int | float | str]
) -> np.ndarray:
    """Build one deterministic smooth periodic initial condition.

    The three family names come from the paper. The formulas, phase range, amplitude range,
    and mode weights are source-derived local choices because the paper omits them. Every
    formula has zero spatial mean, which also makes independent reference checks simpler.
    """

    phase = float(parameters["ic_phase"])
    amplitude = float(parameters["ic_amplitude"])
    mix = float(parameters["ic_mix"])
    if kind == "single_sine":
        return amplitude * np.sin(2.0 * np.pi * x + phase)
    if kind == "two_mode":
        return amplitude * (
            np.sin(2.0 * np.pi * x + phase) + mix * np.sin(4.0 * np.pi * x - 0.5 * phase)
        )
    if kind == "steeper_multimode":
        return amplitude * (
            np.sin(2.0 * np.pi * x + phase)
            + 0.35 * np.sin(6.0 * np.pi * x - phase)
            + 0.20 * np.sin(10.0 * np.pi * x + 0.25 * phase)
        )
    raise ValueError(f"unknown Burgers initial-condition kind: {kind}")


def initial_batch(case: PDECase) -> np.ndarray:
    x, _ = coordinates(case)
    return initial_condition(x, case.input_kind, case.parameters)[None, :]


def _make_parameters(seed: int) -> tuple[str, dict[str, int | float | str]]:
    rng = np.random.default_rng(seed)
    kind = str(rng.choice(IC_KINDS))
    parameters: dict[str, int | float | str] = {
        "nu": float(rng.choice(PAPER_NUS)),
        "ic_phase": float(rng.uniform(-np.pi, np.pi)),
        "ic_amplitude": float(rng.uniform(0.6, 1.0)),
        "ic_mix": float(rng.uniform(0.2, 0.45)),
    }
    return kind, parameters


def make_case(
    *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
) -> PDECase:
    """Create a deterministic paper-range case with explicit local IC parameters."""

    rng = np.random.default_rng(seed)
    input_kind, parameters = _make_parameters(seed)
    nx = int(rng.choice(PAPER_GRID_SIZES))
    nt = int(rng.choice(PAPER_TIME_COUNTS))
    final_time = PAPER_FINAL_TIME

    values = dict(stratum or {})
    input_kind = str(values.pop("ic_kind", values.pop("input_kind", input_kind)))
    nx = int(values.pop("nx", nx))
    nt = int(values.pop("nt", nt))
    final_time = float(values.pop("final_time", final_time))
    for name in ("nu", "ic_phase", "ic_amplitude", "ic_mix"):
        if name in values:
            parameters[name] = float(values.pop(name))
    if values:
        raise ValueError(f"unknown burgers1d stratum fields: {sorted(values)}")
    if input_kind not in IC_KINDS:
        raise ValueError(f"unsupported burgers1d initial-condition family: {input_kind}")
    if float(parameters["nu"]) not in PAPER_NUS:
        raise ValueError(f"nu must be one of the paper values: {PAPER_NUS}")
    if nx not in PAPER_GRID_SIZES:
        raise ValueError(f"nx must be one of the paper values: {PAPER_GRID_SIZES}")
    if nt not in PAPER_TIME_COUNTS:
        raise ValueError(f"nt must be one of the paper values: {PAPER_TIME_COUNTS}")
    if final_time != PAPER_FINAL_TIME:
        raise ValueError(f"final_time must equal the paper value {PAPER_FINAL_TIME}")

    return make_case_record(
        family_id=Burgers1DEnvironment.family_id,
        split=split,
        seed=seed,
        grid_shape=(nx,),
        time_count=nt,
        final_time=final_time,
        input_kind=input_kind,
        parameters=parameters,
    )


def prompt(case: PDECase, *, form: str = "generic") -> str:
    if form not in _PROMPT_FORMS:
        raise ValueError(f"unsupported prompt form: {form}")
    conditioning = ""
    if form in {"parameter", "parameter_ic"}:
        conditioning += f"\nFor this task, nu = {_nu(case):.12g}."
    if form == "parameter_ic":
        conditioning += f"\nThe initial-condition family is {case.input_kind}."
    return f"""Solve the periodic one-dimensional viscous Burgers equation

    u_t + d(u**2 / 2)/dx = (nu / pi) * u_xx,  x in [0, 1).

Write this function:

    def solver(u0_batch, t_coordinate, nu):
        ...

u0_batch has shape [B, N]. t_coordinate has shape [T]. Return a finite NumPy
array with shape [B, T, N]. Returned slice n must correspond to
t_coordinate[n]. Internal substeps are allowed. Use a stable numerical method.
Do not hard-code a solution array.{conditioning}
"""


def invocation(case: PDECase) -> SolverInvocation:
    _, times = coordinates(case)
    return SolverInvocation(
        argument_order=("u0_batch", "t_coordinate", "nu"),
        arrays={"u0_batch": initial_batch(case), "t_coordinate": times},
        scalars={"nu": _nu(case)},
        expected_shape=(1, _nt(case), _nx(case)),
    )


def _spectral_nonlinearity(
    spectrum: np.ndarray, wave_numbers: np.ndarray, dealias: np.ndarray
) -> np.ndarray:
    state = np.fft.ifft(spectrum).real
    result = -1j * wave_numbers * np.fft.fft(0.5 * state**2)
    result[~dealias] = 0.0
    return result


def _ifrk4_step(
    spectrum: np.ndarray,
    step: float,
    linear: np.ndarray,
    wave_numbers: np.ndarray,
    dealias: np.ndarray,
) -> np.ndarray:
    """One integrating-factor RK4 step with exact linear diffusion."""

    full = np.exp(linear * step)
    half = np.exp(linear * (0.5 * step))
    k1 = _spectral_nonlinearity(spectrum, wave_numbers, dealias)
    stage_a = half * (spectrum + 0.5 * step * k1)
    k2 = _spectral_nonlinearity(stage_a, wave_numbers, dealias)
    stage_b = half * spectrum + 0.5 * step * k2
    k3 = _spectral_nonlinearity(stage_b, wave_numbers, dealias)
    stage_c = full * spectrum + step * half * k3
    k4 = _spectral_nonlinearity(stage_c, wave_numbers, dealias)
    updated = full * spectrum + (step / 6.0) * (full * k1 + 2.0 * half * k2 + 2.0 * half * k3 + k4)
    updated[~dealias] = 0.0
    return updated


def _spectral_reference(case: PDECase, *, refinement: int) -> np.ndarray:
    """Refined dealiased Fourier reference with exact integrating-factor diffusion.

    The refinement and advective CFL 0.12 are local accuracy choices. The paper names this
    method family in its bank, but it does not release the hidden reference settings.
    """

    nx = _nx(case)
    refined_nx = refinement * nx
    x = periodic_grid(refined_nx)
    _, times = coordinates(case)
    state = initial_condition(x, case.input_kind, case.parameters)
    spectrum = np.fft.fft(state)
    modes = np.fft.fftfreq(refined_nx) * refined_nx
    wave_numbers = 2.0 * np.pi * modes
    dealias = np.abs(modes) <= refined_nx // 3
    linear = -(_nu(case) / np.pi) * wave_numbers**2
    output = np.empty((len(times), nx), dtype=np.float64)
    output[0] = initial_batch(case)[0]
    now = float(times[0])
    for output_index, target in enumerate(times[1:], start=1):
        target_value = float(target)
        while now < target_value - 1e-14:
            maximum_speed = max(float(np.max(np.abs(np.fft.ifft(spectrum).real))), 1e-8)
            step = min(0.12 / (refined_nx * maximum_speed), target_value - now)
            spectrum = _ifrk4_step(spectrum, step, linear, wave_numbers, dealias)
            now += step
        output[output_index] = np.fft.ifft(spectrum).real[::refinement]
    return output[None, :, :]


def _exact_diffusion(state: np.ndarray, step: float, diffusivity: float) -> np.ndarray:
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(len(state), d=1.0 / len(state))
    return np.fft.ifft(np.fft.fft(state) * np.exp(-diffusivity * wave_numbers**2 * step)).real


def _minmod(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    same_sign = left * right > 0.0
    return np.where(same_sign, np.sign(left) * np.minimum(np.abs(left), np.abs(right)), 0.0)


def _muscl_rusanov_rhs(state: np.ndarray, spacing: float) -> np.ndarray:
    slope = _minmod(state - np.roll(state, 1), np.roll(state, -1) - state)
    interface_left = state + 0.5 * slope
    interface_right = np.roll(state, -1) - 0.5 * np.roll(slope, -1)
    speed = np.maximum(np.abs(interface_left), np.abs(interface_right))
    interface_flux = 0.25 * (interface_left**2 + interface_right**2) - 0.5 * speed * (
        interface_right - interface_left
    )
    return -(interface_flux - np.roll(interface_flux, 1)) / spacing


def _ssprk3_advection(state: np.ndarray, step: float, spacing: float) -> np.ndarray:
    stage_one = state + step * _muscl_rusanov_rhs(state, spacing)
    stage_two = 0.75 * state + 0.25 * (stage_one + step * _muscl_rusanov_rhs(stage_one, spacing))
    return (1.0 / 3.0) * state + (2.0 / 3.0) * (
        stage_two + step * _muscl_rusanov_rhs(stage_two, spacing)
    )


def _finite_volume_audit(case: PDECase, *, refinement: int) -> np.ndarray:
    """Independent MUSCL-Rusanov audit with SSPRK3 and split exact diffusion.

    The method differs from the primary reference in its conservative spatial operator. The
    common refined grid is a local choice. The FFT is used only for the exact periodic heat
    substep.
    """

    nx = _nx(case)
    refined_nx = refinement * nx
    spacing = 1.0 / refined_nx
    x = periodic_grid(refined_nx)
    _, times = coordinates(case)
    state = initial_condition(x, case.input_kind, case.parameters)
    diffusivity = _nu(case) / np.pi
    output = np.empty((len(times), nx), dtype=np.float64)
    output[0] = initial_batch(case)[0]
    now = float(times[0])
    for output_index, target in enumerate(times[1:], start=1):
        target_value = float(target)
        while now < target_value - 1e-14:
            maximum_speed = max(float(np.max(np.abs(state))), 1e-8)
            step = min(0.12 * spacing / maximum_speed, target_value - now)
            state = _exact_diffusion(state, 0.5 * step, diffusivity)
            state = _ssprk3_advection(state, step, spacing)
            state = _exact_diffusion(state, 0.5 * step, diffusivity)
            now += step
        output[output_index] = state[::refinement]
    return output[None, :, :]


def _reference_refinement(case: PDECase) -> int:
    """Use more resolution as the paper viscosity approaches the shock-like regime."""

    nu = _nu(case)
    if nu <= 1e-3:
        return 16
    if nu <= 1e-2:
        return 8
    return 4


def reference_primary(case: PDECase) -> np.ndarray:
    return _spectral_reference(case, refinement=_reference_refinement(case))


def reference_audit(case: PDECase) -> np.ndarray:
    return _finite_volume_audit(case, refinement=_reference_refinement(case))


def residual_norm(field: np.ndarray, case: PDECase) -> float:
    """Return a source-derived discrete RMS residual on the public output grid.

    The paper gives only the continuous residual. This local diagnostic uses a second-order
    time derivative, centered periodic flux derivative, and centered periodic Laplacian.
    """

    values = np.asarray(field, dtype=np.float64)
    expected = (1, _nt(case), _nx(case))
    if values.shape != expected:
        raise ValueError(f"expected field shape {expected}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("field must contain only finite values")
    _, times = coordinates(case)
    spacing = 1.0 / _nx(case)
    du_dt = np.gradient(values, times, axis=1, edge_order=2)
    flux = 0.5 * values**2
    flux_dx = (np.roll(flux, -1, axis=2) - np.roll(flux, 1, axis=2)) / (2.0 * spacing)
    diffusion = (_nu(case) / np.pi) * periodic_second(values, spacing, axis=2)
    residual = du_dt + flux_dx - diffusion
    return float(np.sqrt(np.mean(residual**2)))


class Burgers1DEnvironment:
    family_id = "burgers1d"
    seen_in_training = True
    method_ids = BURGERS_METHOD_IDS

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        _nx(case)
        return case

    make_case = staticmethod(make_case)
    prompt = staticmethod(prompt)
    invocation = staticmethod(invocation)
    reference_primary = staticmethod(reference_primary)
    reference_audit = staticmethod(reference_audit)
    residual_norm = staticmethod(residual_norm)

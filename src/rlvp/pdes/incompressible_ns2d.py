"""Source-derived 2D incompressible Navier--Stokes environment.

Paper-fixed facts
-----------------
The domain is ``[0, 1]^2``.  Velocity satisfies no-slip boundary conditions and

``div(v) = 0``
``v_t + (v . grad)v = -grad(p) + nu * laplacian(v) + f``.

The paper fixes the viscosity, velocity scale, forcing scale, grid-size, output-time,
and final-time sets below.  It also states that the initial streamfunction and the two
independent forcing components are Gaussian random fields with spectral exponents -3
and -1, respectively.

Source-derived local choices
----------------------------
The paper does not release the original random-field generator, solver call signature,
pressure gauge, reference implementation, internal time-step policy, or residual
discretization.  This module therefore makes those choices explicit:

* a real white-noise field is Fourier filtered with amplitude proportional to
  ``(1 + |k|^2)^(tau/2)`` and normalized by its RMS;
* the streamfunction is multiplied by a squared sine wall envelope, differentiated to
  velocity, and the velocity boundary is set to zero;
* the public solver signature is
  ``solver(velocity0_batch, forcing_batch, t_coordinate, nu)``;
* the primary reference is a repeated Chorin projection with skew-symmetric advection
  and RK2;
* the independent audit uses centered advection, Euler stepping, half the CFL target,
  and a tighter pressure iteration;
* pressure has homogeneous-Neumann wall treatment and a zero-mean gauge;
* the pressure-free residual combines the vorticity-equation RMS and divergence RMS,
  using centered finite differences on interior points.

These choices implement a transparent environment.  They are not claimed to reproduce
the unreleased author environment exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

PAPER_VISCOSITIES = (0.005, 0.01, 0.02)
PAPER_VELOCITY_SCALES = (0.1, 0.15, 0.2)
PAPER_FORCING_SCALES = (0.25, 0.4, 0.55)
PAPER_GRID_SIZES = (16, 24)
PAPER_TIME_COUNTS = (11, 21)
PAPER_FINAL_TIME = 1.0
PAPER_STREAMFUNCTION_TAU = -3.0
PAPER_FORCING_TAU = -1.0

INCOMPRESSIBLE_NS2D_METHOD_IDS = (
    "chorin_centered_euler",
    "chorin_centered_rk2",
    "chorin_skewsym_euler",
    "semiimplicit_centered_euler",
    "imex_cnab2_skewsym",
    "dst_projection_cnab2_runtime",
    "imex_cnab2_skewsym_spectral",
    "semiimplicit_centered_spectral",
    "mac_projection_centered",
    "mac_projection_rk2",
)


@dataclass(frozen=True)
class IncompressibleNS2DCase:
    case_id: str
    nu: float
    velocity_scale: float
    forcing_scale: float
    n: int
    nt: int
    final_time: float
    split: str
    seed: int
    streamfunction_tau: float = PAPER_STREAMFUNCTION_TAU
    forcing_tau: float = PAPER_FORCING_TAU
    generator_version: str = "source-derived-fourier-wall-v1"

    @property
    def family_id(self) -> str:
        return "incompressible_ns2d"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IncompressibleNS2DCase:
        return cls(
            case_id=str(value["case_id"]),
            nu=float(value["nu"]),
            velocity_scale=float(value["velocity_scale"]),
            forcing_scale=float(value["forcing_scale"]),
            n=int(value["n"]),
            nt=int(value["nt"]),
            final_time=float(value["final_time"]),
            split=str(value["split"]),
            seed=int(value["seed"]),
            streamfunction_tau=float(value.get("streamfunction_tau", PAPER_STREAMFUNCTION_TAU)),
            forcing_tau=float(value.get("forcing_tau", PAPER_FORCING_TAU)),
            generator_version=str(
                value.get("generator_version", "source-derived-fourier-wall-v1")
            ),
        )

    @property
    def x(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.n, dtype=np.float64)

    @property
    def t(self) -> np.ndarray:
        return np.linspace(0.0, self.final_time, self.nt, dtype=np.float64)

    @property
    def velocity0(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        streamfunction = _gaussian_random_field(rng, self.n, self.streamfunction_tau)
        coordinates = self.x
        wall = np.sin(np.pi * coordinates) ** 2
        streamfunction *= wall[:, None] * wall[None, :]
        spacing = 1.0 / (self.n - 1)
        velocity = np.stack(
            (
                _first_derivative(streamfunction, spacing, axis=1),
                -_first_derivative(streamfunction, spacing, axis=0),
            ),
            axis=-1,
        )
        _set_velocity_wall(velocity)
        rms = float(np.sqrt(np.mean(np.square(velocity))))
        if rms == 0.0:
            raise RuntimeError("streamfunction generator produced zero velocity")
        velocity *= self.velocity_scale / rms
        _set_velocity_wall(velocity)
        return velocity

    @property
    def forcing(self) -> np.ndarray:
        # Consume the streamfunction draw first so both public fields come from one stable
        # seed stream and remain unchanged when accessed independently.
        rng = np.random.default_rng(self.seed)
        _gaussian_random_field(rng, self.n, self.streamfunction_tau)
        forcing = np.stack(
            (
                _gaussian_random_field(rng, self.n, self.forcing_tau),
                _gaussian_random_field(rng, self.n, self.forcing_tau),
            ),
            axis=-1,
        )
        rms = float(np.sqrt(np.mean(np.square(forcing))))
        if rms == 0.0:
            raise RuntimeError("forcing generator produced a zero field")
        return forcing * (self.forcing_scale / rms)

    @property
    def velocity0_batch(self) -> np.ndarray:
        return self.velocity0[None, ...]

    @property
    def forcing_batch(self) -> np.ndarray:
        return self.forcing[None, ...]


def _canonical_case_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "ns2d-" + hashlib.sha256(canonical.encode()).hexdigest()[:20]


def make_case(
    *,
    seed: int,
    split: str = "train",
    stratum: dict[str, Any] | None = None,
) -> IncompressibleNS2DCase:
    rng = np.random.default_rng(seed)
    payload: dict[str, Any] = {
        "nu": float(rng.choice(PAPER_VISCOSITIES)),
        "velocity_scale": float(rng.choice(PAPER_VELOCITY_SCALES)),
        "forcing_scale": float(rng.choice(PAPER_FORCING_SCALES)),
        "n": int(rng.choice(PAPER_GRID_SIZES)),
        "nt": int(rng.choice(PAPER_TIME_COUNTS)),
        "final_time": PAPER_FINAL_TIME,
        "split": split,
        "seed": seed,
        "streamfunction_tau": PAPER_STREAMFUNCTION_TAU,
        "forcing_tau": PAPER_FORCING_TAU,
        "generator_version": "source-derived-fourier-wall-v1",
    }
    if stratum:
        unknown = set(stratum) - {
            "nu",
            "velocity_scale",
            "forcing_scale",
            "n",
            "nt",
            "final_time",
        }
        if unknown:
            raise ValueError(f"unknown incompressible_ns2d stratum keys: {sorted(unknown)}")
        payload.update(stratum)
    payload["case_id"] = _canonical_case_id(payload)
    return IncompressibleNS2DCase.from_dict(payload)


def _gaussian_random_field(rng: np.random.Generator, count: int, tau: float) -> np.ndarray:
    """Generate the declared local Fourier-filtered GRF choice."""
    white = rng.standard_normal((count, count))
    frequencies = np.fft.fftfreq(count, d=1.0 / count)
    wave_x, wave_y = np.meshgrid(frequencies, frequencies, indexing="ij")
    radial_squared = wave_x**2 + wave_y**2
    transfer = np.power(1.0 + radial_squared, 0.5 * tau)
    transfer[0, 0] = 0.0
    field = np.fft.ifftn(np.fft.fftn(white) * transfer).real
    field -= np.mean(field)
    rms = float(np.sqrt(np.mean(np.square(field))))
    if rms == 0.0:
        raise RuntimeError("Gaussian random-field draw has zero variance")
    return field / rms


def _set_velocity_wall(velocity: np.ndarray) -> None:
    velocity[0, :, :] = 0.0
    velocity[-1, :, :] = 0.0
    velocity[:, 0, :] = 0.0
    velocity[:, -1, :] = 0.0


def _first_derivative(values: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    return np.gradient(values, spacing, axis=axis, edge_order=2)


def _laplacian(values: np.ndarray, spacing: float) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:-1, 1:-1] = (
        values[2:, 1:-1]
        + values[:-2, 1:-1]
        + values[1:-1, 2:]
        + values[1:-1, :-2]
        - 4.0 * values[1:-1, 1:-1]
    ) / spacing**2
    return result


def _centered_advection(velocity: np.ndarray, spacing: float) -> np.ndarray:
    u = velocity[..., 0]
    v = velocity[..., 1]
    output = np.zeros_like(velocity)
    for component in range(2):
        quantity = velocity[..., component]
        output[..., component] = u * _first_derivative(
            quantity, spacing, axis=0
        ) + v * _first_derivative(quantity, spacing, axis=1)
    return output


def _skew_advection(velocity: np.ndarray, spacing: float) -> np.ndarray:
    u = velocity[..., 0]
    v = velocity[..., 1]
    advective = _centered_advection(velocity, spacing)
    conservative = np.empty_like(velocity)
    for component in range(2):
        quantity = velocity[..., component]
        conservative[..., component] = _first_derivative(
            u * quantity, spacing, axis=0
        ) + _first_derivative(v * quantity, spacing, axis=1)
    return 0.5 * (advective + conservative)


def _divergence(velocity: np.ndarray, spacing: float) -> np.ndarray:
    return _first_derivative(velocity[..., 0], spacing, axis=0) + _first_derivative(
        velocity[..., 1], spacing, axis=1
    )


def _project(
    tentative: np.ndarray,
    time_step: float,
    spacing: float,
    *,
    pressure_iterations: int,
    projection_passes: int,
) -> np.ndarray:
    """Apply collocated-grid Chorin projection sweeps.

    Repeated sweeps reduce the central-difference divergence that remains because a
    collocated pressure Laplacian is not exactly the composition of the selected
    centered gradient and divergence stencils near walls.
    """
    projected = tentative.copy()
    for _ in range(projection_passes):
        rhs = _divergence(projected, spacing) / time_step
        pressure = np.zeros(tentative.shape[:2], dtype=np.float64)
        for _ in range(pressure_iterations):
            previous = pressure
            pressure = previous.copy()
            pressure[1:-1, 1:-1] = 0.25 * (
                previous[2:, 1:-1]
                + previous[:-2, 1:-1]
                + previous[1:-1, 2:]
                + previous[1:-1, :-2]
                - spacing**2 * rhs[1:-1, 1:-1]
            )
            pressure[0, :] = pressure[1, :]
            pressure[-1, :] = pressure[-2, :]
            pressure[:, 0] = pressure[:, 1]
            pressure[:, -1] = pressure[:, -2]
            pressure -= np.mean(pressure)

        projected[..., 0] -= time_step * _first_derivative(pressure, spacing, axis=0)
        projected[..., 1] -= time_step * _first_derivative(pressure, spacing, axis=1)
        _set_velocity_wall(projected)
    return projected


def _velocity_rhs(
    velocity: np.ndarray,
    forcing: np.ndarray,
    nu: float,
    spacing: float,
    *,
    skew_symmetric: bool,
) -> np.ndarray:
    advection = (
        _skew_advection(velocity, spacing)
        if skew_symmetric
        else _centered_advection(velocity, spacing)
    )
    output = -advection + forcing
    for component in range(2):
        output[..., component] += nu * _laplacian(velocity[..., component], spacing)
    output[0, :, :] = 0.0
    output[-1, :, :] = 0.0
    output[:, 0, :] = 0.0
    output[:, -1, :] = 0.0
    return output


def _stable_step(velocity: np.ndarray, spacing: float, nu: float, factor: float) -> float:
    max_speed = float(np.max(np.sqrt(np.sum(np.square(velocity), axis=-1))))
    advective = np.inf if max_speed < 1e-12 else factor * spacing / max_speed
    diffusive = factor * spacing**2 / max(nu, 1e-12)
    return float(min(advective, diffusive, 0.02))


def _integrate_projection(
    case: IncompressibleNS2DCase,
    *,
    rk2: bool,
    skew_symmetric: bool,
    step_factor: float,
    pressure_iterations: int,
    projection_passes: int,
) -> np.ndarray:
    velocity = case.velocity0.copy()
    forcing = case.forcing
    spacing = 1.0 / (case.n - 1)
    output = np.empty((1, case.nt, case.n, case.n, 2), dtype=np.float64)
    output[0, 0] = velocity
    now = 0.0
    for output_index, target in enumerate(case.t[1:], start=1):
        while now < float(target) - 1e-14:
            time_step = min(
                _stable_step(velocity, spacing, case.nu, step_factor),
                float(target) - now,
            )
            first_rhs = _velocity_rhs(
                velocity,
                forcing,
                case.nu,
                spacing,
                skew_symmetric=skew_symmetric,
            )
            first = _project(
                velocity + time_step * first_rhs,
                time_step,
                spacing,
                pressure_iterations=pressure_iterations,
                projection_passes=projection_passes,
            )
            if rk2:
                second_rhs = _velocity_rhs(
                    first,
                    forcing,
                    case.nu,
                    spacing,
                    skew_symmetric=skew_symmetric,
                )
                velocity = _project(
                    velocity + 0.5 * time_step * (first_rhs + second_rhs),
                    time_step,
                    spacing,
                    pressure_iterations=pressure_iterations,
                    projection_passes=projection_passes,
                )
            else:
                velocity = first
            now += time_step
        output[0, output_index] = velocity
    return output


def _vorticity(velocity: np.ndarray, spacing: float) -> np.ndarray:
    return _first_derivative(velocity[..., 1], spacing, axis=-2) - _first_derivative(
        velocity[..., 0], spacing, axis=-1
    )


def pressure_free_residual_norm(field: np.ndarray, case: IncompressibleNS2DCase) -> float:
    """Return combined vorticity-momentum and incompressibility RMS.

    Taking the curl removes pressure from the momentum equation.  The second term keeps
    the independent divergence constraint visible.  Only interior spatial points and
    centered interior output times are used, to avoid one-sided boundary/time stencils.
    """
    values = np.asarray(field, dtype=np.float64)
    expected = (1, case.nt, case.n, case.n, 2)
    if values.shape != expected:
        raise ValueError(f"expected field shape {expected}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        return float("inf")

    spacing = 1.0 / (case.n - 1)
    velocity = values[0]
    omega = _vorticity(velocity, spacing)
    omega_t = (omega[2:] - omega[:-2]) / (case.t[2:, None, None] - case.t[:-2, None, None])
    middle_velocity = velocity[1:-1]
    omega_middle = omega[1:-1]
    omega_x = _first_derivative(omega_middle, spacing, axis=1)
    omega_y = _first_derivative(omega_middle, spacing, axis=2)
    omega_laplacian = np.empty_like(omega_middle)
    for time_index in range(omega_middle.shape[0]):
        omega_laplacian[time_index] = _laplacian(omega_middle[time_index], spacing)
    forcing_curl = _vorticity(case.forcing, spacing)
    momentum = (
        omega_t
        + middle_velocity[..., 0] * omega_x
        + middle_velocity[..., 1] * omega_y
        - case.nu * omega_laplacian
        - forcing_curl[None, ...]
    )
    divergence = _first_derivative(middle_velocity[..., 0], spacing, axis=1) + (
        _first_derivative(middle_velocity[..., 1], spacing, axis=2)
    )
    interior = (slice(None), slice(2, -2), slice(2, -2))
    momentum_rms = float(np.sqrt(np.mean(np.square(momentum[interior]))))
    divergence_rms = float(np.sqrt(np.mean(np.square(divergence[interior]))))
    return float(np.hypot(momentum_rms, divergence_rms))


class IncompressibleNS2DEnvironment:
    family_id = "incompressible_ns2d"
    seen_in_training = True
    method_ids = INCOMPRESSIBLE_NS2D_METHOD_IDS

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> IncompressibleNS2DCase:
        return IncompressibleNS2DCase.from_dict(value)

    @staticmethod
    def make_case(
        *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
    ) -> IncompressibleNS2DCase:
        return make_case(seed=seed, split=split, stratum=stratum)

    @staticmethod
    def prompt(case: IncompressibleNS2DCase, *, form: str = "generic") -> str:
        if form not in {"generic", "parameter", "parameter_ic"}:
            raise ValueError(f"unknown prompt form: {form}")
        conditioning = ""
        if form in {"parameter", "parameter_ic"}:
            conditioning += f"\nFor this task:\n- nu = {case.nu:g}\n"
        if form == "parameter_ic":
            conditioning += (
                "- the initial velocity comes from a random streamfunction with tau_v = -3\n"
                "- forcing components are independent random fields with tau_f = -1\n"
            )
        return f"""Write Python code only.

Define exactly:
    def solver(velocity0_batch, forcing_batch, t_coordinate, nu):

Solve the PDE on (x,y) in [0,1]^2:
    div(v) = 0
    v_t + (v . grad)v = -grad(p) + nu * laplacian(v) + f
with no-slip velocity v = 0 on the boundary.{conditioning}
Requirements:
- You may use NumPy and/or SciPy
- do not print anything
- do not access files or the network
- output only Python code
- Return a NumPy array of shape [B, T, Nx, Ny, 2].
- velocity0_batch has shape [B, Nx, Ny, 2] and stores (u,v).
- forcing_batch has shape [B, Nx, Ny, 2] and is time independent.
- t_coordinate has shape [T] and gives the exact output times.
- nu is the scalar kinematic viscosity.
- returned slice n must correspond to time t_coordinate[n]
- internal substeps are allowed if needed for stability or accuracy
"""

    @staticmethod
    def invocation(case: IncompressibleNS2DCase) -> SolverInvocation:
        return SolverInvocation(
            argument_order=("velocity0_batch", "forcing_batch", "t_coordinate", "nu"),
            arrays={
                "velocity0_batch": case.velocity0_batch,
                "forcing_batch": case.forcing_batch,
                "t_coordinate": case.t,
            },
            scalars={"nu": case.nu},
            expected_shape=(1, case.nt, case.n, case.n, 2),
        )

    @staticmethod
    def reference_primary(case: IncompressibleNS2DCase) -> np.ndarray:
        return _integrate_projection(
            case,
            rk2=True,
            skew_symmetric=True,
            step_factor=0.16,
            pressure_iterations=90,
            projection_passes=3,
        )

    @staticmethod
    def reference_audit(case: IncompressibleNS2DCase) -> np.ndarray:
        return _integrate_projection(
            case,
            rk2=False,
            skew_symmetric=False,
            step_factor=0.08,
            pressure_iterations=150,
            projection_passes=4,
        )

    @staticmethod
    def residual_norm(field: np.ndarray, case: IncompressibleNS2DCase) -> float:
        return pressure_free_residual_norm(field, case)

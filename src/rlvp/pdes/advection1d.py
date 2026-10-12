from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

PAPER_BETAS = (0.1, 0.4, 1.0, 2.0)
PAPER_GRID_SIZES = (64, 128)
PAPER_TIME_COUNTS = (51, 101)
PAPER_FINAL_TIME = 2.0
IC_KINDS = ("single_sine", "two_mode", "multimode", "windowed_absolute")


@dataclass(frozen=True)
class AdvectionCase:
    case_id: str
    beta: float
    nx: int
    nt: int
    final_time: float
    ic_kind: str
    ic_parameters: dict[str, float]
    split: str
    seed: int

    @property
    def family_id(self) -> str:
        return "advection1d"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AdvectionCase:
        return cls(
            case_id=str(value["case_id"]),
            beta=float(value["beta"]),
            nx=int(value["nx"]),
            nt=int(value["nt"]),
            final_time=float(value["final_time"]),
            ic_kind=str(value["ic_kind"]),
            ic_parameters={str(k): float(v) for k, v in value["ic_parameters"].items()},
            split=str(value["split"]),
            seed=int(value["seed"]),
        )

    @property
    def x(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.nx, endpoint=False, dtype=np.float64)

    @property
    def t(self) -> np.ndarray:
        return np.linspace(0.0, self.final_time, self.nt, dtype=np.float64)

    @property
    def u0(self) -> np.ndarray:
        return initial_condition(self.x, self.ic_kind, self.ic_parameters)

    @property
    def u0_batch(self) -> np.ndarray:
        """Paper-shaped batch input for the first vertical slice (B=1)."""
        return self.u0[None, :]

    def reference(self) -> np.ndarray:
        shifted = (self.x[None, :] - self.beta * self.t[:, None]) % 1.0
        return initial_condition(shifted, self.ic_kind, self.ic_parameters)[None, :, :]


def initial_condition(x: np.ndarray, kind: str, parameters: dict[str, float]) -> np.ndarray:
    phase = parameters.get("phase", 0.0)
    amplitude = parameters.get("amplitude", 1.0)
    offset = parameters.get("offset", 0.0)
    if kind == "single_sine":
        return offset + amplitude * np.sin(2.0 * np.pi * x + phase)
    if kind == "two_mode":
        return offset + amplitude * (
            np.sin(2.0 * np.pi * x + phase)
            + parameters.get("mix", 0.35) * np.sin(4.0 * np.pi * x - 0.5 * phase)
        )
    if kind == "multimode":
        return offset + amplitude * (
            np.sin(2.0 * np.pi * x + phase)
            + 0.30 * np.sin(6.0 * np.pi * x - phase)
            + 0.15 * np.cos(10.0 * np.pi * x + 0.25 * phase)
        )
    if kind == "windowed_absolute":
        center = parameters.get("center", 0.5)
        width = parameters.get("width", 0.2)
        periodic_distance = np.minimum(np.abs(x - center), 1.0 - np.abs(x - center))
        window = np.clip(1.0 - periodic_distance / width, 0.0, 1.0)
        return offset + amplitude * np.abs(np.sin(2.0 * np.pi * x + phase)) * window
    raise ValueError(f"unknown initial-condition kind: {kind}")


def make_case(*, seed: int, split: str = "train") -> AdvectionCase:
    rng = np.random.default_rng(seed)
    payload: dict[str, Any] = {
        "beta": float(rng.choice(PAPER_BETAS)),
        "nx": int(rng.choice(PAPER_GRID_SIZES)),
        "nt": int(rng.choice(PAPER_TIME_COUNTS)),
        "final_time": PAPER_FINAL_TIME,
        "ic_kind": str(rng.choice(IC_KINDS)),
        "ic_parameters": {
            "phase": float(rng.uniform(-np.pi, np.pi)),
            "amplitude": float(rng.uniform(0.5, 1.5)),
            "offset": float(rng.uniform(-0.25, 0.25)),
            "mix": float(rng.uniform(0.2, 0.6)),
            "center": float(rng.uniform(0.2, 0.8)),
            "width": float(rng.uniform(0.12, 0.3)),
        },
        "split": split,
        "seed": seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    case_id = "adv1d-" + hashlib.sha256(canonical.encode()).hexdigest()[:20]
    return AdvectionCase(case_id=case_id, **payload)


ADVECTION_METHOD_IDS = (
    "semi_lagrangian",
    "pdebench_fd2_midpoint_vanleer",
    "pdebench_rusanov2",
    "central4_rk4",
    "muscl_rusanov_ssprk3",
    "spectral_rk4",
    "fourier_exact_shift",
    "fourier_spectral_rk4",
)


class Advection1DEnvironment:
    family_id = "advection1d"
    seen_in_training = True
    method_ids = ADVECTION_METHOD_IDS

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> AdvectionCase:
        return AdvectionCase.from_dict(value)

    @staticmethod
    def make_case(
        *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
    ) -> AdvectionCase:
        case = make_case(seed=seed, split=split)
        if not stratum:
            return case
        payload = case.to_dict()
        for name in ("beta", "nx", "nt", "final_time", "ic_kind"):
            if name in stratum:
                payload[name] = stratum[name]
        canonical = json.dumps(
            {key: value for key, value in payload.items() if key != "case_id"},
            sort_keys=True,
            separators=(",", ":"),
        )
        payload["case_id"] = "adv1d-" + hashlib.sha256(canonical.encode()).hexdigest()[:20]
        return AdvectionCase.from_dict(payload)

    @staticmethod
    def prompt(case: AdvectionCase, *, form: str = "generic") -> str:
        from rlvp.prompts import advection_prompt

        return advection_prompt(case, form=form)

    @staticmethod
    def invocation(case: AdvectionCase) -> SolverInvocation:
        return SolverInvocation(
            argument_order=("u0_batch", "t_coordinate", "beta"),
            arrays={"u0_batch": case.u0_batch, "t_coordinate": case.t},
            scalars={"beta": case.beta},
            expected_shape=(1, case.nt, case.nx),
        )

    @staticmethod
    def reference_primary(case: AdvectionCase) -> np.ndarray:
        return case.reference()

    @staticmethod
    def reference_audit(case: AdvectionCase) -> np.ndarray:
        modes = np.fft.fft(case.u0_batch, axis=-1)
        wave = 2.0 * np.pi * np.fft.fftfreq(case.nx, d=1.0 / case.nx)
        phase = np.exp(-1j * case.beta * case.t[:, None] * wave[None, :])
        return np.fft.ifft(modes[:, None, :] * phase[None, :, :], axis=-1).real

    @staticmethod
    def residual_norm(field: np.ndarray, case: AdvectionCase) -> float:
        from rlvp.rewards import advection_residual_norm

        return advection_residual_norm(field, case)

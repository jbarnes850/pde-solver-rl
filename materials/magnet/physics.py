"""Pseudo-binary spinodal magnet model (research example inspired by Fe-Cr-Co).

This is a labeled research example, not a calibrated description of a real alloy.
Every constant below is a prior choice with a cited order of magnitude; see
materials/README.md for sources and ranges.

Governing equation (2D periodic, field along x):

    dc/dt = div( M(T) grad mu ),
    mu = f'(c; T) - 2 kappa lap(c) + mu_mag,
    f(c; T) = (R/Vm) [ T (c ln c + (1-c) ln(1-c)) + 2 T_top c (1-c) ],
    mu_mag  = B(T) * F^-1[ (k_x^2/|k|^2) F[c] ]   (only while the field is on).

Nondimensionalization: energy E0 = R T_top / Vm, length ell = sqrt(2 kappa / E0),
time tau0 = ell^2 T_ref / (D_ref c0 (1-c0) T_top). Then

    dc/dt~ = m(T) lap~( theta ln(c/(1-c)) + 2 (1 - 2c) - lap~ c + b K_x c ),
    m(T) = (D(T)/D_ref) (T_ref / T),  theta = T / T_top,  b = B(T)/E0.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

R_GAS = 8.314462618
MU0 = 4.0e-7 * math.pi


@dataclass(frozen=True)
class MaterialParams:
    """Uncertain material description. Temperatures in K, SI units otherwise."""

    T_s: float = 933.0  # coherent spinodal temperature at the alloy composition c0 (~660 C)
    c0: float = 0.40  # pseudo-binary composition (Cr-rich fraction); alpha1 is the low-c phase
    D_ref: float = 3.0e-18  # effective interdiffusion coefficient at T_ref (m^2/s)
    Q: float = 250.0e3  # activation energy (J/mol)
    T_ref: float = 913.0  # 640 C, reference for D_ref
    kappa: float = 1.14e-10  # gradient energy coefficient (J/m); 8.125e-16 J m^2/mol / Vm
    Vm: float = 7.1e-6  # molar volume (m^3/mol)
    B0: float = 1.5e6  # magnetostatic coupling at T_ref (J/m^3), (dJs/dc)^2 m_s(T)^2 / mu0
    T_curie: float = 1100.0  # Curie temperature used for the B(T) decline (alpha1-like)
    noise_rms: float = 0.005  # rms quenched-in composition fluctuation after solution treatment
    # Cubic elastic anisotropy (model-form option; 0 disables). Energy (1/2) B_el G(k) |c_k|^2
    # with G = 4 n1^2 n2^2 in crystal axes: 0 along <100>, 1 along <110>.
    B_el: float = 0.0  # J/m^3; estimate (Y<110> - Y<100>) eta^2 ~ 2e6-5e6 for Fe-Cr
    grain_angle_deg: float = 0.0  # angle between the field (x) and the crystal [100] axis
    # Magnetostatic model form. "linear": constant coupling B0 (Cahn 1963 linearization).
    # "curie": Ms(c,T) = Ms_a (1-c) sqrt(1 - T/Tc(c)) (regularized), Tc(c) linear in c; the
    # coupling mu0 Ms'(c)^2 then grows near the local Curie temperature.
    # Thermal (Cahn-Hilliard-Cook) noise: eps = theta * Omega_atom / (ell^2 d_z) by
    # fluctuation-dissipation, d_z the thickness a 2D slice represents (coarse-graining choice).
    thermal_noise: bool = True
    d_z_nm: float = 20.0
    mag_model: str = "linear"
    Tc_a: float = 1250.0  # Curie temperature of the c -> 0 (Fe,Co-rich) end member (K)
    Tc_b: float = 530.0  # Curie temperature of the c -> 1 end; gives Tc(c0=0.45) ~ 926 K
    curie_width: float = 0.01  # regularization of the sqrt at Tc (in 1 - T/Tc units)
    # Optional hypothesis: fixed aligned magnetization in an external induction.
    # Zero preserves M2. Nonzero is supported only for mag_model="curie" and adds
    # -B_ext*Js_a*g/mu0 to the free-energy density while the schedule field is on.
    # This is a specified research model, not a calibrated field-dependent M(H).
    applied_field_T: float = 0.0
    # Property model (room temperature)
    Js_a: float = 2.0  # polarization (T) of the c -> 0 end member (Fe,Co-rich)
    Js_b: float = 0.1  # polarization (T) of the c -> 1 end member (Cr-rich)
    alpha_c: float = 0.35  # coercivity reduction from ideal Stoner-Wohlfarth (incoherent reversal)
    R_coh_nm: float = 13.0  # radius above which curling reduces the nucleation field

    @property
    def T_top(self) -> float:
        return self.T_s / (4.0 * self.c0 * (1.0 - self.c0))

    @property
    def E0(self) -> float:
        return R_GAS * self.T_top / self.Vm

    @property
    def ell(self) -> float:
        return math.sqrt(2.0 * self.kappa / self.E0)

    @property
    def tau0(self) -> float:
        return self.ell**2 * self.T_ref / (self.D_ref * self.c0 * (1.0 - self.c0) * self.T_top)

    def D(self, T: float) -> float:
        return self.D_ref * math.exp(-self.Q / R_GAS * (1.0 / T - 1.0 / self.T_ref))

    def m(self, T: float) -> float:
        """Dimensionless mobility."""
        return self.D(T) / self.D_ref * (self.T_ref / T)

    def theta(self, T: float) -> float:
        return T / self.T_top

    def b(self, T: float, field_on: bool) -> float:
        if not field_on:
            return 0.0
        g = max(0.0, 1.0 - (T / self.T_curie) ** 2) / max(1e-12, 1.0 - (self.T_ref / self.T_curie) ** 2)
        return self.B0 * g / self.E0

    @property
    def eps_coef(self) -> float:
        """Thermal-noise strength per unit theta (dimensionless)."""
        if not self.thermal_noise:
            return 0.0
        omega_atom = self.Vm / 6.02214076e23
        return omega_atom / (self.ell**2 * self.d_z_nm * 1e-9)

    @property
    def beta_M(self) -> float:
        """mu0 Ms_a^2 / E0 with Ms_a = Js_a / mu0 (dimensionless magnetostatic scale)."""
        return (self.Js_a**2 / MU0) / self.E0

    def Tc_of(self, c: float) -> float:
        return self.Tc_a + (self.Tc_b - self.Tc_a) * c

    @property
    def e_el(self) -> float:
        return self.B_el / self.E0

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **changes) -> "MaterialParams":
        return replace(self, **changes)


# ---------------------------------------------------------------------------------------
# Thermodynamics (dimensionless)
# ---------------------------------------------------------------------------------------


def fprime(c: np.ndarray, theta: float) -> np.ndarray:
    return theta * np.log(c / (1.0 - c)) + 2.0 * (1.0 - 2.0 * c)


def fsecond(c, theta: float):
    return theta / (c * (1.0 - c)) - 4.0


def fbulk(c: np.ndarray, theta: float) -> np.ndarray:
    return theta * (c * np.log(c) + (1.0 - c) * np.log(1.0 - c)) + 2.0 * c * (1.0 - c)


def binodal(theta: float) -> tuple[float, float]:
    """Coexisting compositions of the symmetric regular solution at theta = T/T_top < 1."""
    if theta >= 1.0:
        return 0.5, 0.5
    g = lambda c: math.log(c / (1.0 - c)) - (2.0 / theta) * (2.0 * c - 1.0)  # noqa: E731
    lo = brentq(g, 1e-14, 0.5 - 1e-12) if theta < 0.999999 else 0.5
    return lo, 1.0 - lo


def spinodal_theta(c: float) -> float:
    return 4.0 * c * (1.0 - c)


def linear_growth_rate(k2: np.ndarray, kx2_over_k2: np.ndarray, *, theta: float, c0: float,
                       m: float, b: float) -> np.ndarray:
    """Exact growth rate of a small Fourier mode about c0 for the dimensionless model."""
    return -m * k2 * (fsecond(c0, theta) + k2 + b * kx2_over_k2)


# ---------------------------------------------------------------------------------------
# Processing schedules
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One schedule segment: linear temperature ramp from T_start to T_end over hours."""

    hours: float
    T_start: float
    T_end: float
    field: bool

    def T_at(self, frac: float) -> float:
        return self.T_start + (self.T_end - self.T_start) * frac


@dataclass(frozen=True)
class Schedule:
    segments: tuple[Segment, ...]
    label: str = ""
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def hours(self) -> float:
        return sum(s.hours for s in self.segments)

    def to_dict(self) -> dict:
        return {"label": self.label, "segments": [asdict(s) for s in self.segments],
                "meta": dict(self.meta)}

    @staticmethod
    def from_dict(d: dict) -> "Schedule":
        return Schedule(tuple(Segment(**s) for s in d["segments"]), d.get("label", ""),
                        dict(d.get("meta", {})))


def isothermal(T: float, hours: float, field: bool) -> Segment:
    return Segment(hours=hours, T_start=T, T_end=T, field=field)


def schedule_to_dimensionless(schedule: Schedule, params: MaterialParams) -> list[dict]:
    """Return segments in dimensionless time with T endpoints, for solvers."""
    out = []
    for seg in schedule.segments:
        out.append({"duration": seg.hours * 3600.0 / params.tau0, "T_start": seg.T_start,
                    "T_end": seg.T_end, "field": bool(seg.field)})
    return out


def effective_time(schedule: Schedule, params: MaterialParams, *, n: int = 64) -> float:
    """Integral of m(T(t)) dt in dimensionless units (the 'clock' of a mobility-only model)."""
    total = 0.0
    for seg in schedule.segments:
        fr = (np.arange(n) + 0.5) / n
        ms = np.array([params.m(seg.T_at(x)) for x in fr])
        total += float(ms.mean()) * seg.hours * 3600.0 / params.tau0
    return total


def initial_field(shape: Sequence[int], params: MaterialParams, seed: int) -> np.ndarray:
    """Quenched solution-treated state: c0 plus white noise of the stated rms amplitude."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(tuple(shape))
    noise -= noise.mean()
    noise *= params.noise_rms / max(noise.std(), 1e-15)
    return params.c0 + noise

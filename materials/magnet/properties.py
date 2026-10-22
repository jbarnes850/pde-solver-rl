"""Decision-relevant quantities of interest and the structure-property model.

All functions take a final composition field c (nx, ny), field direction along x, and the
grid spacing in dimensionless units; lengths are converted to nm with params.ell.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .physics import MU0, MaterialParams


@dataclass
class Microstructure:
    p: float  # alpha1 (c < 0.5) area fraction
    c1: float  # median composition of alpha1
    c2: float  # median composition of alpha2
    contrast: float  # c2 - c1
    L_par_nm: float  # correlation period along the field
    L_perp_nm: float  # correlation period across the field
    L_iso_nm: float  # isotropic period 2 pi / <|k|>
    aspect: float  # L_par / L_perp
    d_nm: float  # alpha1 width across the field, p * L_perp
    n_alpha1: int  # connected alpha1 components (periodic)
    amplitude: float  # std of c
    corr_x_nm: float = math.nan  # first zero of the autocorrelation along the field
    corr_y_nm: float = math.nan  # first zero of the autocorrelation across the field
    aspect_corr: float = math.nan  # corr_x / corr_y

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Property:
    Hc_kA_m: float
    Br_T: float
    BHmax_kJ_m3: float
    dN: float
    curling_factor: float

    def to_dict(self) -> dict:
        return asdict(self)


def _rfft_power(c: np.ndarray):
    f = np.fft.rfft2(c - c.mean())
    power = np.abs(f) ** 2
    weight = np.full(power.shape, 2.0)
    weight[:, 0] = 1.0
    if c.shape[1] % 2 == 0:
        weight[:, -1] = 1.0
    return power * weight


def periodic_components(mask: np.ndarray) -> int:
    labels, n = ndimage.label(mask)
    if n == 0:
        return 0
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in zip(labels[0, :], labels[-1, :]):
        if a and b:
            union(a, b)
    for a, b in zip(labels[:, 0], labels[:, -1]):
        if a and b:
            union(a, b)
    return len({find(i) for i in range(1, n + 1)})


def microstructure(c: np.ndarray, dx: float, dy: float, params: MaterialParams) -> Microstructure:
    c = np.asarray(c, dtype=np.float64)
    nx, ny = c.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.rfftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    power = _rfft_power(c)
    total = power.sum()
    ell_nm = params.ell * 1e9
    if total <= 0:
        return Microstructure(float((c < 0.5).mean()), float(c.mean()), float(c.mean()), 0.0,
                              math.inf, math.inf, math.inf, 1.0, math.inf, 0, 0.0)
    kx2 = float((KX**2 * power).sum() / total)
    ky2 = float((KY**2 * power).sum() / total)
    kabs = float((np.sqrt(KX**2 + KY**2) * power).sum() / total)
    L_par = 2 * np.pi / math.sqrt(max(kx2, 1e-30)) * ell_nm
    L_perp = 2 * np.pi / math.sqrt(max(ky2, 1e-30)) * ell_nm
    L_iso = 2 * np.pi / max(kabs, 1e-30) * ell_nm
    alpha1 = c < 0.5
    p = float(alpha1.mean())
    c1 = float(np.median(c[alpha1])) if alpha1.any() else float("nan")
    c2 = float(np.median(c[~alpha1])) if (~alpha1).any() else float("nan")
    cx, cy = autocorrelation_zeros(c, dx, dy)
    return Microstructure(
        p=p, c1=c1, c2=c2, contrast=c2 - c1, L_par_nm=L_par, L_perp_nm=L_perp, L_iso_nm=L_iso,
        aspect=L_par / L_perp, d_nm=p * L_perp, n_alpha1=periodic_components(alpha1),
        amplitude=float(c.std()), corr_x_nm=cx * ell_nm, corr_y_nm=cy * ell_nm,
        aspect_corr=cx / cy if cy > 0 else math.nan,
    )


def _first_zero(profile: np.ndarray, spacing: float) -> float:
    """First zero crossing of a normalized autocorrelation profile (linear interpolation)."""
    for i in range(1, len(profile)):
        if profile[i] <= 0:
            a, b = profile[i - 1], profile[i]
            return spacing * (i - 1 + a / (a - b))
    return spacing * len(profile)


def autocorrelation_zeros(c: np.ndarray, dx: float, dy: float) -> tuple[float, float]:
    f = np.fft.fft2(c - c.mean())
    corr = np.fft.ifft2(np.abs(f) ** 2).real
    corr /= corr[0, 0]
    nx, ny = c.shape
    return _first_zero(corr[: nx // 2, 0], dx), _first_zero(corr[0, : ny // 2], dy)


def prolate_dN(aspect: float) -> float:
    """N_perp - N_par for a prolate spheroid elongated along the field (0 if aspect <= 1)."""
    a = float(aspect)
    if not math.isfinite(a) or a <= 1.0 + 1e-9:
        return 0.0
    e = math.sqrt(1.0 - 1.0 / a**2)
    n_par = (1 - e**2) / (2 * e**3) * (math.log((1 + e) / (1 - e)) - 2 * e)
    n_perp = (1 - n_par) / 2
    return n_perp - n_par


def magnet_property(ms: Microstructure, params: MaterialParams) -> Property:
    """Shape-anisotropy coercivity with Neel interaction, curling cut-off and a linear loop."""
    if not (math.isfinite(ms.c1) and math.isfinite(ms.c2)):
        return Property(0.0, 0.0, 0.0, 0.0, 0.0)
    js = lambda c: params.Js_a * (1 - c) + params.Js_b * c  # noqa: E731
    j1, j2 = js(ms.c1), js(ms.c2)
    dMs = max(j1 - j2, 0.0) / MU0
    dN = prolate_dN(ms.aspect)
    radius = 0.5 * ms.d_nm
    curl = 1.0 if radius <= params.R_coh_nm else (params.R_coh_nm / radius) ** 2
    Hc = params.alpha_c * (1 - ms.p) * dN * dMs * curl
    Br = ms.p * j1 + (1 - ms.p) * j2
    if MU0 * Hc <= Br / 2:
        bh = Hc * (Br - MU0 * Hc)
    else:
        bh = Br**2 / (4 * MU0)
    return Property(Hc_kA_m=Hc / 1e3, Br_T=Br, BHmax_kJ_m3=bh / 1e3, dN=dN, curling_factor=curl)

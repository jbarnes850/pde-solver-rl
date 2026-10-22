"""Reference simulator for the pseudo-binary spinodal magnet model (written independently of the models under test).

Batched pseudo-spectral solver in torch (GPU or CPU). Each batch member may carry its own
material parameters, initial field and processing schedule. Time is physical (seconds);
member i evolves as dc/dt = r_i(t) lap~ mu~_i with r_i = m_i(T_i(t)) / tau0_i.

Time integration: stabilized semi-implicit schemes, with the gradient, stabilizer and
magnetostatic terms implicit and f'(c) - S c explicit.
  * order=1: first-order stabilized semi-implicit (SSI1)
  * order=2: variable-step SBDF2 (restarted with SSI1 after any schedule discontinuity)
Step size is adapted on the largest per-step composition change (tolerance `dc_tol`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import torch

from .physics import MaterialParams, Schedule, binodal

C_FLOOR = 1e-9


@dataclass
class Grid:
    nx: int
    ny: int
    Lx: float  # dimensionless box length along the field (x)
    Ly: float

    @property
    def dx(self) -> float:
        return self.Lx / self.nx

    @property
    def dy(self) -> float:
        return self.Ly / self.ny


def wavenumbers(grid: Grid, device, dtype):
    kx = 2.0 * math.pi * torch.fft.fftfreq(grid.nx, d=grid.dx, device=device, dtype=dtype)
    ky = 2.0 * math.pi * torch.fft.rfftfreq(grid.ny, d=grid.dy, device=device, dtype=dtype)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    k2 = KX**2 + KY**2
    kx2_over_k2 = torch.where(k2 > 0, KX**2 / torch.where(k2 > 0, k2, torch.ones_like(k2)),
                              torch.zeros_like(k2))
    return k2, kx2_over_k2, KX, KY


def band_limited_noise(grid: Grid, rms: float, seed: int, k_cut: float = 1.0) -> np.ndarray:
    """Resolution-independent initial noise: random Fourier amplitudes for |k| <= k_cut.

    The amplitudes are drawn per integer mode index, so refining the grid on the same box
    reproduces the same continuous field (needed for refinement studies).
    """
    imax = int(math.floor(k_cut * grid.Lx / (2 * math.pi)))
    jmax = int(math.floor(k_cut * grid.Ly / (2 * math.pi)))
    if 2 * imax + 1 > grid.nx or 2 * jmax + 1 > grid.ny:
        raise ValueError("grid too coarse for the requested noise band")
    rng = np.random.default_rng(seed)
    re = rng.standard_normal((2 * imax + 1, 2 * jmax + 1))
    im = rng.standard_normal((2 * imax + 1, 2 * jmax + 1))
    spec = np.zeros((grid.nx, grid.ny), dtype=np.complex128)
    for a, i in enumerate(range(-imax, imax + 1)):
        for bb, j in enumerate(range(-jmax, jmax + 1)):
            kk = (2 * math.pi * i / grid.Lx) ** 2 + (2 * math.pi * j / grid.Ly) ** 2
            if 0 < kk <= k_cut**2:
                spec[i % grid.nx, j % grid.ny] = re[a, bb] + 1j * im[a, bb]
    field = np.fft.ifft2(spec).real
    field -= field.mean()
    std = field.std()
    return field * (rms / std if std > 0 else 0.0)


@lru_cache(maxsize=100_000)
def _stabilizer(theta_rounded: float) -> float:
    """About max f''(c) over the reachable composition range at this temperature."""
    lo, _ = binodal(min(theta_rounded, 0.9999))
    c_lo = max(0.6 * lo, 0.02)
    return max(1.0, theta_rounded / (c_lo * (1 - c_lo)) - 4.0)


@dataclass
class Coefficients:
    """Piecewise schedule of the dimensionless PDE coefficients in a generic time unit.

    Segment s spans (seg_end[s-1], seg_end[s]] with seg_end[-1] = 0. Within it, theta is
    linear and log(m) is linear in time; b is constant. `rate_scale` converts m into the
    rate per unit of the generic time (1 for dimensionless time, 1/tau0 for seconds).
    """

    seg_end: np.ndarray
    theta0: np.ndarray
    theta1: np.ndarray
    logm0: np.ndarray
    logm1: np.ndarray
    b: np.ndarray
    rate_scale: float = 1.0
    e_el: float = 0.0  # cubic elastic anisotropy coefficient (dimensionless, constant)
    grain_angle_deg: float = 0.0
    # Near-Curie magnetization model: (T_top, Tc_a, Tc_b, width); b(t) then holds beta_M
    # while the field is on and 0 otherwise.
    curie: tuple | None = None
    eps_th: float = 0.0  # thermal noise strength per unit theta (0 = deterministic)
    zeeman_ratio: float = 0.0  # h/beta_M = applied induction / Js_a for Curie members

    @property
    def end(self) -> float:
        return float(self.seg_end[-1])

    @property
    def bounds(self) -> list[float]:
        return [0.0] + [float(x) for x in self.seg_end]

    def at(self, t: float, left: bool = False):
        """(rate, theta, b, segment index) at time t; rate 0 after the end.

        Right-continuous by default; left=True returns the left limit, which is what an
        implicit evaluation at the end of a step that lands on a segment end needs.
        """
        tol = 1e-12 * max(1.0, self.end)
        after = t > self.end + tol if left else t >= self.end - tol
        if after:
            s = len(self.seg_end) - 1
            return 0.0, float(self.theta1[s]), float(self.b[s]), s
        s = int(np.searchsorted(self.seg_end, t - tol if left else t, side="left" if left else "right"))
        s = min(s, len(self.seg_end) - 1)
        start = 0.0 if s == 0 else float(self.seg_end[s - 1])
        span = float(self.seg_end[s]) - start
        f = 0.0 if span <= 0 else min(max((t - start) / span, 0.0), 1.0)
        theta = self.theta0[s] + f * (self.theta1[s] - self.theta0[s])
        logm = self.logm0[s] + f * (self.logm1[s] - self.logm0[s])
        return float(np.exp(logm)) * self.rate_scale, float(theta), float(self.b[s]), s


class _VecCoeffs:
    """All members' piecewise coefficients evaluated at one time with array operations."""

    def __init__(self, mem: list[Coefficients]):
        S = max(len(m.seg_end) for m in mem)
        B = len(mem)
        pad = lambda attr, fill: np.array([np.concatenate([getattr(m, attr), np.full(S - len(m.seg_end), fill)])  # noqa: E731
                                           for m in mem])
        self.ends = pad("seg_end", np.inf)
        self.th0, self.th1 = pad("theta0", 1.0), pad("theta1", 1.0)
        self.lm0, self.lm1 = pad("logm0", 0.0), pad("logm1", 0.0)
        self.b = pad("b", 0.0)
        self.scale = np.array([m.rate_scale for m in mem])
        self.end = np.array([m.end for m in mem])
        self.nseg = np.array([len(m.seg_end) for m in mem])
        self.starts = np.concatenate([np.zeros((B, 1)), self.ends[:, :-1]], axis=1)
        self.rows = np.arange(B)
        self.tol = 1e-12 * np.maximum(1.0, self.end)

    def at(self, t: float, left: bool = False):
        if left:
            s = (t - self.tol[:, None] > self.ends).sum(axis=1)
            after = t > self.end + self.tol
        else:
            s = (t >= self.ends).sum(axis=1)
            after = t >= self.end - self.tol
        s = np.minimum(s, self.nseg - 1)
        r = self.rows
        start, stop = self.starts[r, s], self.ends[r, s]
        span = stop - start
        f = np.where(span > 0, np.clip((t - start) / np.where(span > 0, span, 1.0), 0.0, 1.0), 0.0)
        theta = self.th0[r, s] + f * (self.th1[r, s] - self.th0[r, s])
        rate = np.exp(self.lm0[r, s] + f * (self.lm1[r, s] - self.lm0[r, s])) * self.scale
        rate = np.where(after, 0.0, rate)
        theta = np.where(after, self.th1[r, self.nseg - 1], theta)
        return rate, theta, self.b[r, s]


def schedule_coefficients(params: MaterialParams, schedule: Schedule, *, pieces_per_ramp: int = 24,
                          time_unit: str = "seconds") -> Coefficients:
    """Map a physical schedule to piecewise coefficients (ramps subdivided)."""
    if not np.isfinite(params.applied_field_T) or params.applied_field_T < 0:
        raise ValueError("applied_field_T must be finite and nonnegative")
    if params.applied_field_T and (params.mag_model != "curie" or params.Js_a <= 0):
        raise ValueError("Applied-field hypothesis requires the Curie law and positive Js_a")
    ends, th0, th1, lm0, lm1, bb = [], [], [], [], [], []
    t = 0.0
    for seg in schedule.segments:
        n = 1 if seg.T_start == seg.T_end else pieces_per_ramp
        dur = seg.hours * 3600.0 if time_unit == "seconds" else seg.hours * 3600.0 / params.tau0
        for i in range(n):
            Ta, Tb = seg.T_at(i / n), seg.T_at((i + 1) / n)
            t += dur / n
            ends.append(t)
            th0.append(params.theta(Ta))
            th1.append(params.theta(Tb))
            lm0.append(math.log(params.m(Ta)))
            lm1.append(math.log(params.m(Tb)))
            if params.mag_model == "curie":
                bb.append(params.beta_M if seg.field else 0.0)
            else:
                bb.append(params.b(0.5 * (Ta + Tb), seg.field))
    scale = 1.0 / params.tau0 if time_unit == "seconds" else 1.0
    arr = lambda v: np.asarray(v, dtype=np.float64)  # noqa: E731
    curie = ((params.T_top, params.Tc_a, params.Tc_b, params.curie_width)
             if params.mag_model == "curie" else None)
    return Coefficients(arr(ends), arr(th0), arr(th1), arr(lm0), arr(lm1), arr(bb), scale,
                        params.e_el, params.grain_angle_deg, curie, params.eps_coef,
                        params.applied_field_T / params.Js_a if params.applied_field_T else 0.0)


def curie_g(c, T, Tc_a, Tc_b, width):
    """Normalized magnetization g = Ms/Ms_a and dg/dc for the regularized Curie law."""
    Tc = Tc_a + (Tc_b - Tc_a) * c
    x = 1.0 - T / Tc
    root = torch.sqrt(x * x + width * width)
    s = 0.5 * (x + root)
    ds = 0.5 * (1.0 + x / root)
    h = torch.sqrt(s)
    dx_dc = T * (Tc_b - Tc_a) / (Tc * Tc)
    dh = ds / (2.0 * h) * dx_dc
    return (1.0 - c) * h, -h + (1.0 - c) * dh


def elastic_symbol(grid: Grid, angle_deg: float, device, dtype):
    """G(k) = 4 n1^2 n2^2 with n the unit wavevector in crystal axes (0 at k = 0)."""
    _, _, KX, KY = wavenumbers(grid, device, dtype)
    k = torch.sqrt(KX**2 + KY**2)
    safe = torch.where(k > 0, k, torch.ones_like(k))
    a = math.radians(angle_deg)
    n1 = (math.cos(a) * KX + math.sin(a) * KY) / safe
    n2 = (-math.sin(a) * KX + math.cos(a) * KY) / safe
    return torch.where(k > 0, 4 * n1**2 * n2**2, torch.zeros_like(k))


def simulate_physical(c_init, grid, members, *, out_hours=None, **kwargs):
    """Convenience wrapper: members are (MaterialParams, Schedule); time in seconds."""
    coeffs = [schedule_coefficients(p, s) for p, s in members]
    out_times = [h * 3600.0 for h in (out_hours or [])]
    res = simulate(c_init, grid, coeffs, out_times=out_times, **kwargs)
    res["snapshots"] = {k / 3600.0: v for k, v in res["snapshots"].items()}
    return res


def simulate(
    c_init: np.ndarray,
    grid: Grid,
    members: Sequence[Coefficients],
    *,
    out_times: Sequence[float] | None = None,
    order: int = 2,
    dc_tol: float = 0.01,
    h_max: float = 2.0,
    dt0: float = 1e-3,
    device: str = "cuda",
    dtype=torch.float64,
    max_steps: int = 5_000_000,
    record_energy: bool = False,
    noise_seed: int = 0,
    noise_groups: Sequence[int] | None = None,
    noise_factorization: str = "divergence",
    progress_callback=None,
    progress_every: int = 1000,
    physics_provider=None,
):
    """Integrate a batch of coefficient schedules. Returns final fields, snapshots, stats.

    out_times are in the members' generic time unit and shared by all members; the final
    state of each member is always returned. h_max caps the dimensionless step m*dt.
    """
    if noise_factorization not in ("divergence", "spectral_sqrt"):
        raise ValueError("Unknown noise factorization")
    B = len(members)
    assert c_init.shape == (B, grid.nx, grid.ny)
    mem = list(members)
    if any(m.zeeman_ratio and m.curie is None for m in mem):
        raise ValueError("Nonzero Zeeman ratio requires a Curie constitutive law")
    if physics_provider is not None:
        if str(device) != "cpu" or any(m.curie is not None or m.e_el != 0 for m in mem):
            raise ValueError("The supplied-physics bridge is currently CPU, M1, zero-elastic only")
    t_end = max(m.end for m in mem)
    seg_bounds = sorted({b for m in mem for b in m.bounds if b > 0})
    breaks = sorted(set(seg_bounds) | set(out_times or []))
    breaks = [b for b in breaks if 0.0 < b <= t_end * (1 + 1e-12)]
    out_set = sorted(out_times or [])
    tiny = 1e-12 * max(1.0, t_end)

    k2, kxk, _, _ = wavenumbers(grid, device, dtype)
    if physics_provider is not None:
        k2, kxk = physics_provider.spectral_tensors(grid, dtype)
    c = torch.as_tensor(c_init, device=device, dtype=dtype)
    c_hat = torch.fft.rfft2(c)
    snapshots: dict[float, np.ndarray] = {}
    energies: list[tuple[float, np.ndarray]] = []
    final = [None] * B

    # One stabilizer per member, fixed over the run (the explicit and implicit S terms then
    # cancel to the scheme's order even when theta varies); sized for the coldest segment.
    S_member = torch.tensor(
        [_stabilizer(round(float(min(m.theta0.min(), m.theta1.min())), 4)) for m in mem],
        device=device, dtype=dtype).view(B, 1, 1)

    # Elastic anisotropy: constant per member, folded into the implicit operator.
    el_terms = []
    for m in mem:
        if m.e_el != 0.0:
            el_terms.append(m.e_el * elastic_symbol(grid, m.grain_angle_deg, device, dtype))
        else:
            el_terms.append(torch.zeros_like(k2))
    EL = torch.stack(el_terms)
    curie_mask = [m.curie is not None for m in mem]
    any_curie = any(curie_mask)
    if any_curie:
        cur = [m.curie if m.curie is not None else (1.0, 1.0, 1.0, 1.0) for m in mem]
        T_top_t = torch.tensor([v[0] for v in cur], device=device, dtype=dtype).view(B, 1, 1)
        Tca_t = torch.tensor([v[1] for v in cur], device=device, dtype=dtype).view(B, 1, 1)
        Tcb_t = torch.tensor([v[2] for v in cur], device=device, dtype=dtype).view(B, 1, 1)
        wid_t = torch.tensor([v[3] for v in cur], device=device, dtype=dtype).view(B, 1, 1)
        cmask = torch.tensor(curie_mask, device=device, dtype=dtype).view(B, 1, 1)
        zratio = torch.tensor([m.zeeman_ratio for m in mem], device=device, dtype=dtype).view(B, 1, 1)
    # Thermal noise (conserved): increment div(sqrt(2 r eps h / (dx dy)) zeta) per step, added
    # after the deterministic update so step control is not driven by noise.
    eps_t = torch.tensor([m.eps_th for m in mem], device=device, dtype=dtype).view(B, 1, 1)
    any_noise = bool((eps_t > 0).any())
    noise_index = None
    noise_count = B
    if noise_groups is not None:
        if len(noise_groups) != B:
            raise ValueError("noise_groups must have one label per batch member")
        _, inverse = np.unique(np.asarray(noise_groups), return_inverse=True)
        noise_count = int(inverse.max()) + 1
        noise_index = torch.as_tensor(inverse, device=device)
    if any_noise:
        _, _, KXg, KYg = wavenumbers(grid, device, dtype)
        lowpass = (k2 < 0.25).to(dtype)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(noise_seed))
    # Implicit magnetostatic stabilization for Curie members: b_stab = b * G2_STAB.
    G2_STAB = 4.0

    vec = _VecCoeffs(mem) if physics_provider is None else physics_provider.coefficient_vector(mem)

    def coeff_tensors(t, left=False):
        rate, theta, b = vec.at(t, left)
        arr = torch.tensor(np.stack([rate, theta, b]), device=device, dtype=dtype)
        return (arr[0].view(B, 1, 1), arr[1].view(B, 1, 1), arr[2].view(B, 1, 1), S_member)

    def explicit(c_real, theta, S):
        if physics_provider is not None:
            return torch.fft.rfft2(physics_provider.bulk_tensor(c_real, theta) - S * c_real)
        cc = torch.clamp(c_real, C_FLOOR, 1 - C_FLOOR)
        g = theta * torch.log(cc / (1 - cc)) + 2.0 * (1 - 2 * c_real) - S * c_real
        return torch.fft.rfft2(g)

    t = 0.0
    dt = dt0
    steps = 0
    rejected = 0
    prev = None  # (c_hat_prev, N_prev, dt_prev)
    while t < t_end - tiny and steps < max_steps:
        nxt = next((b for b in breaks if b > t + tiny), t_end)
        step = min(dt, nxt - t)
        rate0, theta0, b0, S0 = coeff_tensors(t)
        rmax = max(float(vec.at(t)[0].max()), float(vec.at(t + step)[0].max()))
        if rmax > 0 and step * rmax > h_max:
            step = h_max / rmax
        at_boundary = any(abs(t - b) <= tiny for b in seg_bounds)
        first_order = order == 1 or prev is None or at_boundary
        # Implicit terms at the new time level for SBDF2, at the midpoint for SSI1; the
        # stabilizer S is the same in the explicit and implicit parts of one step.
        # Left limit at the new level: coefficients are right-continuous at segment ends.
        if first_order:
            rate1, theta1, b1, _ = coeff_tensors(t + 0.5 * step)
        else:
            rate1, theta1, b1, _ = coeff_tensors(t + step, left=True)
        c_real = torch.fft.irfft2(c_hat, s=(grid.nx, grid.ny))
        N_now = -rate0 * k2 * explicit(c_real, theta0, S0)
        if any_curie:
            # Nonlinear magnetostatic potential b * g'(c) * A[g(c)] (explicit), minus the
            # implicit stabilization b * G2_STAB * A[c] (explicit part of the split).
            T_now = theta0 * T_top_t
            g, gp = curie_g(torch.clamp(c_real, 1e-6, 1 - 1e-6), T_now, Tca_t, Tcb_t, wid_t)
            Ag = torch.fft.irfft2(kxk * torch.fft.rfft2(g), s=(grid.nx, grid.ny))
            mag = b0 * cmask * (gp * Ag - zratio * gp)
            N_now = N_now - rate0 * k2 * (torch.fft.rfft2(mag)
                                          - b0 * cmask * G2_STAB * kxk * c_hat)
        b_impl1 = b1 * torch.where(cmask > 0, G2_STAB, 1.0) if any_curie else b1
        lam = rate1 * k2 * (S0 + k2 + b_impl1 * kxk + EL)
        if first_order:
            numer = c_hat + step * N_now
            denom = 1 + step * lam
        else:
            c_prev, N_prev, dt_prev = prev
            w = step / dt_prev
            a0 = (1 + 2 * w) / (1 + w)
            a1 = -(1 + w)
            a2 = w * w / (1 + w)
            numer = -a1 * c_hat - a2 * c_prev + step * ((1 + w) * N_now - w * N_prev)
            denom = a0 + step * lam
        det_hat = numer / denom
        if any_noise:
            # Semi-implicit Euler-Maruyama: the conserved noise increment enters the numerator,
            # so stiff high-k noise is damped within the same step.
            r_mid = 0.5 * (rate0 + rate1)
            amp = torch.sqrt(torch.clamp(2.0 * r_mid * eps_t * theta1 * step / (grid.dx * grid.dy), min=0.0))
            zx = torch.randn((noise_count, grid.nx, grid.ny), generator=gen, device=device, dtype=dtype)
            if noise_factorization == "divergence":
                # Historical default and random stream are intentionally preserved.
                zy = torch.randn((noise_count, grid.nx, grid.ny), generator=gen, device=device, dtype=dtype)
                if noise_index is not None:
                    zx, zy = zx[noise_index], zy[noise_index]
                new_hat = det_hat + amp * 1j * (KXg * torch.fft.rfft2(zx) + KYg * torch.fft.rfft2(zy)) / denom
            else:
                # Diagnostic finite-grid covariance factorization, including Nyquist.
                # This does not repair the separate temporal stochastic bias.
                if noise_index is not None:
                    zx = zx[noise_index]
                noise_hat = torch.sqrt(k2) * torch.fft.rfft2(zx)
                new_hat = det_hat + amp * noise_hat / denom
        else:
            new_hat = det_hat
        new_real = torch.fft.irfft2(new_hat, s=(grid.nx, grid.ny))
        if any_noise:
            # Step control on morphology scales only (|k| < 0.5, wavelengths > ~12 ell): the damping
            # of injected grid-scale noise is not an integration error.
            low = torch.fft.irfft2((det_hat - c_hat) * lowpass, s=(grid.nx, grid.ny))
            change = low.abs().amax(dim=(1, 2))
        else:
            change = (new_real - c_real).abs().amax(dim=(1, 2))
        bad = (~torch.isfinite(new_real)).any() or (new_real.amin() <= 0) or (new_real.amax() >= 1)
        if bad or float(change.max()) > 4 * dc_tol:
            dt = step * 0.3
            rejected += 1
            if dt < 1e-12:
                raise FloatingPointError("time step underflow")
            continue
        prev = (c_hat, N_now, step)
        c_hat = new_hat
        t += step
        if abs(t - nxt) <= 1e-9 * max(1.0, abs(nxt)):
            t = nxt
        steps += 1
        worst = float(change.max())
        grow = (dc_tol / max(worst, 1e-12)) ** 0.5
        dt = step * min(1.25, max(0.5, grow))
        if progress_callback is not None and steps % progress_every == 0:
            progress_callback({"steps": steps, "rejected": rejected, "time": t,
                               "end_time": t_end, "step": step, "next_step": dt,
                               "max_rate": rmax, "max_change": worst})
        # Members whose schedule ended: record their final state once.
        for i, m in enumerate(mem):
            if final[i] is None and t >= m.end - tiny:
                final[i] = new_real[i].detach().cpu().numpy().copy()
        for o in out_set:
            if abs(t - o) <= tiny and o not in snapshots:
                snapshots[o] = new_real.detach().cpu().numpy()
        if record_energy:
            energies.append((t, free_energy(new_hat, grid, theta1, k2, b1 * kxk + EL)))
    if t < t_end - tiny:
        raise RuntimeError(f"max_steps reached at t={t}")
    c_real = torch.fft.irfft2(c_hat, s=(grid.nx, grid.ny)).detach().cpu().numpy()
    for i in range(B):
        if final[i] is None:
            final[i] = c_real[i]
    return {
        "final": np.stack(final),
        "snapshots": snapshots,
        "steps": steps,
        "rejected": rejected,
        "energies": energies,
    }


def free_energy(c_hat, grid: Grid, theta, k2, W) -> np.ndarray:
    """Dimensionless free energy per unit area: bulk + gradient + nonlocal (W = b K_x + e G)."""
    c = torch.fft.irfft2(c_hat, s=(grid.nx, grid.ny))
    cc = torch.clamp(c, C_FLOOR, 1 - C_FLOOR)
    bulk = theta * (cc * torch.log(cc) + (1 - cc) * torch.log(1 - cc)) + 2 * cc * (1 - cc)
    n = grid.nx * grid.ny
    weight = torch.full_like(k2, 2.0)
    weight[:, 0] = 1.0
    if grid.ny % 2 == 0:
        weight[:, -1] = 1.0
    power = (c_hat.abs() ** 2) * weight / n**2
    grad = 0.5 * (k2 * power).sum(dim=(1, 2))
    nonlocal_ = 0.5 * (W * power).sum(dim=(1, 2))
    return (bulk.mean(dim=(1, 2)) + grad + nonlocal_).detach().cpu().numpy()

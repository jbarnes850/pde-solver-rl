"""Generated-solver task: statement, invocations, analytic checks, residual selector.

The mathematical problem is fully specified (equation, spectral spatial discretization,
coefficient semantics). The physics mapping from temperature and field to coefficients is
owned by the harness (physics.py) and is not part of the generated code.

Guidance levels (prompt variants), each an attributable intervention:
  plain   - problem statement only (RLVP style).
  hint    - plus one sentence naming the method family (experimenter guidance).
  scheme  - plus the explicit stabilized semi-implicit update formula (experimenter guidance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import fsolve

from .physics import binodal, fbulk, fprime, fsecond
from .sandbox import Invocation, RunResult, Sandbox

RLVP_SYSTEM = """You write numerical PDE solver code.
Return only Python code. Do not use Markdown fences. Do not print. Do not use files,
the network, subprocesses, or environment variables. NumPy and SciPy are allowed.
The verifier will call the exact function signature in the user request.

How the code runs:
- The module is imported once and solver(...) is called once, with positional arguments
  in the order of the signature. Array arguments are float64 NumPy arrays. Scalar
  parameters are Python floats.
- Python 3.12 with NumPy 2.5 and SciPy 1.17 only, two CPU cores, 4 GB of memory and a
  30-second limit. Use only functions that exist in these versions. For example,
  scipy.integrate.solve_ivp accepts only the methods RK45, RK23, DOP853, Radau, BDF
  and LSODA.
- The batch size B and the grid sizes change between calls, and Nx may differ from Ny.
  Read every size from the input arrays. Do not check, assert or reject input shapes
  or values.

What to return:
- One finite float64 NumPy array with exactly the shape stated in the request.
- For time-dependent problems, t_coordinate[0] is 0 and output index 0 is the initial
  state. Take substeps sized by a stability limit and land exactly on every requested
  output time.

How to write it:
- One complete module: imports, any helper functions, then a module-level
  def solver(...) with exactly the requested name and arguments. Define every
  variable before using it.
- Write one clear, vectorized implementation. Do not include alternative versions,
  drafts, tests, example usage or explanations outside comments."""

STATEMENT = """Solve a Cahn-Hilliard phase-separation equation with time-dependent coefficients on a periodic two-dimensional grid:

    c_t = m(t) * Lap(mu),
    mu  = theta(t) * log(c / (1 - c)) + 2 * (1 - 2 * c) - Lap(c) + b(t) * A(c) + e_el * G(c),

for 0 < c < 1 on the periodic rectangle [0, Nx*dx) x [0, Ny*dy). Array axis 1 is x and axis 2 is y. A and G are linear operators defined by Fourier multipliers, both 0 for the zero wavevector. For a nonzero wavevector (kx, ky) with k = sqrt(kx**2 + ky**2), the multiplier of A is kx**2 / k**2 and the multiplier of G is 4 * n1**2 * n2**2 with n1 = (cos(phi)*kx + sin(phi)*ky) / k and n2 = (-sin(phi)*kx + cos(phi)*ky) / k. Compute Lap, A and G spectrally with the FFT on this grid, using wavenumbers 2*pi*fftfreq(N, d=spacing) along each axis.

The coefficients are piecewise in time with n_seg segments. Segment s covers seg_end[s-1] < t <= seg_end[s], where seg_end[-1] is taken as 0. Inside segment s, with f = (t - start) / (seg_end[s] - start) and start the previous segment end:

    theta(t) = seg_theta0[s] + f * (seg_theta1[s] - seg_theta0[s])
    m(t)     = exp(seg_logm0[s] + f * (seg_logm1[s] - seg_logm0[s]))
    b(t)     = seg_b[s]

seg_end[n_seg-1] >= t_coordinate[-1].

Write this function:

    def solver(c0_batch, t_coordinate, dx, dy, seg_end, seg_theta0, seg_theta1, seg_logm0, seg_logm1, seg_b, e_el, phi):
        ...

c0_batch has shape [B, Nx, Ny] with values in (0, 1). t_coordinate has shape [T]. Each seg_* array has shape [n_seg]. e_el and phi are scalars. Return a finite NumPy array with shape [B, T, Nx, Ny]. Returned slice n must correspond to t_coordinate[n]. Internal substeps are allowed. The solution must stay inside (0, 1), conserve the mean of c, and be accurate, because it is compared with a converged reference. Final times reach 5000 and m(t) ranges from 0.001 to 5, so the problem is stiff. Do not hard-code a solution array.
"""

HINT = """
Use a pseudo-spectral, linearly stabilized semi-implicit time-stepping scheme; explicit time stepping is too slow for this problem.
"""

SCHEME = """
Numerical method: use a pseudo-spectral, linearly stabilized semi-implicit scheme. With k2 = kx**2 + ky**2, K and G the multipliers of A and G, a stabilization constant s_stab = 4 and fp(c) = theta*log(c/(1-c)) + 2*(1-2*c), one step of size dt is

    c_hat_new = (c_hat - dt*m*k2*FFT(fp(c) - s_stab*c)) / (1 + dt*m*k2*(s_stab + k2 + b*K + e_el*G))

with theta, m and b evaluated at the step. Use steps with m*dt <= 0.5 and land exactly on segment ends and output times. Clip c to [1e-9, 1 - 1e-9] only inside the logarithm.
"""

SCHEME2 = """
Numerical method (second order): pseudo-spectral, linearly stabilized BDF2 (SBDF2). Let k2 = kx**2 + ky**2, K and G the multipliers of A and G, s_stab = 4, fp(c) = theta*log(c/(1-c)) + 2*(1-2*c) (clip c to [1e-9, 1 - 1e-9] only inside the logarithm),
    N(c, t) = -m(t)*k2*FFT(fp(c) - s_stab*c)   (theta and m evaluated at t)
    L(t)    = m(t)*k2*(s_stab + k2 + b(t)*K + e_el*G).
Split each interval between consecutive boundaries (segment ends and requested output times) into equal steps dt with m*dt <= 1 for the largest m in the interval. In each interval, take the first step with
    c_hat_1 = (c_hat_0 + dt*N(c_0, t_0)) / (1 + dt*L(t_1))
and every later step with
    c_hat_new = (4*c_hat - c_hat_old + 2*dt*(2*N(c, t) - N(c_old, t_old))) / (3 + 2*dt*L(t_new)),
where t_new is the time at the end of the step. The steps land exactly on every boundary.
"""

VARIANTS = {"plain": STATEMENT, "hint": STATEMENT + HINT, "scheme": STATEMENT + SCHEME,
            "scheme2": STATEMENT + SCHEME2}
ARGS = ("c0_batch", "t_coordinate", "dx", "dy", "seg_end", "seg_theta0", "seg_theta1",
        "seg_logm0", "seg_logm1", "seg_b", "e_el", "phi")


def messages(variant: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": RLVP_SYSTEM},
            {"role": "user", "content": VARIANTS[variant]}]


@dataclass
class Problem:
    """A fully specified instance of the mathematical problem."""

    c0: np.ndarray  # (B, Nx, Ny)
    t: np.ndarray
    dx: float
    dy: float
    seg: np.ndarray  # (S, 6): end, theta0, theta1, m0, m1, b  (m in linear units)
    meta: dict = field(default_factory=dict)
    e_el: float = 0.0
    phi: float = 0.0  # radians

    def invocation(self) -> Invocation:
        s = self.seg
        return Invocation(
            argument_order=ARGS,
            arrays={"c0_batch": self.c0, "t_coordinate": self.t, "seg_end": s[:, 0].copy(),
                    "seg_theta0": s[:, 1].copy(), "seg_theta1": s[:, 2].copy(),
                    "seg_logm0": np.log(s[:, 3]), "seg_logm1": np.log(s[:, 4]),
                    "seg_b": s[:, 5].copy()},
            scalars={"dx": float(self.dx), "dy": float(self.dy), "e_el": float(self.e_el),
                     "phi": float(self.phi)},
            expected_shape=(self.c0.shape[0], len(self.t), self.c0.shape[1], self.c0.shape[2]),
        )

    def coefficients(self):
        from .reference import Coefficients  # torch import only where the reference runs
        s = self.seg
        return Coefficients(s[:, 0], s[:, 1], s[:, 2], np.log(s[:, 3]), np.log(s[:, 4]), s[:, 5],
                            1.0, float(self.e_el), math.degrees(self.phi))


# ---------------------------------------------------------------------------------------
# Analytic (reference-free) check cases
# ---------------------------------------------------------------------------------------
C0 = 0.40
# Modes chosen so that no pairwise sum, difference or double lies in the set.
LIN_MODES = [(3, 0), (0, 3), (2, 2), (5, 0), (0, 7), (1, 4)]
LIN_EPS = 1e-8


def g_symbol(kx, ky, phi: float):
    """Multiplier of the cubic-anisotropy operator G (0 at k = 0)."""
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    k = np.sqrt(kx**2 + ky**2)
    safe = np.where(k > 0, k, 1.0)
    n1 = (math.cos(phi) * kx + math.sin(phi) * ky) / safe
    n2 = (-math.sin(phi) * kx + math.cos(phi) * ky) / safe
    return np.where(k > 0, 4 * n1**2 * n2**2, 0.0)


def analytic_log_growth(seg: np.ndarray, t_end: float, kx: float, ky: float, *,
                        e_el: float = 0.0, phi: float = 0.0, n: int = 4000):
    k2 = kx * kx + ky * ky
    G = float(g_symbol(kx, ky, phi))
    total, start = 0.0, 0.0
    for end, th0, th1, m0, m1, b in seg:
        stop = min(end, t_end)
        if stop <= start:
            break
        f = (np.arange(n) + 0.5) / n
        # Integrate over [start, stop] within this segment.
        frac_hi = (stop - start) / (end - start)
        ff = f * frac_hi
        th = th0 + ff * (th1 - th0)
        m = np.exp(np.log(m0) + ff * (np.log(m1) - np.log(m0)))
        lam = -m * k2 * (fsecond(C0, th) + k2 + b * kx * kx / k2 + e_el * G)
        total += float(lam.mean()) * (stop - start)
        start = end
    return total


def linear_problem() -> Problem:
    nx = ny = 64
    dx = dy = 1.5
    x = np.arange(nx)[:, None] * dx
    y = np.arange(ny)[None, :] * dy
    c = np.full((nx, ny), C0)
    for i, j in LIN_MODES:
        c = c + LIN_EPS * np.cos(2 * np.pi * (i * x / (nx * dx) + j * y / (ny * dy)))
    seg = np.array([[1500.0, 0.95, 0.95, 1.3, 1.3, 0.004],
                    [2500.0, 1.05, 1.05, 0.4, 0.4, 0.0],
                    [4000.0, 0.94, 0.90, 1.0, 0.3, 0.0]])
    t = np.array([0.0, 750.0, 1500.0, 2500.0, 4000.0])
    return Problem(c[None], t, dx, dy, seg, {"check": "linear"}, e_el=0.006, phi=0.35)


def band_noise(nx, ny, dx, dy, rms, seed, k_cut=1.0):
    rng = np.random.default_rng(seed)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2
    spec = (rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny))) * (K2 <= k_cut**2)
    spec[0, 0] = 0
    f = np.fft.ifft2(spec).real
    return f * (rms / f.std())


def nonlinear_problem(dense: bool = True) -> Problem:
    nx, ny, dx = 96, 96, 1.0
    c = C0 + band_noise(nx, ny, dx, dx, 0.02, seed=11)
    seg = np.array([[2000.0, 0.90, 0.90, 1.0, 1.0, 0.003],
                    [3000.0, 0.90, 0.86, 1.0, 0.2, 0.0]])
    t = np.linspace(0.0, 3000.0, 61) if dense else np.linspace(0.0, 3000.0, 7)
    return Problem(c[None], t, dx, dx, seg, {"check": "nonlinear"}, e_el=0.004, phi=0.2)


def coexistence(theta: float, b: float, c0: float = C0) -> tuple[float, float]:
    """Coexisting compositions of f(c) + (b/2)(c - c0)^2 (flat layer with a nonlocal term
    whose multiplier is the constant b on every wavevector of the modulation)."""
    def eqs(v):
        lo, hi = v
        g = lambda c: float(fprime(np.array(c), theta) + b * (c - c0))  # noqa: E731
        f = lambda c: float(fbulk(np.array(c), theta) + 0.5 * b * (c - c0) ** 2)  # noqa: E731
        return [g(lo) - g(hi), (f(hi) - f(lo)) - g(lo) * (hi - lo)]
    lo, hi = fsolve(eqs, list(binodal(theta)), xtol=1e-13)
    return float(lo), float(hi)


EQ_E, EQ_PHI = 0.02, 0.3


def equilibrium_problem(along: str, theta=0.88, b=0.05, n=128) -> Problem:
    # A layer modulated along x has A-multiplier 1 and G-multiplier sin^2(2 phi);
    # modulated along y, A-multiplier 0 and the same G-multiplier.
    s2 = math.sin(2 * EQ_PHI) ** 2
    lo, hi = coexistence(theta, (b if along == "x" else 0.0) + EQ_E * s2)
    frac = (hi - C0) / (hi - lo)
    z = np.arange(n) + 0.5
    inside = 0.5 * (np.tanh((z - 0.25 * n) / 2.0) - np.tanh((z - (0.25 + frac) * n) / 2.0))
    prof = hi + (lo - hi) * inside
    prof = C0 + 0.7 * (prof - prof.mean())
    c = np.tile(prof[:, None], (1, 4)) if along == "x" else np.tile(prof[None, :], (4, 1))
    seg = np.array([[30000.0, theta, theta, 1.0, 1.0, b]])
    t = np.array([0.0, 15000.0, 30000.0])
    return Problem(c[None], t, 1.0, 1.0, seg, {"check": f"equilibrium_{along}",
                                                  "expected": [lo, hi]}, e_el=EQ_E, phi=EQ_PHI)


def runtime_problem() -> Problem:
    nx, ny = 256, 128
    c = C0 + band_noise(nx, ny, 1.0, 1.0, 0.005, seed=5)
    seg = np.array([[2500.0, 0.955, 0.955, 2.0, 2.0, 0.002],
                    [5000.0, 0.92, 0.88, 0.5, 0.05, 0.0]])
    t = np.array([0.0, 2500.0, 5000.0])
    return Problem(c[None], t, 1.0, 1.0, seg, {"check": "runtime"}, e_el=0.003, phi=0.0)


# ---------------------------------------------------------------------------------------
# Scoring of checks (uses only analytic expectations and the candidate's own outputs)
# ---------------------------------------------------------------------------------------


def _operators(nx, ny, dx, dy, phi):
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    Kop = np.where(K2 > 0, KX**2 / np.where(K2 > 0, K2, 1), 0)
    return K2, Kop, g_symbol(KX, KY, phi)


def spectral_energy(c: np.ndarray, dx, dy, theta, b, e_el=0.0, phi=0.0) -> float:
    cc = np.clip(c, 1e-12, 1 - 1e-12)
    nx, ny = c.shape
    K2, Kop, G = _operators(nx, ny, dx, dy, phi)
    ch = np.fft.fft2(c) / (nx * ny)
    p = np.abs(ch) ** 2
    return float(fbulk(cc, theta).mean() + 0.5 * (K2 * p).sum() + 0.5 * b * (Kop * p).sum()
                 + 0.5 * e_el * (G * p).sum())


def residual_score(out: np.ndarray, prob: Problem) -> float:
    """RLVP-style reference-free residual on the output grid (no IC or BC check)."""
    c = out[0]
    t = prob.t
    nx, ny = c.shape[1:]
    K2, Kop, G = _operators(nx, ny, prob.dx, prob.dy, prob.phi)
    num, den = 0.0, 0.0
    co = prob.seg
    for n in range(1, len(t) - 1):
        tm = t[n]
        s = int(np.searchsorted(co[:, 0], tm, side="left"))
        start = 0.0 if s == 0 else co[s - 1, 0]
        f = (tm - start) / (co[s, 0] - start)
        th = co[s, 1] + f * (co[s, 2] - co[s, 1])
        m = math.exp(math.log(co[s, 3]) + f * (math.log(co[s, 4]) - math.log(co[s, 3])))
        b = co[s, 5]
        cn = np.clip(c[n], 1e-12, 1 - 1e-12)
        mu_hat = np.fft.fft2(fprime(cn, th)) + (K2 + b * Kop + prob.e_el * G) * np.fft.fft2(c[n])
        rhs = np.fft.ifft2(-m * K2 * mu_hat).real
        dcdt = (c[n + 1] - c[n - 1]) / (t[n + 1] - t[n - 1])
        num += float(((dcdt - rhs) ** 2).mean())
        den += float((rhs**2).mean()) + 1e-300
    return math.sqrt(num / den)


@dataclass
class CheckReport:
    passed: dict
    numbers: dict
    errors: dict
    seconds: float
    residual: float | None

    @property
    def n_passed(self) -> int:
        return sum(bool(v) for v in self.passed.values())

    @property
    def all_passed(self) -> bool:
        return all(self.passed.values())

    def feedback(self) -> str:
        """Plain-language repair feedback built only from analytic checks and own outputs."""
        lines = []
        for name, ok in self.passed.items():
            if ok:
                continue
            if name in self.errors:
                lines.append(f"- {name}: the call failed: {self.errors[name]}")
            else:
                lines.append(f"- {name}: {self.numbers.get(name + '_msg', 'failed')}")
        return "\n".join(lines) if lines else "All checks passed."


def run_checks(code: str, sandbox: Sandbox, *, timeout_runtime: float = 60.0) -> CheckReport:
    passed, numbers, errors = {}, {}, {}
    total = 0.0
    residual = None

    # Linear growth: analytic dispersion through three segments.
    prob = linear_problem()
    r: RunResult = sandbox.run(code, prob.invocation())
    total += r.seconds
    if not r.ok:
        passed["linear"] = False
        errors["linear"] = (r.error or "") + (" | " + r.stderr_tail[-600:] if r.stderr_tail else "")
        if r.error == "timeout":
            errors["linear"] += (" | the call did not finish within the time limit on a 64x64 grid "
                                 "integrated to t = 4000 (m up to 1.3); explicit or very small time "
                                 "steps are too slow for this stiff problem")
    else:
        nx, ny = prob.c0.shape[1:]
        worst, detail = 0.0, []
        for n in range(1, len(prob.t)):
            spec = np.fft.fft2(r.output[0, n] - C0) / (nx * ny) * 2
            for i, j in LIN_MODES:
                kx, ky = 2 * np.pi * i / (nx * prob.dx), 2 * np.pi * j / (ny * prob.dy)
                a = analytic_log_growth(prob.seg, prob.t[n], kx, ky, e_el=prob.e_el, phi=prob.phi)
                if LIN_EPS * math.exp(a) < 1e-13:
                    continue  # predicted amplitude below the round-off floor: not measurable
                amp = abs(spec[i % nx, j % ny])
                meas = math.log(max(amp, 1e-300) / LIN_EPS)
                err = meas - a
                if abs(err) > abs(worst):
                    worst = err
                detail.append((float(prob.t[n]), i, j, round(a, 4), round(meas, 4)))
        ic_err = float(np.abs(r.output[0, 0] - prob.c0[0]).max())
        numbers["linear_worst_log_err"] = worst
        numbers["linear_ic_err"] = ic_err
        # 0.10 log-amplitude over 5-8 e-folds ~ 1-2% growth-rate error. Calibrated on the
        # Reference control (0.051 at m*dt <= 0.5) and mutants (>= 0.35); see check-validation.
        ok = abs(worst) < 0.10 and ic_err < 1e-12
        passed["linear"] = ok
        if not ok:
            bad = max(detail, key=lambda d: abs(d[4] - d[3]))
            numbers["linear_msg"] = (
                f"small-amplitude Fourier modes must grow or decay as exp(integral of "
                f"-m*k2*(f''(c0) + k2 + b*kx2/k2 + e_el*G) dt) with f''(c) = theta/(c(1-c)) - 4 and "
                f"c0 = 0.4; worst log-amplitude error {worst:+.3f} (mode {bad[1]},{bad[2]} at "
                f"t = {bad[0]}: expected {bad[3]}, got {bad[4]})"
                + ("; the returned slice 0 differs from c0_batch" if ic_err >= 1e-12 else ""))

    if errors.get("linear") and ("timeout" in errors["linear"] or "exit_" in errors["linear"]
                                 or "shape" in errors["linear"] or "nonfinite" in errors["linear"]):
        # The program does not run correctly at all: skip the remaining checks (they fail the
        # same way) and report the execution error, which is what a repair must fix first.
        for k in ("bounds", "mass", "energy", "equilibrium_x", "equilibrium_y", "runtime"):
            passed[k] = False
        numbers["early_exit"] = 1.0
        return CheckReport({k: bool(v) for k, v in passed.items()},
                           {k: (v if isinstance(v, str) else float(v)) for k, v in numbers.items()},
                           errors, total, None)

    # Nonlinear: bounds, mass, energy dissipation in the constant segment, residual.
    prob = nonlinear_problem(dense=True)
    r = sandbox.run(code, prob.invocation())
    total += r.seconds
    if not r.ok:
        for k in ("bounds", "mass", "energy"):
            passed[k] = False
            errors[k] = (r.error or "") + (" | " + r.stderr_tail[-600:] if r.stderr_tail else "")
    else:
        out = r.output[0]
        lo, hi = float(out.min()), float(out.max())
        passed["bounds"] = 0.0 < lo and hi < 1.0
        numbers["bounds_msg"] = f"values must stay in (0, 1); got min {lo:.3g}, max {hi:.3g}"
        drift = float(np.abs(out.mean(axis=(1, 2)) - out[0].mean()).max())
        numbers["mass_drift"] = drift
        passed["mass"] = drift < 1e-8
        numbers["mass_msg"] = f"the mean of c must be conserved; max drift {drift:.3g}"
        idx = [n for n, tt in enumerate(prob.t) if tt <= 2000.0]
        e = [spectral_energy(out[n], prob.dx, prob.dy, 0.90, 0.003, prob.e_el, prob.phi)
             for n in idx]
        inc = float(np.max(np.diff(e))) if len(e) > 1 else 0.0
        decrease = float(e[0] - e[-1])
        numbers["energy_max_increase"] = inc
        numbers["energy_decrease"] = decrease
        passed["energy"] = inc <= 1e-9 and decrease > 1e-6
        numbers["energy_msg"] = (
            "during the constant-coefficient segment the free energy "
            "mean(theta*(c ln c + (1-c) ln(1-c)) + 2c(1-c)) + mean(|grad c|^2)/2 + b/2*<A c, c> "
            "+ e_el/2*<G c, c> "
            f"must decrease monotonically; max increase {inc:.3g}, total decrease {decrease:.3g}")
        try:
            residual = residual_score(r.output, prob)
        except Exception:  # noqa: BLE001
            residual = None

    # Flat-interface equilibrium, along x with the field term and along y without effect.
    for along in ("x", "y"):
        prob = equilibrium_problem(along)
        r = sandbox.run(code, prob.invocation())
        total += r.seconds
        key = f"equilibrium_{along}"
        if not r.ok:
            passed[key] = False
            errors[key] = (r.error or "") + (" | " + r.stderr_tail[-600:] if r.stderr_tail else "")
            continue
        fin = r.output[0, -1]
        lo, hi = prob.meta["expected"]
        err = max(abs(float(fin.min()) - lo), abs(float(fin.max()) - hi))
        numbers[f"{key}_err"] = err
        passed[key] = err < 2e-3
        numbers[f"{key}_msg"] = (
            f"a flat two-phase layer modulated along {along} (theta 0.88, b 0.05, e_el {EQ_E}, "
            f"phi {EQ_PHI}) must relax to the coexisting compositions {lo:.4f} and {hi:.4f} of "
            + ("f(c) + ((b + e_el*sin(2 phi)^2)/2)(c - c0)^2" if along == "x"
               else "f(c) + ((e_el*sin(2 phi)^2)/2)(c - c0)^2 (A has no effect here)")
            + f"; got {float(fin.min()):.4f} and {float(fin.max()):.4f}")

    # Runtime at a production-like size.
    prob = runtime_problem()
    rs = Sandbox(timeout_seconds=timeout_runtime)
    r = rs.run(code, prob.invocation())
    total += r.seconds
    passed["runtime"] = r.ok
    numbers["runtime_seconds"] = r.seconds
    if not r.ok:
        errors["runtime"] = f"256x128 grid to t = 5000: {r.error}"
    passed = {k: bool(v) for k, v in passed.items()}
    numbers = {k: (v if isinstance(v, str) else float(v)) for k, v in numbers.items()}
    return CheckReport(passed, numbers, errors, total,
                       None if residual is None else float(residual))

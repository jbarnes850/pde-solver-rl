"""The 71 family-scoped entries in RLVP Appendix tables ``adv-bank`` to ``sw-bank``.

The paper reuses four bare codenames across families. This module therefore preserves the
printed codename and uses ``family_id/codename`` as the unique method ID. Expected orders
marked as nominal are clean-room method inferences, not published implementation results.
"""

from __future__ import annotations

from types import MappingProxyType

from .base import SolverMethod

IMPLEMENTATION_VERSION = "source-derived-clean-room-v1-pending"

AUTHOR_IMPLEMENTATION_UNRELEASED = "author_implementation_unreleased"
EXPECTED_ORDER_INFERRED = "expected_order_inferred_from_method_description"
EXPECTED_ORDER_NOT_STATED = "expected_order_not_stated_by_paper"
CFL_NOT_STATED = "cfl_or_internal_step_rule_not_stated"
TOLERANCE_NOT_STATED = "solver_tolerance_not_stated"


def _method(
    codename: str,
    family_id: str,
    scheme: str,
    integrator: str,
    stability_control: str,
    citation_keys: tuple[str, ...],
    expected_order: str,
    *,
    fallback: str | None = None,
    ambiguity_flags: tuple[str, ...] = (),
) -> SolverMethod:
    return SolverMethod(
        codename=codename,
        family_id=family_id,
        scheme=scheme,
        integrator=integrator,
        stability_control=stability_control,
        citation_keys=citation_keys,
        expected_order=expected_order,
        fallback=fallback,
        implementation_version=IMPLEMENTATION_VERSION,
        ambiguity_flags=(AUTHOR_IMPLEMENTATION_UNRELEASED, *ambiguity_flags),
    )


_ADVECTION1D = (
    _method(
        "semi_lagrangian",
        "advection1d",
        "periodic linear interpolation",
        "characteristic backtrace with interpolation",
        "periodic wrap; internal step rule not stated",
        ("staniforth1991semi",),
        "not stated",
        ambiguity_flags=(EXPECTED_ORDER_NOT_STATED, CFL_NOT_STATED),
    ),
    _method(
        "pdebench_fd2_midpoint_vanleer",
        "advection1d",
        "limited MUSCL reconstruction with Rusanov flux",
        "RK2 midpoint",
        "Van Leer limiting; CFL value not stated",
        ("takamoto2022pdebench", "van1979towards"),
        "second order, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED),
    ),
    _method(
        "pdebench_rusanov2",
        "advection1d",
        "minmod MUSCL reconstruction with Rusanov flux",
        "RK2 midpoint",
        "minmod limiting; CFL value not stated",
        ("takamoto2022pdebench", "leveque2002finite"),
        "second order, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED),
    ),
    _method(
        "central4_rk4",
        "advection1d",
        "fourth-order centered finite difference",
        "classical RK4",
        "internal step rule not stated",
        ("leveque2007finite",),
        "fourth order, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED),
    ),
    _method(
        "muscl_rusanov_ssprk3",
        "advection1d",
        "limited MUSCL reconstruction with Rusanov flux",
        "SSPRK3",
        "limiting; CFL value not stated",
        ("van1979towards", "leveque2002finite", "gottlieb2001strong"),
        "second order, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED),
    ),
    _method(
        "spectral_rk4",
        "advection1d",
        "Fourier pseudospectral derivative",
        "classical RK4 with CFL substeps",
        "CFL substeps; CFL value not stated",
        ("trefethen2000spectral",),
        "spectral in space and fourth order in time, nominal",
        ambiguity_flags=(
            EXPECTED_ORDER_INFERRED,
            CFL_NOT_STATED,
            "same_paper_description_as_fourier_spectral_rk4",
        ),
    ),
    _method(
        "fourier_exact_shift",
        "advection1d",
        "FFT phase shift",
        "exact shift per output interval",
        "no step stability limit stated",
        ("trefethen2000spectral",),
        "exact for represented Fourier modes",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED,),
    ),
    _method(
        "fourier_spectral_rk4",
        "advection1d",
        "Fourier pseudospectral derivative",
        "classical RK4 with CFL substeps",
        "CFL substeps; CFL value not stated",
        ("trefethen2000spectral",),
        "spectral in space and fourth order in time, nominal",
        ambiguity_flags=(
            EXPECTED_ORDER_INFERRED,
            CFL_NOT_STATED,
            "same_paper_description_as_spectral_rk4",
        ),
    ),
)


def _burgers_flux_methods(
    flux_name: str, citation_keys: tuple[str, ...]
) -> tuple[SolverMethod, ...]:
    scheme = f"{flux_name} flux with centered finite-difference viscous Laplacian"
    return tuple(
        _method(
            f"{flux_name}_{integrator_id}",
            "burgers1d",
            scheme,
            integrator,
            "CFL substeps; CFL and diffusion limits not stated",
            citation_keys,
            "first order, nominal",
            ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED),
        )
        for integrator_id, integrator in (
            ("euler", "forward Euler"),
            ("ssprk2", "SSPRK2"),
            ("ssprk3", "SSPRK3"),
        )
    )


_BURGERS1D = (
    *_burgers_flux_methods(
        "godunov", ("godunov1959finite", "leveque2002finite", "gottlieb2001strong")
    ),
    *_burgers_flux_methods("lax_friedrichs", ("leveque2002finite", "gottlieb2001strong")),
    *_burgers_flux_methods(
        "rusanov", ("rusanov1962calculation", "leveque2002finite", "gottlieb2001strong")
    ),
    *tuple(
        _method(
            f"muscl_rusanov_{integrator_id}",
            "burgers1d",
            "minmod MUSCL reconstruction, Rusanov flux, and centered viscous Laplacian",
            integrator,
            "minmod limiting; CFL and diffusion limits not stated",
            ("van1979towards", "leveque2002finite", "gottlieb2001strong"),
            expected_order,
            ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED),
        )
        for integrator_id, integrator, expected_order in (
            ("euler", "forward Euler", "first order, nominal"),
            ("rk2", "RK2", "second order, nominal"),
            ("ssprk2", "SSPRK2", "second order, nominal"),
            ("ssprk3", "SSPRK3", "second order; reported by Burgers MMS"),
        )
    ),
    _method(
        "lax_wendroff",
        "burgers1d",
        "two-step Lax-Wendroff with centered viscous Laplacian",
        "single step within CFL substeps",
        "CFL substeps; CFL and diffusion limits not stated",
        ("lax1959systems", "leveque2002finite"),
        "second order; reported by Burgers MMS",
        ambiguity_flags=(CFL_NOT_STATED,),
    ),
    _method(
        "maccormack",
        "burgers1d",
        "MacCormack predictor-corrector with centered viscous Laplacian",
        "predictor-corrector",
        "internal step rule not stated",
        ("maccormack2003effect", "leveque2002finite"),
        "second order; reported by Burgers MMS",
        ambiguity_flags=(CFL_NOT_STATED,),
    ),
    _method(
        "imex_cn_fixed",
        "burgers1d",
        "Rusanov flux with periodic finite-difference Laplacian",
        "explicit advection and Crank-Nicolson diffusion by operator splitting",
        "fixed implicit diffusion solve; CFL value not stated",
        ("ascher1997implicit", "hundsdorfer2003numerical"),
        "not stated",
        ambiguity_flags=(EXPECTED_ORDER_NOT_STATED, CFL_NOT_STATED, TOLERANCE_NOT_STATED),
    ),
    _method(
        "spectral_rk4_dealias",
        "burgers1d",
        "Fourier pseudospectral derivative with two-thirds dealiasing",
        "classical RK4",
        "two-thirds dealiasing; internal step rule not stated",
        ("orszag1971elimination", "trefethen2000spectral"),
        "spectral in space and fourth order in time, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED),
    ),
    _method(
        "spectral_if_rk4_dealias",
        "burgers1d",
        "dealiased Fourier nonlinear flux with exact Fourier diffusion",
        "integrating-factor RK4",
        "dealiasing; internal step rule not stated",
        ("orszag1971elimination", "trefethen2000spectral", "kassam2005fourth"),
        "superalgebraic in space; reported as saturated by Burgers MMS",
        ambiguity_flags=(CFL_NOT_STATED,),
    ),
)


def _explicit_rd_methods(family_id: str, *, dimension: str) -> tuple[SolverMethod, ...]:
    return tuple(
        _method(
            f"fd_explicit_{integrator_id}",
            family_id,
            f"{dimension} centered finite-difference Laplacian",
            f"{integrator} on the complete reaction-diffusion right-hand side",
            "explicit diffusion and reaction substep limits not stated",
            ("leveque2007finite", "hundsdorfer2003numerical"),
            expected_order,
            ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED),
        )
        for integrator_id, integrator, expected_order in (
            ("euler", "forward Euler", "first order in time and second order in space, nominal"),
            ("rk2", "RK2", "second order, nominal"),
            ("rk4", "classical RK4", "second order in space and fourth order in time, nominal"),
        )
    )


_REACTION_DIFFUSION1D = (
    *_explicit_rd_methods("reaction_diffusion1d", dimension="periodic second-order"),
    _method(
        "split_strang_fd_rk2_pes",
        "reaction_diffusion1d",
        "centered finite-difference diffusion and pointwise logistic reaction",
        "Strang split with exact or PES reaction half-steps and RK2 diffusion",
        "internal diffusion step rule not stated",
        ("strang1968construction", "hundsdorfer2003numerical"),
        "second order, nominal",
        ambiguity_flags=(
            EXPECTED_ORDER_INFERRED,
            CFL_NOT_STATED,
            "pes_reaction_update_not_defined",
        ),
    ),
    _method(
        "spectral_strang_exact_reaction",
        "reaction_diffusion1d",
        "Fourier spectral diffusion and pointwise logistic reaction",
        "Strang split with exact reaction and exact spectral diffusion",
        "exact split subflows",
        ("strang1968construction", "trefethen2000spectral"),
        "second order in splitting and spectral in space, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED,),
    ),
    _method(
        "spectral_imex_exact_diffusion",
        "reaction_diffusion1d",
        "Fourier spectral diffusion and pointwise logistic reaction",
        "RK4 reaction with exact spectral diffusion",
        "coupling order and internal step rule not stated",
        ("hundsdorfer2003numerical", "trefethen2000spectral"),
        "not stated",
        ambiguity_flags=(
            EXPECTED_ORDER_NOT_STATED,
            CFL_NOT_STATED,
            "reaction_diffusion_coupling_not_defined",
        ),
    ),
)


_DIFFUSION_SORPTION1D = (
    _method(
        "euler_fixed",
        "diffusion_sorption1d",
        "lagged-mobility finite-difference Laplacian with mixed boundaries",
        "forward Euler",
        "fixed step; stability limit not stated",
        ("leveque2007finite",),
        "first order in time and second order in space, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED),
    ),
    _method(
        "heun_fixed",
        "diffusion_sorption1d",
        "lagged-mobility finite-difference Laplacian with mixed boundaries",
        "Heun RK2",
        "fixed step; stability limit not stated",
        ("hundsdorfer2003numerical",),
        "second order, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED),
    ),
    _method(
        "rk4_fixed",
        "diffusion_sorption1d",
        "lagged-mobility finite-difference Laplacian with mixed boundaries",
        "classical RK4",
        "fixed step; stability limit not stated",
        ("hundsdorfer2003numerical",),
        "second order in space and fourth order in time, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED),
    ),
    _method(
        "backward_euler_lagged",
        "diffusion_sorption1d",
        "lagged-linearized mixed-boundary tridiagonal system",
        "backward Euler",
        "implicit solve; tolerance not stated",
        ("hundsdorfer2003numerical",),
        "first order in time and second order in space, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, TOLERANCE_NOT_STATED),
    ),
    _method(
        "crank_nicolson_lagged",
        "diffusion_sorption1d",
        "lagged-linearized mixed-boundary tridiagonal system",
        "Crank-Nicolson",
        "implicit solve; tolerance not stated",
        ("hundsdorfer2003numerical",),
        "second order, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, TOLERANCE_NOT_STATED),
    ),
    _method(
        "backward_euler_newton_tridiag",
        "diffusion_sorption1d",
        "nonlinear mixed-boundary tridiagonal Jacobian",
        "Newton backward Euler",
        "Newton solve; tolerances and maximum iterations not stated",
        ("hundsdorfer2003numerical",),
        "first order in time and second order in space, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, TOLERANCE_NOT_STATED),
    ),
    _method(
        "crank_nicolson_newton_tridiag",
        "diffusion_sorption1d",
        "nonlinear mixed-boundary tridiagonal Jacobian",
        "Newton Crank-Nicolson",
        "Newton solve; tolerances and maximum iterations not stated",
        ("hundsdorfer2003numerical",),
        "second order, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, TOLERANCE_NOT_STATED),
    ),
    _method(
        "mol_bdf_solve_ivp_mixedbc",
        "diffusion_sorption1d",
        "mixed-boundary method of lines with tridiagonal sparsity",
        "adaptive SciPy solve_ivp BDF",
        "adaptive BDF; tolerances not stated",
        ("virtanen2020scipy",),
        "variable order; exact order controls not stated",
        ambiguity_flags=(EXPECTED_ORDER_NOT_STATED, TOLERANCE_NOT_STATED),
    ),
)


_REACTION_DIFFUSION2D = (
    *_explicit_rd_methods("reaction_diffusion2d", dimension="five-point Neumann edge-padded"),
    _method(
        "strang_rk4_reaction_cn_sparse",
        "reaction_diffusion2d",
        "sparse five-point Neumann Laplacian with pointwise reaction",
        "Strang split with RK4 reaction and Crank-Nicolson diffusion",
        "sparse implicit solve; tolerances not stated",
        ("strang1968construction", "hundsdorfer2003numerical"),
        "second order, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, TOLERANCE_NOT_STATED),
    ),
    _method(
        "dct_neumann_strang_split",
        "reaction_diffusion2d",
        "DCT-diagonalized Neumann Laplacian with pointwise reaction",
        "Strang split with RK4 reaction and exact DCT diffusion",
        "exact diagonal diffusion; DCT convention not stated",
        ("strang1968construction", "trefethen2000spectral"),
        "second order in splitting, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, "dct_type_and_normalization_not_stated"),
    ),
    _method(
        "dct_neumann_imex_rk2",
        "reaction_diffusion2d",
        "DCT-diagonalized Neumann Laplacian with pointwise reaction",
        "RK2 reaction with exact DCT diffusion",
        "coupling and DCT conventions not stated",
        ("hundsdorfer2003numerical", "trefethen2000spectral"),
        "not stated",
        ambiguity_flags=(
            EXPECTED_ORDER_NOT_STATED,
            "dct_type_and_normalization_not_stated",
            "reaction_diffusion_coupling_not_defined",
        ),
    ),
)


def _darcy_method(
    codename: str,
    linear_solver: str,
    citation_keys: tuple[str, ...],
    *,
    fallback: str | None = None,
    ambiguity_flags: tuple[str, ...] = (),
) -> SolverMethod:
    return _method(
        codename,
        "darcy2d",
        "five-point cell-centered finite volume with arithmetic face permeability",
        linear_solver,
        "iteration and residual tolerances not stated",
        citation_keys,
        "second order for the converged spatial operator; reported by Darcy MMS",
        fallback=fallback,
        ambiguity_flags=(TOLERANCE_NOT_STATED, *ambiguity_flags),
    )


_DARCY2D = (
    _darcy_method(
        "steady_state_cg_fv_arithmetic",
        "conjugate gradient",
        ("eymard2000finite", "hestenes1952methods"),
    ),
    _darcy_method(
        "steady_state_jacobi_fv_arithmetic",
        "Jacobi relaxation",
        ("eymard2000finite", "saad2003iterative"),
    ),
    _darcy_method(
        "steady_state_weighted_jacobi_fv_arithmetic",
        "weighted Jacobi with omega=0.75",
        ("eymard2000finite", "saad2003iterative"),
    ),
    _darcy_method(
        "steady_state_redblack_gauss_seidel_fv_arithmetic",
        "red-black Gauss-Seidel",
        ("eymard2000finite", "saad2003iterative"),
    ),
    _darcy_method(
        "steady_state_sor_fv_arithmetic",
        "successive over-relaxation with omega=1.7",
        ("eymard2000finite", "saad2003iterative"),
    ),
    _darcy_method(
        "steady_state_pcg_jacobi_fv_arithmetic",
        "Jacobi-preconditioned conjugate gradient",
        ("eymard2000finite", "hestenes1952methods"),
    ),
    _darcy_method(
        "geometric_multigrid_vcycle_fv",
        "Jacobi-preconditioned conjugate gradient",
        ("eymard2000finite", "hestenes1952methods", "saad2003iterative", "virtanen2020scipy"),
        fallback="sparse-direct solve",
        ambiguity_flags=(
            "codename_says_multigrid_but_paper_description_says_jacobi_pcg",
            "fallback_trigger_not_stated",
        ),
    ),
    _darcy_method(
        "pcg_ilu_fv",
        "ILU-preconditioned conjugate gradient",
        ("eymard2000finite", "hestenes1952methods", "saad2003iterative", "virtanen2020scipy"),
        fallback="sparse-direct solve",
        ambiguity_flags=("fallback_trigger_not_stated",),
    ),
)


def _ns_method(
    codename: str,
    scheme: str,
    integrator: str,
    citation_keys: tuple[str, ...],
    expected_order: str,
    *,
    ambiguity_flags: tuple[str, ...] = (),
) -> SolverMethod:
    return _method(
        codename,
        "incompressible_ns2d",
        scheme,
        integrator,
        "projection solve and time-step tolerances not stated",
        citation_keys,
        expected_order,
        ambiguity_flags=(
            EXPECTED_ORDER_INFERRED,
            CFL_NOT_STATED,
            TOLERANCE_NOT_STATED,
            *ambiguity_flags,
        ),
    )


_INCOMPRESSIBLE_NS2D = (
    _ns_method(
        "chorin_centered_euler",
        "centered finite-difference advection and diffusion",
        "forward Euler Chorin projection with Jacobi pressure solve",
        ("chorin1967numerical", "ferziger2002computational"),
        "first order in time and second order in space, nominal",
    ),
    _ns_method(
        "chorin_centered_rk2",
        "centered finite-difference advection and diffusion",
        "RK2 Chorin projection with Jacobi pressure solve",
        ("chorin1967numerical", "ferziger2002computational"),
        "second order, nominal",
    ),
    _ns_method(
        "chorin_skewsym_euler",
        "skew-symmetric finite-difference advection with centered diffusion",
        "forward Euler Chorin projection with Jacobi pressure solve",
        ("chorin1967numerical", "ferziger2002computational"),
        "first order in time and second order in space, nominal",
    ),
    _ns_method(
        "semiimplicit_centered_euler",
        "centered advection with semi-implicit diffusion",
        "IMEX Euler projection with Jacobi pressure solve",
        ("chorin1967numerical", "ascher1997implicit", "ferziger2002computational"),
        "first order in time and second order in space, nominal",
    ),
    _ns_method(
        "imex_cnab2_skewsym",
        "skew-symmetric advection with Crank-Nicolson diffusion",
        "CNAB2 projection with Jacobi pressure solve",
        ("ascher1997implicit", "ferziger2002computational"),
        "second order, nominal",
    ),
    _ns_method(
        "dst_projection_cnab2_runtime",
        "centered or skew-symmetric projection form",
        "CNAB2 projection with DST Poisson and Helmholtz solves",
        (
            "chorin1967numerical",
            "ascher1997implicit",
            "strang1986introduction",
            "virtanen2020scipy",
        ),
        "second order, nominal",
        ambiguity_flags=("advection_form_not_selected_by_paper", "dst_convention_not_stated"),
    ),
    _ns_method(
        "imex_cnab2_skewsym_spectral",
        "skew-symmetric advection with Crank-Nicolson diffusion",
        "CNAB2 projection with DST Poisson and Helmholtz solves",
        ("ascher1997implicit", "strang1986introduction"),
        "second order, nominal",
        ambiguity_flags=("dst_convention_not_stated",),
    ),
    _ns_method(
        "semiimplicit_centered_spectral",
        "centered advection with semi-implicit diffusion",
        "IMEX Euler projection with DST solves",
        ("ascher1997implicit", "strang1986introduction"),
        "first order in time and second order in space, nominal",
        ambiguity_flags=("dst_convention_not_stated",),
    ),
    _ns_method(
        "mac_projection_centered",
        "centered advection and semi-implicit diffusion",
        "IMEX Euler projection with DST Poisson and Helmholtz solves",
        (
            "chorin1967numerical",
            "ascher1997implicit",
            "strang1986introduction",
            "virtanen2020scipy",
        ),
        "first order in time and second order in space, nominal",
        ambiguity_flags=(
            "paper_calls_method_a_dst_projection_alias_not_a_defined_mac_grid",
            "dst_convention_not_stated",
        ),
    ),
    _ns_method(
        "mac_projection_rk2",
        "skew-symmetric advection with Crank-Nicolson diffusion",
        "CNAB2 projection with DST Poisson and Helmholtz solves",
        ("ascher1997implicit", "strang1986introduction", "virtanen2020scipy"),
        "second order, nominal",
        ambiguity_flags=(
            "paper_calls_method_a_dst_projection_alias_not_a_defined_mac_grid",
            "dst_convention_not_stated",
        ),
    ),
)


def _shallow_water_method(
    codename: str,
    flux: str,
    integrator: str,
    citation_keys: tuple[str, ...],
    *,
    ambiguity_flags: tuple[str, ...] = (),
) -> SolverMethod:
    return _method(
        codename,
        "shallow_water2d",
        f"{flux} flux with MC-limited MUSCL reconstruction",
        integrator,
        "CFL, positivity floor, and wet-dry handling not stated",
        citation_keys,
        "second order in space, nominal",
        ambiguity_flags=(EXPECTED_ORDER_INFERRED, CFL_NOT_STATED, *ambiguity_flags),
    )


_SHALLOW_WATER2D = (
    _shallow_water_method(
        "fv_muscl_hll_ssprk3",
        "HLL",
        "SSPRK3",
        ("toro2013riemann", "van1979towards", "gottlieb2001strong"),
    ),
    _shallow_water_method(
        "fv_muscl_rusanov_ssprk2",
        "Rusanov",
        "SSPRK2",
        ("toro2013riemann", "van1979towards", "gottlieb2001strong"),
    ),
    _shallow_water_method(
        "fv_muscl_rusanov_ssprk3",
        "Rusanov",
        "SSPRK3",
        ("toro2013riemann", "van1979towards", "gottlieb2001strong"),
    ),
    _shallow_water_method(
        "kt_central_upwind_muscl_ssprk3",
        "Kurganov-Tadmor central-upwind",
        "SSPRK3",
        ("kurganov2000new", "van1979towards", "gottlieb2001strong"),
    ),
    _shallow_water_method(
        "weno5_rusanov_ssprk3_pos",
        "Rusanov",
        "SSPRK3",
        ("toro2013riemann", "van1979towards", "gottlieb2001strong"),
        ambiguity_flags=(
            "codename_says_weno5_but_paper_description_says_muscl_mc",
            "paper_table_omits_weno_citation",
        ),
    ),
    _shallow_water_method(
        "roe_entropy_fix_muscl_ssprk3",
        "Roe with entropy fix",
        "SSPRK3",
        ("roe1981approximate", "toro2013riemann", "van1979towards", "gottlieb2001strong"),
        ambiguity_flags=("entropy_fix_parameter_not_stated",),
    ),
    _shallow_water_method(
        "hllc_muscl_ssprk3_pos",
        "HLLC",
        "SSPRK3",
        ("toro2013riemann", "van1979towards", "gottlieb2001strong"),
    ),
)


SOLVER_METHODS: tuple[SolverMethod, ...] = (
    *_ADVECTION1D,
    *_BURGERS1D,
    *_REACTION_DIFFUSION1D,
    *_DIFFUSION_SORPTION1D,
    *_REACTION_DIFFUSION2D,
    *_DARCY2D,
    *_INCOMPRESSIBLE_NS2D,
    *_SHALLOW_WATER2D,
)

PAPER_FAMILY_COUNTS = MappingProxyType(
    {
        "advection1d": 8,
        "burgers1d": 18,
        "reaction_diffusion1d": 6,
        "diffusion_sorption1d": 8,
        "reaction_diffusion2d": 6,
        "darcy2d": 8,
        "incompressible_ns2d": 10,
        "shallow_water2d": 7,
    }
)

_BY_ID = MappingProxyType({method.method_id: method for method in SOLVER_METHODS})


def all_solver_methods() -> tuple[SolverMethod, ...]:
    return SOLVER_METHODS


def methods_for_family(family_id: str) -> tuple[SolverMethod, ...]:
    if family_id not in PAPER_FAMILY_COUNTS:
        raise KeyError(f"unknown solver-bank PDE family: {family_id}")
    return tuple(method for method in SOLVER_METHODS if method.family_id == family_id)


def get_solver_method(method_id: str) -> SolverMethod:
    try:
        return _BY_ID[method_id]
    except KeyError as exc:
        raise KeyError(f"unknown solver-bank method: {method_id}") from exc


def validate_catalog() -> None:
    if len(SOLVER_METHODS) != 71:
        raise ValueError(f"paper solver bank must contain 71 entries, got {len(SOLVER_METHODS)}")
    if len(_BY_ID) != len(SOLVER_METHODS):
        raise ValueError("family-scoped solver method IDs must be unique")
    observed = {family_id: len(methods_for_family(family_id)) for family_id in PAPER_FAMILY_COUNTS}
    if observed != dict(PAPER_FAMILY_COUNTS):
        raise ValueError(f"solver-bank family counts do not match the paper: {observed}")


validate_catalog()

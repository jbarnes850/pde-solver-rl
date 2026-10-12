"""Clean-room renderers for the eight paper-listed Darcy solver-bank methods.

The paper fixes the cell-centered finite-volume operator and arithmetic face averages. It
does not release code, convergence tolerances, iteration limits, or fallback triggers. The
rendered programs therefore use reviewable local choices. The method called
``geometric_multigrid_vcycle_fv`` follows the *table description* (Jacobi-PCG plus direct
fallback); it is not presented as a geometric multigrid implementation.
"""

from __future__ import annotations

from textwrap import dedent

METHODS = (
    "steady_state_cg_fv_arithmetic",
    "steady_state_jacobi_fv_arithmetic",
    "steady_state_weighted_jacobi_fv_arithmetic",
    "steady_state_redblack_gauss_seidel_fv_arithmetic",
    "steady_state_sor_fv_arithmetic",
    "steady_state_pcg_jacobi_fv_arithmetic",
    "geometric_multigrid_vcycle_fv",
    "pcg_ilu_fv",
)

AMBIGUITIES = {
    "all": (
        "The author implementation, tolerance, iteration limit, and exact boundary-face "
        "assembly are not public."
    ),
    "geometric_multigrid_vcycle_fv": (
        "The codename says geometric multigrid, but the paper table says Jacobi-PCG with "
        "a sparse-direct fallback. This renderer follows the table description."
    ),
    "pcg_ilu_fv": "The paper does not state the sparse-direct fallback trigger.",
}


_SOLVE_BODIES = {
    "steady_state_cg_fv_arithmetic": """
        x, info = cg(A, rhs, rtol=1e-10, atol=0.0, maxiter=max(2000, 10 * rhs.size))
        if info != 0:
            raise RuntimeError("CG did not converge")
        return x
    """,
    "steady_state_pcg_jacobi_fv_arithmetic": """
        inv_diag = 1.0 / A.diagonal()
        M = LinearOperator(A.shape, matvec=lambda x: inv_diag * x)
        x, info = cg(A, rhs, M=M, rtol=1e-10, atol=0.0,
                     maxiter=max(2000, 10 * rhs.size))
        if info != 0:
            raise RuntimeError("Jacobi-PCG did not converge")
        return x
    """,
    "geometric_multigrid_vcycle_fv": """
        # Source-derived implementation of the paper table description, not the codename:
        # Jacobi-preconditioned CG with a sparse-direct fallback.
        inv_diag = 1.0 / A.diagonal()
        M = LinearOperator(A.shape, matvec=lambda x: inv_diag * x)
        x, info = cg(A, rhs, M=M, rtol=1e-10, atol=0.0,
                     maxiter=max(2000, 10 * rhs.size))
        return spsolve(A, rhs) if info != 0 else x
    """,
    "pcg_ilu_fv": """
        try:
            ilu = spilu(A.tocsc(), drop_tol=1e-6, fill_factor=12.0)
            M = LinearOperator(A.shape, matvec=ilu.solve)
            x, info = cg(A, rhs, M=M, rtol=1e-10, atol=0.0,
                         maxiter=max(2000, 10 * rhs.size))
        except Exception:
            info = 1
        return spsolve(A, rhs) if info != 0 else x
    """,
    "steady_state_jacobi_fv_arithmetic": """
        diagonal = A.diagonal()
        remainder = A - diags(diagonal)
        x = np.zeros_like(rhs)
        for _ in range(max(20000, 20 * rhs.size)):
            new = (rhs - remainder @ x) / diagonal
            if np.linalg.norm(new - x, ord=np.inf) <= 1e-10 * (1.0 + np.linalg.norm(new, ord=np.inf)):
                return new
            x = new
        raise RuntimeError("Jacobi relaxation did not converge")
    """,
    "steady_state_weighted_jacobi_fv_arithmetic": """
        diagonal = A.diagonal()
        remainder = A - diags(diagonal)
        x = np.zeros_like(rhs)
        for _ in range(max(20000, 20 * rhs.size)):
            proposal = (rhs - remainder @ x) / diagonal
            new = 0.25 * x + 0.75 * proposal
            if np.linalg.norm(new - x, ord=np.inf) <= 1e-10 * (1.0 + np.linalg.norm(new, ord=np.inf)):
                return new
            x = new
        raise RuntimeError("weighted Jacobi relaxation did not converge")
    """,
    "steady_state_redblack_gauss_seidel_fv_arithmetic": """
        x = np.zeros_like(rhs)
        diagonal = A.diagonal()
        remainder = A - diags(diagonal)
        rows = np.arange(rhs.size)
        colors = ((rows // ny) + (rows % ny)) % 2
        for _ in range(max(12000, 12 * rhs.size)):
            old = x.copy()
            for color in (0, 1):
                proposal = (rhs - remainder @ x) / diagonal
                selected = colors == color
                x[selected] = proposal[selected]
            if np.linalg.norm(x - old, ord=np.inf) <= 1e-10 * (1.0 + np.linalg.norm(x, ord=np.inf)):
                return x
        raise RuntimeError("red-black Gauss-Seidel did not converge")
    """,
    "steady_state_sor_fv_arithmetic": """
        x = np.zeros_like(rhs)
        diagonal = A.diagonal()
        omega = 1.7
        for _ in range(max(12000, 12 * rhs.size)):
            old = x.copy()
            for row in range(rhs.size):
                start, stop = A.indptr[row], A.indptr[row + 1]
                cols, vals = A.indices[start:stop], A.data[start:stop]
                off = np.dot(vals[cols != row], x[cols[cols != row]])
                proposal = (rhs[row] - off) / diagonal[row]
                x[row] += omega * (proposal - x[row])
            if np.linalg.norm(x - old, ord=np.inf) <= 1e-10 * (1.0 + np.linalg.norm(x, ord=np.inf)):
                return x
        raise RuntimeError("SOR did not converge")
    """,
}


_TEMPLATE = """
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg, spilu, spsolve


def _assemble(a):
    # Cell centers are (i+1/2)/N. A boundary face is half a cell from u=0.
    nx, ny = a.shape
    dx, dy = 1.0 / nx, 1.0 / ny
    rows, cols, data = [], [], []
    for i in range(nx):
        for j in range(ny):
            row = i * ny + j
            diagonal = 0.0
            for di, dj, spacing in ((-1, 0, dx), (1, 0, dx), (0, -1, dy), (0, 1, dy)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    face = 0.5 * (a[i, j] + a[ii, jj])
                    coefficient = face / (spacing * spacing)
                    rows.append(row); cols.append(ii * ny + jj); data.append(-coefficient)
                    diagonal += coefficient
                else:
                    # Zero Dirichlet data at the physical face, half a cell away.
                    diagonal += 2.0 * a[i, j] / (spacing * spacing)
            rows.append(row); cols.append(row); data.append(diagonal)
    return coo_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()


def _linear_solve(A, rhs, ny):
__SOLVE_BODY__


def solver(diffusion_batch, beta):
    a_batch = np.asarray(diffusion_batch, dtype=np.float64)
    if a_batch.ndim != 3 or not np.all(np.isfinite(a_batch)) or np.any(a_batch <= 0.0):
        raise ValueError("diffusion_batch must be finite, positive, and have shape [B,Nx,Ny]")
    result = np.empty_like(a_batch)
    for batch_index, a in enumerate(a_batch):
        nx, ny = a.shape
        A = _assemble(a)
        rhs = np.full(nx * ny, float(beta), dtype=np.float64)
        result[batch_index] = _linear_solve(A, rhs, ny).reshape(nx, ny)
    return result
"""


def render(codename: str) -> str:
    """Return standalone solver code for one exact paper codename."""

    try:
        solve_body = dedent(_SOLVE_BODIES[codename]).strip("\n")
    except KeyError as exc:
        raise KeyError(f"unknown darcy2d solver-bank codename: {codename}") from exc
    indented = "\n".join(f"    {line}" if line else "" for line in solve_body.splitlines())
    return dedent(_TEMPLATE).replace("__SOLVE_BODY__", indented).lstrip()


__all__ = ["AMBIGUITIES", "METHODS", "render"]

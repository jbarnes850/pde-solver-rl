"""Render clean-room solver-bank programs for two-species reaction--diffusion.

The paper fixes the six method names and their high-level numerical schemes. It does not
publish the solver implementations, the internal step rule, the DCT type, or the precise
coupling used by ``dct_neumann_imex_rk2``. The rendered programs therefore make these local
choices explicit: a maximum internal step of 0.02, an endpoint-grid edge-padded Neumann
Laplacian on ``[-1, 1]^2``, orthonormal DCT-II transforms, and an integrating-factor
explicit-midpoint update for the otherwise unspecified IMEX coupling.
"""

from __future__ import annotations

from textwrap import dedent

_COMMON = dedent(
    """\
    import numpy as np


    # Source-derived clean-room reconstruction of the RLVP solver-bank entry.
    # The paper does not state the internal step rule. We use max(dt) = 0.02.
    _MAX_STEP = 0.02


    def _validate(uv0_batch, t_coordinate):
        state = np.asarray(uv0_batch, dtype=np.float64)
        times = np.asarray(t_coordinate, dtype=np.float64)
        if state.ndim != 4 or state.shape[-1] != 2:
            raise ValueError("uv0_batch must have shape [B, Nx, Ny, 2]")
        if times.ndim != 1 or times.size == 0:
            raise ValueError("t_coordinate must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(times)):
            raise ValueError("inputs must be finite")
        if np.any(np.diff(times) < 0.0):
            raise ValueError("t_coordinate must be nondecreasing")
        return state.copy(), times


    def _reaction(state, k):
        u = state[..., 0]
        v = state[..., 1]
        return np.stack((u - u * u * u - k - v, u - v), axis=-1)


    def _rk2(state, step, rhs):
        first = rhs(state)
        return state + 0.5 * step * (first + rhs(state + step * first))


    def _rk4(state, step, rhs):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * step * k1)
        k3 = rhs(state + 0.5 * step * k2)
        k4 = rhs(state + step * k3)
        return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


    def _integrate_requested(state, times, advance):
        output = np.empty((state.shape[0], times.size, *state.shape[1:]), dtype=np.float64)
        output[:, 0] = state
        now = float(times[0])
        for output_index, target_value in enumerate(times[1:], start=1):
            target = float(target_value)
            while now < target - 1.0e-14:
                step = min(_MAX_STEP, target - now)
                state = advance(state, step)
                now += step
            output[:, output_index] = state
        return output
    """
)


_FINITE_DIFFERENCE = dedent(
    """\

    def _laplacian(values, dx, dy):
        # Edge padding is the paper-specified homogeneous-Neumann treatment.
        padded = np.pad(values, ((0, 0), (1, 1), (1, 1)), mode="edge")
        center = padded[:, 1:-1, 1:-1]
        second_x = (padded[:, 2:, 1:-1] - 2.0 * center + padded[:, :-2, 1:-1]) / (dx * dx)
        second_y = (padded[:, 1:-1, 2:] - 2.0 * center + padded[:, 1:-1, :-2]) / (dy * dy)
        return second_x + second_y


    def _full_rhs(state, Du, Dv, k, dx, dy):
        result = _reaction(state, k)
        result[..., 0] += Du * _laplacian(state[..., 0], dx, dy)
        result[..., 1] += Dv * _laplacian(state[..., 1], dx, dy)
        return result
    """
)


def _explicit(integrator: str) -> str:
    if integrator == "euler":
        update = "value + step * rhs(value)"
    elif integrator == "rk2":
        update = "_rk2(value, step, rhs)"
    elif integrator == "rk4":
        update = "_rk4(value, step, rhs)"
    else:  # pragma: no cover - private construction guard
        raise ValueError(f"unknown explicit integrator: {integrator}")
    return (
        _COMMON
        + _FINITE_DIFFERENCE
        + dedent(
            f'''\

        def solver(uv0_batch, t_coordinate, Du, Dv, k):
            """Solve the paper's two-species system with explicit {integrator.upper()}."""
            state, times = _validate(uv0_batch, t_coordinate)
            nx, ny = state.shape[1:3]
            dx = 2.0 / (nx - 1)
            dy = 2.0 / (ny - 1)
            Du = float(Du)
            Dv = float(Dv)
            k = float(k)
            rhs = lambda value: _full_rhs(value, Du, Dv, k, dx, dy)

            def advance(value, step):
                return {update}

            return _integrate_requested(state, times, advance)
        '''
        )
    )


_SPARSE_CN = _COMMON + dedent(
    '''\
    from scipy.sparse import eye, kron, lil_matrix
    from scipy.sparse.linalg import factorized


    def _neumann_1d(size, spacing):
        operator = lil_matrix((size, size), dtype=np.float64)
        for index in range(size):
            operator[index, index] = -2.0
            if index > 0:
                operator[index, index - 1] = 1.0
            if index + 1 < size:
                operator[index, index + 1] = 1.0
        # Edge padding replaces the missing neighbour with the boundary value.
        operator[0, 0] = -1.0
        operator[-1, -1] = -1.0
        return operator.tocsr() / (spacing * spacing)


    def solver(uv0_batch, t_coordinate, Du, Dv, k):
        """Use RK4 reaction half-steps and sparse Crank--Nicolson diffusion."""
        state, times = _validate(uv0_batch, t_coordinate)
        nx, ny = state.shape[1:3]
        dx = 2.0 / (nx - 1)
        dy = 2.0 / (ny - 1)
        Du = float(Du)
        Dv = float(Dv)
        k = float(k)
        identity_x = eye(nx, format="csr")
        identity_y = eye(ny, format="csr")
        laplacian = (
            kron(_neumann_1d(nx, dx), identity_y)
            + kron(identity_x, _neumann_1d(ny, dy))
        ).tocsr()
        identity = eye(nx * ny, format="csc")
        cache = {}

        def diffuse(values, coefficient, step):
            key = (coefficient, step)
            if key not in cache:
                left = (identity - 0.5 * step * coefficient * laplacian).tocsc()
                right = (identity + 0.5 * step * coefficient * laplacian).tocsr()
                cache[key] = (factorized(left), right)
            solve, right = cache[key]
            flat = values.reshape(values.shape[0], nx * ny)
            return np.stack([solve(right @ row) for row in flat], axis=0).reshape(values.shape)

        reaction_rhs = lambda value: _reaction(value, k)

        def advance(value, step):
            value = _rk4(value, 0.5 * step, reaction_rhs)
            u = diffuse(value[..., 0], Du, step)
            v = diffuse(value[..., 1], Dv, step)
            value = np.stack((u, v), axis=-1)
            return _rk4(value, 0.5 * step, reaction_rhs)

        return _integrate_requested(state, times, advance)
    '''
)


_DCT_COMMON = _COMMON + dedent(
    """\
    from scipy.fft import dctn, idctn


    def _diffusion_eigenvalues(nx, ny, dx, dy):
        # Local declared convention: orthonormal DCT-II for the edge-padded stencil.
        mode_x = np.arange(nx, dtype=np.float64)
        mode_y = np.arange(ny, dtype=np.float64)
        lambda_x = -4.0 * np.sin(0.5 * np.pi * mode_x / nx) ** 2 / (dx * dx)
        lambda_y = -4.0 * np.sin(0.5 * np.pi * mode_y / ny) ** 2 / (dy * dy)
        return lambda_x[:, None] + lambda_y[None, :]


    def _make_diffuser(nx, ny, Du, Dv):
        eigenvalues = _diffusion_eigenvalues(nx, ny, 2.0 / (nx - 1), 2.0 / (ny - 1))

        def diffuse(values, coefficient, step):
            modes = dctn(values, type=2, norm="ortho", axes=(-2, -1))
            modes *= np.exp(coefficient * step * eigenvalues)[None, :, :]
            return idctn(modes, type=2, norm="ortho", axes=(-2, -1))

        return diffuse
    """
)


_DCT_STRANG = _DCT_COMMON + dedent(
    '''\

    def solver(uv0_batch, t_coordinate, Du, Dv, k):
        """Use RK4 reaction half-steps and exact discrete DCT diffusion."""
        state, times = _validate(uv0_batch, t_coordinate)
        nx, ny = state.shape[1:3]
        Du = float(Du)
        Dv = float(Dv)
        k = float(k)
        diffuse = _make_diffuser(nx, ny, Du, Dv)
        reaction_rhs = lambda value: _reaction(value, k)

        def advance(value, step):
            value = _rk4(value, 0.5 * step, reaction_rhs)
            u = diffuse(value[..., 0], Du, step)
            v = diffuse(value[..., 1], Dv, step)
            value = np.stack((u, v), axis=-1)
            return _rk4(value, 0.5 * step, reaction_rhs)

        return _integrate_requested(state, times, advance)
    '''
)


_DCT_IMEX_RK2 = _DCT_COMMON + dedent(
    '''\

    def solver(uv0_batch, t_coordinate, Du, Dv, k):
        """Use a source-derived integrating-factor explicit-midpoint update."""
        state, times = _validate(uv0_batch, t_coordinate)
        nx, ny = state.shape[1:3]
        Du = float(Du)
        Dv = float(Dv)
        k = float(k)
        diffuse = _make_diffuser(nx, ny, Du, Dv)

        def diffuse_state(value, step):
            return np.stack(
                (diffuse(value[..., 0], Du, step), diffuse(value[..., 1], Dv, step)),
                axis=-1,
            )

        def advance(value, step):
            # Explicit midpoint in integrating-factor variables. The exact authors'
            # reaction/diffusion coupling is not published.
            first = _reaction(value, k)
            midpoint = diffuse_state(value + 0.5 * step * first, 0.5 * step)
            return diffuse_state(value, step) + step * diffuse_state(
                _reaction(midpoint, k), 0.5 * step
            )

        return _integrate_requested(state, times, advance)
    '''
)


_RENDERERS = {
    "fd_explicit_euler": lambda: _explicit("euler"),
    "fd_explicit_rk2": lambda: _explicit("rk2"),
    "fd_explicit_rk4": lambda: _explicit("rk4"),
    "strang_rk4_reaction_cn_sparse": lambda: _SPARSE_CN,
    "dct_neumann_strang_split": lambda: _DCT_STRANG,
    "dct_neumann_imex_rk2": lambda: _DCT_IMEX_RK2,
}


def render(codename: str) -> str:
    """Return one standalone solver source string for ``codename``."""
    try:
        source = _RENDERERS[codename]()
    except KeyError as exc:
        raise KeyError(f"unknown reaction_diffusion2d solver method: {codename}") from exc
    return source.rstrip() + "\n"

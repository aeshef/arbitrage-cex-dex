import time
import numpy as np
import scipy.sparse as sp_s
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded

from .config import (GAS_USED, N_Z_HJB, N_T_HJB, N_X_HJB_XY, N_Y_HJB_XY,
                     N_T_HJB_XY, G_SCAN, T_HORIZON, GAMMA, Q_ETH)


def solve_hjb_1d(params, T_horizon=T_HORIZON, N_z=N_Z_HJB, N_t=N_T_HJB,
                 gamma=GAMMA, Q=Q_ETH, g_lo_hi=None,
                 kappa_override=None, beta_override=None, n_g=None):
    S    = float(params['S0'])
    W    = float(params['W0'])
    sig_c = params['sigma_c'] / np.sqrt(365 * 24)
    sig_d = params['sigma_d'] / np.sqrt(365 * 24)
    rho_cd = params['rho']
    sig_res_h = np.sqrt(max(sig_c ** 2 - 2 * rho_cd * sig_c * sig_d + sig_d ** 2, 1e-20))

    kappa = kappa_override if kappa_override is not None else params['kappa']
    beta_ = beta_override  if beta_override  is not None else params['beta']

    def lam(g):
        return kappa * max(float(g), 1e-9) ** beta_

    u_max  = 0.06
    u_grid = np.linspace(-u_max, u_max, N_z)
    du     = u_grid[1] - u_grid[0]
    dt     = T_horizon / N_t
    r_eff  = 0.0

    nu       = Q * S / W
    slip_rel = params['slip_alpha'] * (Q * S) ** params['slip_beta'] / 1e4 * S / W

    g_lo, g_hi = g_lo_hi if g_lo_hi is not None else (params['g_lo'], params['g_hi'])
    n_g_       = int(n_g) if n_g is not None else 40

    # No CFL cap needed with exact jump update — scan the full admissible range.
    g_scan = np.linspace(g_lo, max(g_hi, g_lo + 1e-9), n_g_)

    D   = sig_res_h ** 2 / 2
    b0  = -sig_res_h ** 2 / 2

    a_c = D / du ** 2 - b0 / (2 * du)
    c_c = D / du ** 2 + b0 / (2 * du)
    b_c = -2 * D / du ** 2 + r_eff

    N = N_z
    # LHS matrix (I - dt/2 · L) in scipy banded form (l=1, u=1):
    #   ab[1, j] = M[j, j]   = 1 - dt/2 · b_c    (diagonal)
    #   ab[0, j] = M[j-1, j] = -dt/2 · c_c       (super-diagonal)
    #   ab[2, j] = M[j+1, j] = -dt/2 · a_c       (sub-diagonal)
    # Verify: row sum = 1 - dt/2 · (a_c + b_c + c_c) = 1 (since L conserves constants).
    ab = np.zeros((3, N))
    ab[1, :]   = 1 - dt / 2 * b_c
    ab[0, 1:]  = -dt / 2 * c_c
    ab[2, :-1] = -dt / 2 * a_c
    ab[1, 0] = 1.0;  ab[1, -1] = 1.0
    ab[0, 1] = 0.0;  ab[2, -2] = 0.0

    def apply_explicit(phi_):
        rhs = phi_.copy()
        rhs[1:-1] += dt / 2 * (a_c * phi_[:-2] + b_c * phi_[1:-1] + c_c * phi_[2:])
        return rhs

    phi          = np.ones(N)
    g_star_store = np.full(N, float(params.get('propose_gwei', g_lo)))

    one_minus_gamma = 1.0 - gamma

    for n in range(N_t):
        # ── Step 1: Jump (exact ODE per cell) ─────────────────────────────
        # For each u_i, find best g maximizing jump term gain
        # jump_gain(g) = λ(g) · (φ_i − target(g)),  target(g) = (1+G̃)^{1-γ}
        # If best jump_gain ≤ 0, controller picks g=0 (no execution): φ unchanged.
        lam_eff    = np.zeros(N)
        target_eff = phi.copy()                       # default: no jump → unchanged
        g_opt      = np.full(N, g_lo)

        for i in range(N):
            spread_frac = abs(np.expm1(u_grid[i]))
            best_R, best_g, best_target = 0.0, g_lo, phi[i]
            for g in g_scan:
                c_gas_rel = GAS_USED * g * 1e-9 * S / W
                G_tilde   = nu * spread_frac - slip_rel - c_gas_rel
                if 1 + G_tilde <= 0:
                    continue
                if gamma != 1.0:
                    target_g = (1 + G_tilde) ** one_minus_gamma
                else:
                    target_g = -np.log(1 + G_tilde)   # log-utility limit
                R = lam(g) * (phi[i] - target_g)      # gain rate from execution
                if R > best_R:
                    best_R, best_g, best_target = R, g, target_g

            if best_R > 0:
                lam_eff[i]    = lam(best_g)
                target_eff[i] = best_target
                g_opt[i]      = best_g

        # Exact ODE: φ_new = target* + (φ − target*) · exp(-λ* dt)
        decay  = np.exp(-lam_eff * dt)
        phi    = target_eff + (phi - target_eff) * decay

        # ── Step 2: Diffusion (Crank-Nicolson on L_u only) ────────────────
        rhs    = apply_explicit(phi)
        rhs[0]  = phi[0]
        rhs[-1] = phi[-1]
        phi    = solve_banded((1, 1), ab, rhs)
        # Tiny floor for numerical safety; no upper clip needed (φ ≤ 1 by construction).
        phi    = np.maximum(phi, 1e-12)

        if n == N_t - 1:
            g_star_store = g_opt

    return u_grid, phi, g_star_store


def _d1_neu(N, h):
    o = 0.5 / h
    Dm = sp_s.lil_matrix((N, N))
    for i in range(1, N - 1):
        Dm[i, i - 1] = -o
        Dm[i, i + 1] = o
    Dm[0, 0] = -1.0 / h;  Dm[0, 1]   = 1.0 / h
    Dm[-1, -1] = 1.0 / h; Dm[-1, -2] = -1.0 / h
    return Dm.tocsc()


def _d2_neu(N, h):
    k = 1.0 / (h * h)
    sub  = k * np.ones(N - 1)
    main = -2.0 * k * np.ones(N)
    m = sp_s.diags([sub, main, sub], [-1, 0, 1], shape=(N, N), format='lil')
    m[0, 0]   = 2.0 * k;  m[0, 1]   = -2.0 * k
    m[-1, -1] = 2.0 * k;  m[-1, -2] = -2.0 * k
    return m.tocsc()


def solve_hjb_2d_xy(params, T_horizon=T_HORIZON,
                    Nx=None, Ny=None, N_t=None,
                    gamma=GAMMA, Q=Q_ETH,
                    n_g=None, g_lo_hi=None,
                    kappa_override=None, beta_override=None,
                    print_every=None):
    Nx  = N_X_HJB_XY  if Nx  is None else int(Nx)
    Ny  = N_Y_HJB_XY  if Ny  is None else int(Ny)
    N_t_ = N_T_HJB_XY if N_t is None else int(N_t)
    n_g_ = G_SCAN      if n_g is None else int(n_g)
    pev  = 50          if print_every is None else int(print_every)

    S      = float(params['S0'])
    W      = float(params['W0'])
    sig_c  = params['sigma_c'] / np.sqrt(365 * 24)
    sig_d  = params['sigma_d'] / np.sqrt(365 * 24)
    rho_cd = params['rho']
    r_eff  = 0.0

    kappa = kappa_override if kappa_override is not None else params['kappa']
    beta_ = beta_override  if beta_override  is not None else params['beta']

    def lam_local(g_):
        return kappa * np.clip(np.asarray(g_, float), 1e-9, None) ** beta_

    dlog = 0.06
    L0   = float(np.log(S))
    xg   = np.linspace(L0 - dlog, L0 + dlog, Nx)
    yg   = np.linspace(L0 - dlog, L0 + dlog, Ny)
    hx   = (xg[-1] - xg[0]) / max(Nx - 1, 1)
    hy   = (yg[-1] - yg[0]) / max(Ny - 1, 1)

    Ix, Iy = sp_s.eye(Nx, format='csc'), sp_s.eye(Ny, format='csc')
    d1x, d1y = _d1_neu(Nx, hx), _d1_neu(Ny, hy)
    d2x, d2y = _d2_neu(Nx, hx), _d2_neu(Ny, hy)

    A = (0.5 * sig_c ** 2 * sp_s.kron(Iy, d2x)
         + 0.5 * sig_d ** 2 * sp_s.kron(d2y, Ix)
         + rho_cd * sig_c * sig_d * sp_s.kron(d1y, d1x)
         + (-0.5 * sig_c ** 2) * sp_s.kron(Iy, d1x)
         + (-0.5 * sig_d ** 2) * sp_s.kron(d1y, Ix)
         + r_eff * sp_s.eye(Nx * Ny, format='csc'))

    n_tot = Nx * Ny
    dt    = T_horizon / N_t_
    M     = sp_s.eye(n_tot, format='csc') - dt * A + 1e-7 * sp_s.eye(n_tot, format='csc')
    fac   = spla.splu(M)

    g_lo, g_hi = g_lo_hi if g_lo_hi is not None else (params['g_lo'], params['g_hi'])
    # Stability cap: λ(g)·dt < 1 for explicit jump term
    if beta_ > 0 and kappa > 0:
        g_stable_2d = (1.0 / (kappa * dt)) ** (1.0 / beta_)
    else:
        g_stable_2d = g_hi
    g_scan_hi_2d = max(min(g_hi, g_stable_2d), g_lo + 1e-9)
    g_scan       = np.linspace(g_lo, g_scan_hi_2d, max(4, n_g_))
    nu           = Q * S / W
    slip_rel     = params['slip_alpha'] * (Q * S) ** params['slip_beta'] / 1e4 * S / W

    phi  = np.ones(n_tot)
    gopt = np.full(n_tot, g_lo)

    for s in range(N_t_):
        Rv = np.zeros(n_tot)
        for k in range(n_tot):
            ix = k % Nx
            iy = k // Nx
            u_k = xg[ix] - yg[iy]
            spread_frac = abs(np.expm1(u_k))
            best_R, best_g = -1e15, g_lo
            for g in g_scan:
                c_gas_rel = GAS_USED * g * 1e-9 * S / W
                G_t = nu * spread_frac - slip_rel - c_gas_rel
                if 1 + G_t <= 0:
                    continue
                util_jump = phi[k] - (1 + G_t) ** (1 - gamma)
                R = float(lam_local(g)) * util_jump
                if R > best_R:
                    best_R, best_g = R, g
            # Controller can choose not to execute (g=0 → R=0)
            Rv[k]   = max(0.0, best_R)
            gopt[k] = best_g if best_R >= 0 else g_lo

        rhs = phi - dt * Rv
        phi = fac.solve(rhs)
        phi = np.clip(phi, 1e-8, 2.0)

    phi_2d    = phi.reshape(Ny, Nx)
    g_star_2d = gopt.reshape(Ny, Nx)
    meta = {'Nx': Nx, 'Ny': Ny, 'N_t': N_t_, 'x_grid': xg, 'y_grid': yg}
    return xg, yg, phi_2d, g_star_2d, meta

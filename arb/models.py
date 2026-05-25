import numpy as np

from .config import GAS_USED


def gas_cost_usd(g_gwei, eth_price, gas_used=GAS_USED):
    return gas_used * g_gwei * 1e-9 * eth_price


def lambda_g(g, kappa, beta):
    return kappa * max(float(g), 1e-9) ** beta


def lambda_g_vec(g_arr, kappa, beta):
    g = np.asarray(g_arr, dtype=float)
    return kappa * np.clip(g, 1e-9, None) ** beta


def expected_delay_s(g, kappa, beta):
    return 3600.0 / max(lambda_g(g, kappa, beta), 1e-9)


def slip_usd(Q, S, slip_alpha, slip_beta):
    return slip_alpha * (Q * S) ** slip_beta / 1e4 * S


def g_closedform(
    K_t, A_t, S, kappa, beta, gamma,
    gas_used=GAS_USED, g_lo=0.0, g_hi=500.0,
):
    B_t = A_t + 0.5 * gamma * K_t
    p_gas = gas_used * 1e-9 * S
    if p_gas <= 0 or B_t <= 0:
        return float(g_lo)
    g_raw = (beta * B_t / (kappa * p_gas)) ** (1.0 / (beta + 1.0))
    return float(np.clip(g_raw, g_lo, g_hi))


def h_minvar(rho, sigma_d, sigma_f):
    return rho * sigma_d / max(sigma_f, 1e-6)


def crra(w, gamma=3.0):
    if np.isscalar(w):
        return w ** (1 - gamma) / (1 - gamma) if w > 0 else -1e12
    w = np.asarray(w, dtype=float)
    out = np.full_like(w, -1e12)
    pos = w > 0
    out[pos] = w[pos] ** (1 - gamma) / (1 - gamma)
    return out


def build_K_t(Q, S, sigma_d_h, rho):
    return (Q * S) ** 2 * sigma_d_h ** 2 * max(1.0 - rho ** 2, 1e-6)


def build_A_t(Q, S, sigma_d_h, slip_alpha, slip_beta):
    return slip_usd(Q, S, slip_alpha, slip_beta) * sigma_d_h


def mean_risk_objective(
    g, K_t, A_t, S, Q, spread_usd, kappa, beta, gamma, gas_used=GAS_USED,
):
    lam = lambda_g(g, kappa, beta)
    return (
        Q * spread_usd
        - gas_cost_usd(g, S, gas_used)
        - A_t / max(lam, 1e-9)
        - 0.5 * gamma * K_t / max(lam, 1e-9)
    )

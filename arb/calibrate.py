import warnings

import numpy as np
import pandas as pd

from .config import (
    Q_ETH,
    ROLL_H,
    SLIP_ALPHA_DEFAULT,
    SLIP_BETA_DEFAULT,
    fallback_regime,
    load_gas_regimes,
)
from .models import slip_usd, h_minvar, build_K_t, build_A_t, g_closedform


def calibrate_lambda_beta(base_fees_gwei, block_time_s=12.0, n_pts=30):
    bf = np.sort(np.asarray(base_fees_gwei, dtype=float))
    if len(bf) < 5:
        return None, None, None
    g_range = np.linspace(bf.min() * 0.9, bf.max() * 1.1, n_pts)
    p_incl = np.array([np.mean(bf <= g) for g in g_range])
    lam = p_incl / block_time_s * 3600.0
    mask = (g_range > 0) & (lam > 0)
    if mask.sum() < 3:
        return None, None, None
    coef = np.polyfit(np.log(g_range[mask]), np.log(lam[mask]), 1)
    beta_ = float(np.clip(coef[0], 0.1, 5.0))
    kappa_ = float(np.exp(coef[1]))
    pred = coef[1] + coef[0] * np.log(g_range[mask])
    ss_res = np.sum((np.log(lam[mask]) - pred) ** 2)
    ss_tot = np.sum((np.log(lam[mask]) - np.log(lam[mask]).mean()) ** 2)
    r2_ = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return kappa_, beta_, r2_


def calibrate_slippage(swaps_df):
    if swaps_df.empty or 'exec_price' not in swaps_df.columns:
        return None, None
    sw = swaps_df.copy()
    sw['abs_exec'] = pd.to_numeric(sw.get('exec_price', pd.Series(dtype=float)), errors='coerce')
    sw['amountUSD'] = pd.to_numeric(sw.get('amountUSD', pd.Series(dtype=float)), errors='coerce')
    sw = sw.dropna(subset=['abs_exec', 'amountUSD'])
    sw = sw[(sw['abs_exec'] > 100) & (sw['abs_exec'] < 1e6) & (sw['amountUSD'] > 0)]
    if len(sw) < 10:
        return None, None
    log_vol = np.log(sw['amountUSD'].values)
    log_slip = np.log(np.abs(sw['abs_exec'].pct_change().fillna(0)).clip(1e-6) * 1e4 + 1e-4)
    if len(log_vol) < 10:
        return None, None
    coef = np.polyfit(log_vol, log_slip, 1)
    return float(np.exp(coef[1])), float(np.clip(coef[0], 0.0, 2.0))


def calibrate_all(
    px, blocks, gas_oracle, swaps=None, Q=Q_ETH, roll_h=ROLL_H,
    slip_alpha=None, slip_beta=None,
):
    params = {}
    ann = np.sqrt(365 * 24)
    params['sigma_d'] = float(px['ret_dex'].std() * ann)
    params['sigma_c'] = float(px['ret_cex'].std() * ann)
    if 'ret_perp' in px.columns and px['ret_perp'].notna().sum() > 50:
        params['sigma_f'] = float(px['ret_perp'].std() * ann)
        params['rho_cf'] = float(px['ret_cex'].corr(px['ret_perp']))
    else:
        params['sigma_f'] = params['sigma_c']
        params['rho_cf'] = 1.0
    params['rho'] = float(px['ret_cex'].corr(px['ret_dex']))

    assert not blocks.empty
    bf_col = 'baseFeePerGas' if 'baseFeePerGas' in blocks.columns else 'baseFee_gwei'
    base_fees = blocks[bf_col].dropna().values
    if 'ts' in blocks.columns:
        block_time_s = float(blocks['ts'].diff().dt.total_seconds().dropna().median())
    elif isinstance(blocks.index, pd.DatetimeIndex):
        block_time_s = float(blocks.index.to_series().diff().dt.total_seconds().dropna().median())
    else:
        block_time_s = 12.0
    block_time_s = max(block_time_s, 1.0)

    kappa, beta, r2 = calibrate_lambda_beta(base_fees, block_time_s=block_time_s)
    if kappa is None:
        _, fb = fallback_regime(load_gas_regimes())
        warnings.warn('lambda calibration failed; using GAS_REGIME_FALLBACK')
        kappa = fb['kappa']
        beta = fb['beta']
    params['kappa'] = float(kappa)
    params['beta'] = float(beta)
    if r2 is not None:
        params['lambda_r2'] = float(r2)

    if swaps is not None and not swaps.empty:
        alpha_slip, beta_slip = calibrate_slippage(swaps)
    else:
        alpha_slip, beta_slip = None, None
    if alpha_slip is None:
        if slip_alpha is not None and slip_beta is not None:
            alpha_slip, beta_slip = float(slip_alpha), float(slip_beta)
        elif SLIP_ALPHA_DEFAULT is not None and SLIP_BETA_DEFAULT is not None:
            warnings.warn('slippage calibration failed; using SLIP_*_DEFAULT')
            alpha_slip, beta_slip = SLIP_ALPHA_DEFAULT, SLIP_BETA_DEFAULT
        else:
            raise ValueError(
                'slippage calibration failed; pass slip_alpha/slip_beta or set '
                'SLIP_ALPHA_DEFAULT and SLIP_BETA_DEFAULT'
            )
    params['slip_alpha'] = float(alpha_slip)
    params['slip_beta'] = float(beta_slip)
    params.update(gas_oracle)

    S0 = float(px['cex_close'].iloc[-1])
    sig_d_h = params['sigma_d'] / np.sqrt(365 * 24)
    params['S0'] = S0
    params['W0'] = Q * S0
    params['Q'] = Q
    params['h_star'] = h_minvar(params['rho'], params['sigma_d'], params['sigma_f'])
    params['sig_res_h'] = S0 * sig_d_h * np.sqrt(max(1.0 - params['rho'] ** 2, 1e-4))
    params['K_ref'] = build_K_t(Q, S0, sig_d_h, params['rho'])
    params['A_ref'] = build_A_t(Q, S0, sig_d_h, alpha_slip, beta_slip)

    _bf = float(gas_oracle.get('base_fee', 2.0))
    params['g_lo'] = max(_bf + 0.05, 0.1)
    params['g_hi'] = max(params.get('fast_gwei', 50.0) * 3.0, params['g_lo'] + 10.0)
    return params


def build_state_df(df, params, Q=None):
    Q = Q or params.get('Q', Q_ETH)
    df = df.copy()
    df['sigma_d_h'] = df['ret_dex'].rolling(ROLL_H).std()
    df['sigma_f_h'] = df['ret_cex'].rolling(ROLL_H).std()
    df['rho_r'] = df['ret_cex'].rolling(ROLL_H).corr(df['ret_dex'])
    df['K_t'] = (Q * df['cex_close']) ** 2 * (
        df['sigma_d_h'] ** 2
        - 2 * df['rho_r'] * df['sigma_d_h'] * df['sigma_f_h']
        + df['sigma_f_h'] ** 2
    ).clip(lower=0)
    sl = params['slip_alpha'] * (Q * df['cex_close']) ** params['slip_beta']
    df['slip_proxy'] = sl / 1e4 * df['cex_close'] * Q
    df['A_t'] = df['slip_proxy'] * df['sigma_d_h'].fillna(0)

    kappa = params['kappa']
    beta = params['beta']
    g_lo = params['g_lo']
    g_hi = params['g_hi']
    gamma = params.get('gamma', 3.0)

    def _g_star(row):
        K = float(row['K_t']) if pd.notna(row['K_t']) else params['K_ref']
        A = float(row['A_t']) if pd.notna(row['A_t']) else params['A_ref']
        return g_closedform(K, A, float(row['cex_close']), kappa, beta, gamma, g_lo=g_lo, g_hi=g_hi)

    df['g_star_cf'] = df.apply(_g_star, axis=1)
    q75 = df['spread_usd'].abs().rolling(24).quantile(0.75)
    df['is_opp'] = df['spread_usd'].abs() > q75
    return df.dropna(subset=['sigma_d_h'])

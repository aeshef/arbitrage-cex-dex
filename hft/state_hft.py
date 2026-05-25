import numpy as np
import pandas as pd

from arb.config import GAMMA, Q_ETH
from arb.models import build_A_t, build_K_t, g_closedform


def build_state_hft(arb_swaps, px_1m, params, Q=Q_ETH, roll_min=60):
    if arb_swaps.empty:
        return pd.DataFrame()

    px = px_1m.copy()
    ann = np.sqrt(525_600)
    px['sigma_d_1m'] = px['ret_dex'].rolling(roll_min).std()
    px['rho_r'] = px['ret_cex'].rolling(roll_min).corr(px['ret_dex'])
    px['sigma_d_h'] = px['ret_dex'].rolling(roll_min).std() * ann / np.sqrt(365 * 24)

    df = arb_swaps.copy()
    df['dt_1m'] = df['dt'].dt.floor('1min')
    df = df.merge(px[['sigma_d_h', 'rho_r']].dropna(), left_on='dt_1m', right_index=True, how='left')
    sig_d_h_global = params['sigma_d'] / np.sqrt(365 * 24)
    df['sigma_d_h'] = df['sigma_d_h'].fillna(sig_d_h_global)
    df['rho_r'] = df['rho_r'].fillna(params['rho'])

    kappa = params['kappa']
    beta = params['beta']
    g_lo = params['g_lo']
    g_hi = params['g_hi']

    def _state_row(r):
        S = float(r['cex_price'])
        K = build_K_t(Q, S, float(r['sigma_d_h']), float(r['rho_r']))
        A = build_A_t(Q, S, float(r['sigma_d_h']), params['slip_alpha'], params['slip_beta'])
        g_star = g_closedform(K, A, S, kappa, beta, GAMMA, g_lo=g_lo, g_hi=g_hi)
        return pd.Series({'K_t': K, 'A_t': A, 'g_star_cf': g_star})

    state_cols = df.apply(_state_row, axis=1)
    df = pd.concat([df, state_cols], axis=1)
    df['u_val'] = np.log((df['cex_price'] / df['dex_price'].replace(0, np.nan)).clip(lower=1e-6))
    return df.drop(columns=['dt_1m'], errors='ignore').reset_index(drop=True)


def describe_state_hft(state):
    if state.empty:
        return pd.DataFrame()
    if 'abs_spread_usd' not in state.columns:
        state = state.copy()
        state['abs_spread_usd'] = state['spread_usd'].abs()

    rows = []

    def _row(name, col):
        s = state[col].dropna()
        if s.empty:
            return
        rows.append({
            'Variable': name,
            'N': len(s),
            'Mean': round(float(s.mean()), 4),
            'Std': round(float(s.std()), 4),
            'Min': round(float(s.min()), 4),
            'Median': round(float(s.median()), 4),
            'Max': round(float(s.max()), 4),
        })

    _row('CEX price (USD)', 'cex_price')
    _row('DEX price (USD)', 'dex_price')
    _row('|spread| (USD)', 'abs_spread_usd')
    _row('|spread| (bps)', 'abs_spread_bps')
    _row('size (ETH)', 'amount_eth')
    _row('gas (gwei)', 'gas_price_gwei')
    _row('g* CF (gwei)', 'g_star_cf')
    out = pd.DataFrame(rows)
    return out.set_index('Variable') if not out.empty else out

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from arb.backtest import compute_sharpe
from arb.config import GAMMA, GAS_USED, Q_ETH
from arb.hjb import solve_hjb_1d
from arb.models import build_A_t, build_K_t, g_closedform, slip_usd

from arb.calibrate import calibrate_all

from .calibrate_hft import build_px_hft, gas_oracle_from_blocks, _prepare_swaps_for_slip


def add_model_bids(arb, params, u_hjb=None, g_star_hjb=None, Q=10.0):
    df = arb.copy()
    S = float(params.get('S0', df['cex_price'].median()))
    sig_d_h = params['sigma_d'] / np.sqrt(365 * 24)
    K_ref = build_K_t(Q, S, sig_d_h, params['rho'])
    A_ref = build_A_t(Q, S, sig_d_h, params['slip_alpha'], params['slip_beta'])
    g_naive = float(params.get('propose_gwei', params['g_lo'] * 1.5))

    df['u_val'] = np.log((df['cex_price'] / df['dex_price'].replace(0, np.nan)).clip(lower=1e-6))
    df['g_cf'] = g_closedform(
        K_ref, A_ref, S, params['kappa'], params['beta'], GAMMA,
        g_lo=params['g_lo'], g_hi=params['g_hi'],
    )
    if u_hjb is not None and g_star_hjb is not None:
        u_clip = df['u_val'].clip(lower=float(u_hjb[0]), upper=float(u_hjb[-1]))
        df['g_hjb'] = np.interp(u_clip, u_hjb, g_star_hjb)
    else:
        df['g_hjb'] = np.nan
    df['g_naive'] = g_naive

    actual = df['gas_price_gwei']
    df['inc_cf'] = (df['g_cf'] >= actual).astype(int)
    df['inc_hjb'] = (df['g_hjb'] >= actual).astype(int)
    df['inc_naive'] = (df['g_naive'] >= actual).astype(int)
    return df


def compute_strategy_pnl(df, params, Q=10.0):
    results = {}
    S = float(params.get('S0', df['cex_price'].median()))
    slip = slip_usd(Q, S, params['slip_alpha'], params['slip_beta'])

    for strat, g_col, inc_col in (
        ('Actual', 'gas_price_gwei', None),
        ('CF', 'g_cf', 'inc_cf'),
        ('HJB', 'g_hjb', 'inc_hjb'),
        ('Naive', 'g_naive', 'inc_naive'),
    ):
        if g_col not in df.columns:
            continue
        sub = df if inc_col is None else df[df[inc_col] == 1]
        if sub.empty:
            results[strat] = pd.DataFrame()
            continue
        sub = sub.copy()
        sub['gas_cost_usd_model'] = GAS_USED * sub[g_col] * 1e-9 * sub['cex_price']
        sub['net_pnl_model'] = sub['gross_pnl_usd'] - sub['gas_cost_usd_model'] - slip
        results[strat] = sub[
            ['dt', 'timestamp', 'cex_price', 'dex_price', 'spread_bps', 'amount_eth',
             g_col, 'gas_cost_usd_model', 'net_pnl_model']
        ].rename(columns={g_col: 'g_bid', 'net_pnl_model': 'net_pnl_usd'})

    return results


def summary_table(pnl_dict):
    rows = []
    for name, bt in pnl_dict.items():
        if bt is None or bt.empty:
            continue
        pnl = bt['net_pnl_usd']
        rows.append({
            'strategy': name,
            'n_trades': len(bt),
            'total_pnl': round(float(pnl.sum()), 2),
            'mean_pnl': round(float(pnl.mean()), 2),
            'sharpe': round(compute_sharpe(pnl), 3),
            'win_rate': round(float((pnl > 0).mean()), 3),
            'mean_g_gwei': round(float(bt['g_bid'].mean()), 3),
        })
    return pd.DataFrame(rows).set_index('strategy')


def gas_bid_comparison(df):
    rows = []
    actual = df['gas_price_gwei'].values
    for col, name in (('g_cf', 'CF'), ('g_hjb', 'HJB'), ('g_naive', 'Naive')):
        if col not in df.columns:
            continue
        pred = df[col].values
        mask = np.isfinite(pred) & np.isfinite(actual)
        if mask.sum() < 5:
            continue
        r, _ = pearsonr(actual[mask], pred[mask])
        rows.append({
            'model': name,
            'mean_actual': round(actual[mask].mean(), 3),
            'mean_model': round(pred[mask].mean(), 3),
            'bias': round((pred[mask] - actual[mask]).mean(), 3),
            'mae': round(np.abs(pred[mask] - actual[mask]).mean(), 3),
            'corr': round(r, 3),
            'pct_undercut': round((pred[mask] < actual[mask]).mean() * 100, 1),
        })
    return pd.DataFrame(rows).set_index('model')


def walk_forward_hft(state, params_global, blocks, cex_1m,
                     u_hjb, g_star_hjb, n_splits=4, Q=Q_ETH):
    state = state.sort_values('timestamp').reset_index(drop=True)
    n = len(state)
    fold_size = n // (n_splits + 1)
    if fold_size < 10:
        return pd.DataFrame()

    all_rows = []
    for k in range(1, n_splits + 1):
        train_end = k * fold_size
        test_end = min((k + 1) * fold_size, n)
        train = state.iloc[:train_end]
        test = state.iloc[train_end:test_end]
        if len(test) < 5:
            continue

        px_train = build_px_hft(train, cex_1m.loc[:train['dt'].max()])
        if px_train.empty:
            continue
        oracle = gas_oracle_from_blocks(blocks)
        slip_train = _prepare_swaps_for_slip(train)
        p_k = calibrate_all(
            px_train, blocks, oracle, Q=Q,
            swaps=slip_train,
            slip_alpha=params_global.get('slip_alpha'),
            slip_beta=params_global.get('slip_beta'),
        )
        u_k, _, g_k = solve_hjb_1d(p_k, Q=Q)

        test_bids = add_model_bids(test, p_k, u_hjb=u_k, g_star_hjb=g_k, Q=Q)
        pnl_k = compute_strategy_pnl(test_bids, p_k, Q=Q)
        for strat, bt in pnl_k.items():
            if bt.empty:
                continue
            all_rows.append({
                'fold': k,
                'strategy': strat,
                'n_trades': len(bt),
                'sharpe': round(compute_sharpe(bt['net_pnl_usd']), 3),
                'total_pnl': round(float(bt['net_pnl_usd'].sum()), 2),
            })
    return pd.DataFrame(all_rows)


def regime_comparison_hft(state, params_base, gas_regimes, u_hjb, g_star_hjb, Q=Q_ETH):
    all_rows = []
    for regime_name, override in gas_regimes.items():
        p = {**params_base, **override}
        g_lo = p.get('g_lo', p.get('base_fee', 1.0) + 0.05)
        g_hi = p.get('g_hi', p.get('fast_gwei', 50.0) * 3.0)
        p['g_lo'] = g_lo
        p['g_hi'] = g_hi

        bids = state.copy()
        bids['g_hjb'] = bids['u_val'].apply(
            lambda u: float(np.interp(np.clip(u, u_hjb[0], u_hjb[-1]), u_hjb, g_star_hjb))
        )
        bids['g_cf'] = bids.apply(
            lambda r: g_closedform(
                float(r.get('K_t', params_base['K_ref'])),
                float(r.get('A_t', params_base['A_ref'])),
                float(r['cex_price']),
                p['kappa'], p['beta'], GAMMA, g_lo=g_lo, g_hi=g_hi,
            ),
            axis=1,
        )
        bids['g_naive'] = float(p.get('propose_gwei', g_lo * 1.5))
        actual = bids['gas_price_gwei']
        bids['inc_cf'] = (bids['g_cf'] >= actual).astype(int)
        bids['inc_hjb'] = (bids['g_hjb'] >= actual).astype(int)
        bids['inc_naive'] = (bids['g_naive'] >= actual).astype(int)

        pnl = compute_strategy_pnl(bids, p, Q=Q)
        naive_sh = (
            compute_sharpe(pnl['Naive']['net_pnl_usd'])
            if 'Naive' in pnl and not pnl['Naive'].empty
            else 0.0
        )
        for strat, bt in pnl.items():
            if bt.empty:
                continue
            sh = compute_sharpe(bt['net_pnl_usd'])
            all_rows.append({
                'regime': regime_name,
                'strategy': strat,
                'n_trades': len(bt),
                'total_pnl': round(float(bt['net_pnl_usd'].sum()), 2),
                'sharpe': round(sh, 3),
                'win_rate': round(float((bt['net_pnl_usd'] > 0).mean()), 3),
                'mean_g_gwei': round(float(bt['g_bid'].mean()), 4),
                'delta_sharpe_vs_naive': round(sh - naive_sh, 3),
            })
    return pd.DataFrame(all_rows)

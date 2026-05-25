import warnings
import numpy as np
import pandas as pd
from scipy import stats as _stats

from .config import Q_GRID, SYNTH_REGIMES, GAMMA, GAS_USED, Q_ETH
from .models import (g_closedform, lambda_g, expected_delay_s,
                     gas_cost_usd, slip_usd, build_K_t, build_A_t)
from .backtest import run_backtest, compute_sharpe, backtest_summary, bootstrap_pnl
from .hjb import solve_hjb_1d


def compute_q_regime_grid(base_params, gas_regimes, Q_grid=Q_GRID,
                           median_spread_usd=0.5, spread_std_usd=None, S=None):
    S = S or base_params['S0']
    sig_d_h = base_params['sigma_d'] / np.sqrt(365 * 24)
    rho     = base_params['rho']
    rows    = []
    # Spread std: variance floor so Sharpe doesn't blow up when λ is huge.
    # When execution is instantaneous the only risk is spread variation at entry.
    _spread_std = spread_std_usd if spread_std_usd is not None else median_spread_usd

    for regime_name, regime_override in gas_regimes.items():
        p = {**base_params, **regime_override}
        kappa = p['kappa'];  beta_ = p['beta']
        g_lo  = p.get('g_lo', p.get('base_fee', 1.0) + 0.05)
        g_hi  = p.get('g_hi', p.get('fast_gwei', 50.0) * 3.0)
        g_naive = float(p.get('propose_gwei', g_lo * 1.5))

        for Q in Q_grid:
            K_t = build_K_t(Q, S, sig_d_h, rho)
            A_t = build_A_t(Q, S, sig_d_h, p['slip_alpha'], p['slip_beta'])
            g_raw = g_closedform(K_t, A_t, S, kappa, beta_, GAMMA,
                                 g_lo=g_lo, g_hi=g_hi)
            g_raw_nc = g_closedform(K_t, A_t, S, kappa, beta_, GAMMA,
                                    g_lo=0.0, g_hi=1e9)

            tau_opt   = expected_delay_s(g_raw,   kappa, beta_)
            tau_naive = expected_delay_s(g_naive,  kappa, beta_)

            def _sharpe(g_, tau_, _K=K_t, _kappa=kappa, _beta=beta_, _Q=Q):
                gas  = gas_cost_usd(g_, S)
                slip = slip_usd(_Q, S, p['slip_alpha'], p['slip_beta'])
                # Total variance = execution-delay price var + entry spread var
                exec_var   = _K / max(lambda_g(g_, _kappa, _beta), 1e-9)
                spread_var = (_Q * _spread_std) ** 2
                mean_  = _Q * median_spread_usd - gas - slip
                std_   = np.sqrt(max(exec_var + spread_var, 1e-12))
                return mean_ / std_ if std_ > 0 else 0.0

            sh_opt   = _sharpe(g_raw,   tau_opt)
            sh_naive = _sharpe(g_naive, tau_naive)

            rows.append({
                'regime':        regime_name,
                'Q_ETH':         Q,
                'g_star_cf':     round(g_raw, 4),
                'g_raw_nc':      round(g_raw_nc, 6),
                'corner':        g_raw_nc < g_lo - 1e-9,
                'E_tau_opt_s':   round(tau_opt, 1),
                'E_tau_naive_s': round(tau_naive, 1),
                'Sharpe_CF':     round(sh_opt, 3),
                'Sharpe_Naive':  round(sh_naive, 3),
                'Delta_Sharpe':  round(sh_opt - sh_naive, 3),
            })

    return pd.DataFrame(rows)


# ── Regime comparison backtest ────────────────────────────────────────────────

def compute_regime_comparison(opp_df, base_params, gas_regimes,
                               u_hjb, g_star_hjb,
                               dex_series=None, Q=Q_ETH):
    """
    Run the backtest under each gas regime and return a comparison table.

    This is the primary experimental result: in calm regimes the model correctly
    falls back to the corner solution (all strategies equal), while in
    congested/moderate regimes the optimised gas bid produces ΔSharpe > 0.

    Returns
    -------
    summary_df : DataFrame with columns:
        regime, strategy, n_trades, total_pnl, sharpe, win_rate,
        mean_g_gwei, mean_tau_s, delta_sharpe_vs_naive
    """
    all_rows = []

    for regime_name, regime_override in gas_regimes.items():
        p = {**base_params, **regime_override, 'Q': Q}
        kappa = p['kappa'];  beta_ = p['beta']
        g_lo  = p.get('g_lo',  p.get('base_fee', 1.0) + 0.05)
        g_hi  = p.get('g_hi',  p.get('fast_gwei', 50.0) * 3.0)
        p['g_lo'] = g_lo;  p['g_hi'] = g_hi

        def hjb_pol(row):
            u = np.log(max(float(row['cex_close']) /
                           max(float(row.get('dex_mid', row['cex_close'])), 1e-6), 1e-6))
            return float(np.interp(np.clip(u, u_hjb[0], u_hjb[-1]), u_hjb, g_star_hjb))

        def cf_pol(row):
            return g_closedform(
                row.get('K_t', p['K_ref']), row.get('A_t', p['A_ref']),
                float(row['cex_close']), kappa, beta_, GAMMA, g_lo=g_lo, g_hi=g_hi)

        def naive_pol(row):
            return float(p.get('propose_gwei', g_lo * 1.5))

        strategies = [('HJB', hjb_pol), ('CF', cf_pol), ('Naive', naive_pol)]
        bt_dict = {}
        for name, pol in strategies:
            bt = run_backtest(opp_df, pol, f'{name} [{regime_name}]', p,
                              dex_series=dex_series, Q_trade=Q,
                              kappa_override=kappa, beta_override=beta_,
                              g_lo_override=g_lo, g_hi_override=g_hi)
            bt_dict[name] = bt

        naive_sharpe = compute_sharpe(bt_dict['Naive']['net_pnl']) if not bt_dict['Naive'].empty else 0.0

        for name, bt in bt_dict.items():
            if bt.empty:
                continue
            sh = compute_sharpe(bt['net_pnl'])
            all_rows.append({
                'regime':                regime_name,
                'strategy':              name,
                'n_trades':              len(bt),
                'total_pnl':             float(bt['net_pnl'].sum()),
                'mean_pnl':              float(bt['net_pnl'].mean()),
                'sharpe':                round(sh, 3),
                'win_rate':              round(float((bt['net_pnl'] > 0).mean()), 3),
                'mean_g_gwei':           round(float(bt['g'].mean()), 4),
                'mean_tau_s':            round(float(bt['tau_s'].mean()), 1),
                'delta_sharpe_vs_naive': round(sh - naive_sharpe, 3),
            })

    return pd.DataFrame(all_rows)


# ── γ × Q sensitivity surface ─────────────────────────────────────────────────

def compute_sensitivity_surface(base_params, gammas, Qs, median_spread_usd=0.5, S=None):
    """
    Compute g*(γ, Q) and analytical Sharpe proxy on a grid.
    Returns three 2D arrays: g_surface, tau_surface, sharpe_surface.
    """
    S = S or base_params['S0']
    sig_d_h = base_params['sigma_d'] / np.sqrt(365 * 24)
    rho     = base_params['rho']
    kappa   = base_params['kappa'];  beta_ = base_params['beta']
    g_lo    = base_params['g_lo'];   g_hi  = base_params['g_hi']

    G   = len(gammas);  Q_ = len(Qs)
    g_surf    = np.zeros((G, Q_))
    g_raw_surf = np.zeros((G, Q_))
    tau_surf  = np.zeros((G, Q_))
    sh_surf   = np.zeros((G, Q_))

    for i, ga in enumerate(gammas):
        for j, Q in enumerate(Qs):
            K_t = build_K_t(Q, S, sig_d_h, rho)
            A_t = build_A_t(Q, S, sig_d_h, base_params['slip_alpha'], base_params['slip_beta'])
            g_raw    = g_closedform(K_t, A_t, S, kappa, beta_, ga, g_lo=g_lo, g_hi=g_hi)
            g_raw_nc = g_closedform(K_t, A_t, S, kappa, beta_, ga, g_lo=0.0, g_hi=1e9)
            tau_opt  = expected_delay_s(g_raw, kappa, beta_)
            gas      = gas_cost_usd(g_raw, S)
            sl       = slip_usd(Q, S, base_params['slip_alpha'], base_params['slip_beta'])
            res_var  = K_t / max(lambda_g(g_raw, kappa, beta_), 1e-9)
            mean_    = Q * median_spread_usd - gas - sl
            std_     = np.sqrt(max(res_var, 1e-12))
            g_surf[i, j]     = g_raw
            g_raw_surf[i, j] = g_raw_nc
            tau_surf[i, j]   = tau_opt
            sh_surf[i, j]    = mean_ / std_ if std_ > 0 else 0.0

    return g_surf, g_raw_surf, tau_surf, sh_surf


# ── Regime detection ──────────────────────────────────────────────────────────

def detect_regimes(df, roll_trend=4 * 24 * 7, roll_vol=2 * 24 * 7):
    """Classify hourly bars as Bull/Bear/Volatile/Calm based on trend and vol."""
    df = df.copy()
    lp = np.log(df['cex_close'])
    df['trend']    = (lp - lp.shift(roll_trend)) / roll_trend
    df['roll_vol'] = lp.diff().rolling(roll_vol).std() * np.sqrt(24 * 365)
    vol_med = df['roll_vol'].median()
    df['regime'] = 'Calm'
    df.loc[(df['trend'] > 0) & (df['roll_vol'] <= vol_med), 'regime'] = 'Bull'
    df.loc[(df['trend'] < 0) & (df['roll_vol'] <= vol_med), 'regime'] = 'Bear'
    df.loc[df['roll_vol'] > vol_med,                         'regime'] = 'Volatile'
    return df


# ── Bootstrap table ───────────────────────────────────────────────────────────

def bootstrap_ci_table(bt_dict, n_boot=500):
    """
    Compute bootstrap 95% CI for each strategy in bt_dict.
    Returns DataFrame with columns: strategy, sharpe_lo, sharpe_hi, mean_lo, mean_hi.
    """
    rows = []
    for name, bt in bt_dict.items():
        ci = bootstrap_pnl(bt, n_boot=n_boot)
        if ci:
            rows.append({'strategy': name, **ci})
    return pd.DataFrame(rows)


# ── Descriptive statistics (Table 1) ─────────────────────────────────────────

def compute_descriptive_stats(px, blocks=None, gas_oracle=None):
    """
    Build a thesis-standard descriptive statistics table for the aligned panel.

    Returns DataFrame indexed by variable name with columns:
    Mean, Std, Min, Median, Max, N, Freq.
    Source: «Расчёты автора».
    """
    rows = []

    def _row(name, series, freq='1h'):
        s = series.dropna()
        if len(s) == 0:
            return
        rows.append({
            'Variable':  name,
            'N':         len(s),
            'Mean':      float(s.mean()),
            'Std':       float(s.std()),
            'Min':       float(s.min()),
            'Median':    float(s.median()),
            'Max':       float(s.max()),
            'Skew':      float(s.skew()),
            'Kurt':      float(s.kurtosis()),
            'Freq':      freq,
        })

    if 'cex_close' in px.columns:
        _row('CEX price S^C (USD)',       px['cex_close'])
    if 'dex_mid' in px.columns:
        _row('DEX mid price S^D (USD)',   px['dex_mid'])
    if 'spread_usd' in px.columns:
        _row('Spread |S^C − S^D| (USD)',  px['spread_usd'].abs())
    if 'spread_bps' in px.columns:
        _row('Spread (bps)',               px['spread_bps'].abs())
    if 'ret_cex' in px.columns:
        _row('CEX log-return (hourly)',    px['ret_cex'] * 100)
    if 'ret_dex' in px.columns:
        _row('DEX log-return (hourly)',    px['ret_dex'] * 100)

    if blocks is not None and not blocks.empty:
        bf_col = 'baseFeePerGas' if 'baseFeePerGas' in blocks.columns else 'baseFee_gwei'
        if bf_col in blocks.columns:
            _row('Base fee (gwei)', blocks[bf_col], freq='~12s')

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index('Variable').round(4)


# ── Paired t-test (statistical significance of ΔPnL) ─────────────────────────

def paired_ttest_strategies(bt_dict, reference='Naive', alpha=0.05):
    """
    Paired t-test: for each non-reference strategy, test H₀: E[PnL_strat − PnL_naive] = 0
    vs H₁: E[PnL_strat − PnL_naive] > 0.

    Trades must be aligned by timestamp (same opportunity set).

    Returns DataFrame: strategy, n, mean_delta, std_delta, t_stat, p_value, significant.
    """
    ref = bt_dict.get(reference)
    if ref is None or ref.empty:
        warnings.warn(f'Reference strategy {reference!r} not found or empty.')
        return pd.DataFrame()

    rows = []
    for name, bt in bt_dict.items():
        if name == reference or bt is None or bt.empty:
            continue
        # Align by index (timestamp)
        common = bt.index.intersection(ref.index)
        if len(common) < 10:
            continue
        delta = bt.loc[common, 'net_pnl'].values - ref.loc[common, 'net_pnl'].values
        t, p_two = _stats.ttest_1samp(delta, 0.0)
        p_one = p_two / 2 if t > 0 else 1.0 - p_two / 2
        rows.append({
            'strategy':    name,
            'n':           len(common),
            'mean_delta':  float(delta.mean()),
            'std_delta':   float(delta.std()),
            't_stat':      round(float(t), 3),
            'p_value':     round(float(p_one), 4),
            'significant': bool(p_one < alpha),
        })
    return pd.DataFrame(rows)


# ── Period-specific historical backtest ───────────────────────────────────────

def compute_period_backtest(px_full, period_specs, params_base,
                             u_hjb_base, g_star_hjb_base,
                             dex_series=None, Q=Q_ETH,
                             n_boot=500):
    from .calibrate import calibrate_all, build_state_df

    results = {}

    for period_name, spec in period_specs.items():
        start = spec.get('start')
        end   = spec.get('end')
        hist_blocks = spec.get('blocks', pd.DataFrame())
        oracle_proxy = spec.get('gas_oracle_proxy', {})

        # Slice the price panel for this period
        if start and end:
            px_slice = px_full.loc[start:end]
        elif start:
            px_slice = px_full.loc[start:]
        elif end:
            px_slice = px_full.loc[:end]
        else:
            px_slice = px_full.copy()

        if len(px_slice) < 50:
            warnings.warn(f'{period_name}: only {len(px_slice)} rows, skipping.')
            continue
        if hist_blocks.empty:
            warnings.warn(f'{period_name}: no block data, skipping.')
            continue

        params_p = calibrate_all(px_slice, hist_blocks, oracle_proxy, Q=Q)
        params_p['gamma'] = params_base.get('gamma', GAMMA)
        params_p['slip_alpha'] = params_base['slip_alpha']
        params_p['slip_beta'] = params_base['slip_beta']

        state_p = build_state_df(px_slice, params_p, Q=Q)

        is_opp_mask = state_p['is_opp'] if 'is_opp' in state_p.columns else pd.Series(True, index=state_p.index)
        opp_p = state_p[is_opp_mask].copy()
        if len(opp_p) < 10:
            warnings.warn(f'{period_name}: only {len(opp_p)} opportunities, skipping.')
            continue

        u_p, phi_p, g_p = solve_hjb_1d(params_p, gamma=GAMMA, Q=Q)

        kappa_p = params_p['kappa'];  beta_p = params_p['beta']
        g_lo_p  = params_p['g_lo'];   g_hi_p = params_p['g_hi']

        def hjb_pol(row, _u=u_p, _g=g_p):
            uu = np.log(max(float(row['cex_close']) /
                            max(float(row.get('dex_mid', row['cex_close'])), 1e-6), 1e-6))
            return float(np.interp(np.clip(uu, _u[0], _u[-1]), _u, _g))

        def cf_pol(row, _p=params_p):
            return g_closedform(
                row.get('K_t', _p['K_ref']), row.get('A_t', _p['A_ref']),
                float(row['cex_close']), _p['kappa'], _p['beta'], GAMMA,
                g_lo=_p['g_lo'], g_hi=_p['g_hi'])

        def naive_pol(row, _p=params_p):
            return float(_p.get('propose_gwei', _p['g_lo']))

        bt_h = run_backtest(opp_p, hjb_pol,   f'HJB [{period_name}]', params_p, dex_series=dex_series, Q_trade=Q)
        bt_c = run_backtest(opp_p, cf_pol,    f'CF  [{period_name}]', params_p, dex_series=dex_series, Q_trade=Q)
        bt_n = run_backtest(opp_p, naive_pol, f'NV  [{period_name}]', params_p, dex_series=dex_series, Q_trade=Q)

        sh_cf = compute_sharpe(bt_c['net_pnl']) if not bt_c.empty else float('nan')
        sh_nv = compute_sharpe(bt_n['net_pnl']) if not bt_n.empty else float('nan')

        ttest_df = paired_ttest_strategies({'CF': bt_c, 'HJB': bt_h, 'Naive': bt_n},
                                            reference='Naive')

        results[period_name] = {
            'params':        params_p,
            'n_opp':         len(opp_p),
            'bt_hjb':        bt_h,
            'bt_cf':         bt_c,
            'bt_naive':      bt_n,
            'sharpe_cf':     sh_cf,
            'sharpe_naive':  sh_nv,
            'delta_sharpe':  sh_cf - sh_nv,
            'ttest':         ttest_df,
        }

    return results

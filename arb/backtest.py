import numpy as np
import pandas as pd

from .config import GAS_USED, GAMMA, Q_ETH, N_MC_PATHS, T_HORIZON, NAIVE_GAS_FIELD
from .models import gas_cost_usd, lambda_g, slip_usd, crra, g_closedform


def run_backtest(
    opp_df,
    g_policy_fn,
    label,
    params,
    dex_series=None,
    Q_trade=None,
    kappa_override=None,
    beta_override=None,
    g_lo_override=None,
    g_hi_override=None,
    rng_seed=123,
):
    if opp_df is None or opp_df.empty:
        return pd.DataFrame()

    Q_tr = float(Q_trade) if Q_trade is not None else float(params.get('Q', Q_ETH))
    kappa = kappa_override if kappa_override is not None else params['kappa']
    beta_ = beta_override if beta_override is not None else params['beta']
    g_lo = g_lo_override if g_lo_override is not None else params['g_lo']
    g_hi = g_hi_override if g_hi_override is not None else params['g_hi']
    is_naive = 'naive' in label.lower()

    dex_sorted = (
        dex_series.sort_index()
        if dex_series is not None and not dex_series.empty
        else pd.Series(dtype=float)
    )
    sig_per_s = float(params['sigma_d']) / np.sqrt(365 * 24 * 3600)
    rng = np.random.default_rng(rng_seed)
    n_actual = n_sim = 0
    records = []

    for ts, row in opp_df.iterrows():
        g_raw = float(g_policy_fn(row))
        floor = float(params.get('base_fee', g_lo)) if is_naive else g_lo
        g = float(np.clip(g_raw, floor, g_hi))
        lam = lambda_g(g, kappa, beta_)
        tau_s = float(np.clip(3600.0 / max(lam, 1e-9), 5, 3600))

        direction = float(np.sign(row.get('spread_usd', 1.0))) or 1.0
        dex_now = float(row.get('dex_mid', row['cex_close']))
        S = float(row['cex_close'])
        fut_dex = np.nan

        if not dex_sorted.empty:
            next_idx = dex_sorted.index.searchsorted(ts, side='right')
            if next_idx < len(dex_sorted):
                next_price = float(dex_sorted.iloc[next_idx])
                next_ts = dex_sorted.index[next_idx]
                bar_secs = max((next_ts - ts).total_seconds(), 60.0)
                weight = min(tau_s / bar_secs, 1.0)
                fut_dex = dex_now + weight * (next_price - dex_now)
                n_actual += 1

        if pd.isna(fut_dex):
            fut_dex = dex_now * (1.0 + rng.normal(0.0, sig_per_s * np.sqrt(tau_s)))
            n_sim += 1

        delta_dex = float(fut_dex) - dex_now
        pnl_spread = Q_tr * abs(float(row.get('spread_usd', 0.0)))
        pnl_move = -direction * Q_tr * delta_dex
        cost_gas_v = gas_cost_usd(g, S)
        cost_slip_v = slip_usd(Q_tr, S, params['slip_alpha'], params['slip_beta'])
        net = pnl_spread + pnl_move - cost_gas_v - cost_slip_v
        records.append({
            'ts': ts,
            'regime': row.get('regime', '?'),
            'g': g,
            'tau_s': tau_s,
            'net_pnl': net,
            'pnl_spread': pnl_spread,
            'pnl_move': pnl_move,
            'cost_gas': cost_gas_v,
            'direction': direction,
        })

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records).set_index('ts')
    out['cum_pnl'] = out['net_pnl'].cumsum()
    sh = compute_sharpe(out['net_pnl'])
    print(label, len(out), round(out['net_pnl'].sum(), 2), round(sh, 3))
    return out


def monte_carlo_eval(
    u_hjb, g_star_hjb, params,
    cf_K_ref=None, cf_A_ref=None,
    N_paths=N_MC_PATHS, T_horizon=T_HORIZON,
    dt_h=1 / 60, gamma=GAMMA, Q=Q_ETH,
):
    N_steps = int(T_horizon / dt_h)
    S = params['S0']
    W0 = params['W0']
    kappa = params['kappa']
    beta_ = params['beta']
    sig_c = params['sigma_c'] / np.sqrt(365 * 24)
    sig_d = params['sigma_d'] / np.sqrt(365 * 24)
    sig_f = params['sigma_f'] / np.sqrt(365 * 24)
    rho = params['rho']
    rho_cf = params.get('rho_cf', rho)

    C = np.array([[1, rho, rho_cf], [rho, 1, rho], [rho_cf, rho, 1]])
    L = np.linalg.cholesky(C + 1e-6 * np.eye(3))

    K_ref = cf_K_ref or params.get('K_ref', 1.0)
    A_ref = cf_A_ref or params.get('A_ref', 0.0)
    g_lo = params['g_lo']
    g_hi = params['g_hi']
    g_naive = float(params.get(NAIVE_GAS_FIELD, params.get('propose_gwei', g_lo)))

    policies = {
        'HJB': lambda z: float(np.interp(np.log(max(z, 1e-6)), u_hjb, g_star_hjb)),
        'CF': lambda z: g_closedform(
            K_ref, A_ref, S, kappa, beta_, gamma, g_lo=g_lo, g_hi=g_hi
        ),
        'Naive': lambda z: g_naive,
    }
    results = {}

    for pol_name, g_pol in policies.items():
        Sc = S * np.ones(N_paths)
        Sd = S * np.ones(N_paths)
        F = S * np.ones(N_paths)
        W = W0 * np.ones(N_paths)
        executed = np.zeros(N_paths, dtype=bool)

        for _ in range(N_steps):
            Z = np.random.randn(3, N_paths)
            dW = L @ Z * np.sqrt(dt_h)
            Sc *= np.exp(-sig_c ** 2 / 2 * dt_h + sig_c * dW[0])
            Sd *= np.exp(-sig_d ** 2 / 2 * dt_h + sig_d * dW[1])
            F *= np.exp(-sig_f ** 2 / 2 * dt_h + sig_f * dW[2])

            z = Sc / np.maximum(Sd, 1e-6)
            g_v = np.array([g_pol(z[i]) for i in range(N_paths)])
            lam_v = lambda_g(g_v, kappa, beta_)
            jump = np.random.rand(N_paths) < lam_v * dt_h
            newly_exec = jump & ~executed
            if newly_exec.any():
                spread_frac = np.abs(z[newly_exec] - 1)
                Q_frac = Q * Sc[newly_exec] / W[newly_exec]
                slip_frac = (
                    params['slip_alpha'] * (Q * Sc[newly_exec]) ** params['slip_beta']
                    / 1e4 * Sc[newly_exec] / W[newly_exec]
                )
                gas_frac = GAS_USED * g_v[newly_exec] * 1e-9 * Sc[newly_exec] / W[newly_exec]
                G_tilde = Q_frac * spread_frac - slip_frac - gas_frac
                W[newly_exec] *= (1.0 + G_tilde)
                executed[newly_exec] = True

        results[pol_name] = W.copy()

    return results


def _pnl_col(bt):
    for col in ('net_pnl', 'net_pnl_usd', 'net_pnl_model'):
        if col in bt.columns:
            return col
    raise KeyError(f'no PnL column in backtest frame: {list(bt.columns)}')


def bootstrap_pnl(bt, n_boot=500):
    if bt is None or bt.empty:
        return {}
    pnl = bt[_pnl_col(bt)].values
    means, sharpes = [], []
    for _ in range(n_boot):
        sample = np.random.choice(pnl, size=len(pnl), replace=True)
        means.append(sample.mean())
        sharpes.append(sample.mean() / sample.std() if sample.std() > 0 else 0.0)
    return {
        'mean_lo': float(np.percentile(means, 2.5)),
        'mean_hi': float(np.percentile(means, 97.5)),
        'sharpe_lo': float(np.percentile(sharpes, 2.5)),
        'sharpe_hi': float(np.percentile(sharpes, 97.5)),
    }


def walk_forward_backtest(
    df_state, params, u_hjb, g_star_hjb,
    dex_series=None, wf_train=14 * 24, wf_test=7 * 24,
):
    n = len(df_state)
    step = wf_test
    results = []

    for start in range(wf_train, n - step, step):
        test_df = df_state.iloc[start: start + step]
        opp_test = test_df[test_df.get('is_opp', True)].copy()
        if opp_test.empty:
            continue

        def hjb_pol(row):
            u = np.log(
                max(
                    float(row['cex_close'])
                    / max(float(row.get('dex_mid', row['cex_close'])), 1e-6),
                    1e-6,
                )
            )
            return float(np.interp(np.clip(u, u_hjb[0], u_hjb[-1]), u_hjb, g_star_hjb))

        def cf_pol(row):
            return g_closedform(
                row.get('K_t', params['K_ref']),
                row.get('A_t', params['A_ref']),
                float(row['cex_close']),
                params['kappa'],
                params['beta'],
                params.get('gamma', GAMMA),
                g_lo=params['g_lo'],
                g_hi=params['g_hi'],
            )

        def naive_pol(row):
            return float(params.get(NAIVE_GAS_FIELD, params.get('propose_gwei', params['g_lo'])))

        results.append({
            'start': test_df.index[0],
            'bt_hjb': run_backtest(opp_test, hjb_pol, 'HJB', params, dex_series),
            'bt_cf': run_backtest(opp_test, cf_pol, 'CF', params, dex_series),
            'bt_naive': run_backtest(opp_test, naive_pol, 'Naive', params, dex_series),
        })

    return results


def compute_sharpe(pnl_series):
    pnl = np.asarray(pnl_series, dtype=float)
    std = pnl.std()
    return float(pnl.mean() / std) if std > 0 else 0.0


def backtest_summary(bt, label=''):
    if bt is None or bt.empty:
        return {}
    pnl = bt[_pnl_col(bt)]
    return {
        'label': label,
        'n_trades': len(bt),
        'total_pnl': float(pnl.sum()),
        'mean_pnl': float(pnl.mean()),
        'std_pnl': float(pnl.std()),
        'sharpe': compute_sharpe(pnl),
        'win_rate': float((pnl > 0).mean()),
        'mean_g_gwei': float(bt['g'].mean()),
        'mean_tau_s': float(bt['tau_s'].mean()),
    }

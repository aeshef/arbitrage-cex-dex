import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from pathlib import Path

from .config import OUTPUT_DIR, ROLL_H

_FIG_DIR = OUTPUT_DIR / 'figures'
_FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, name):
    stem = name[len('figures/'):] if name.startswith('figures/') else name
    path = _FIG_DIR / f'{stem}.png'
    fig.savefig(path, dpi=130, bbox_inches='tight')
    return path


# ── Stylized facts ────────────────────────────────────────────────────────────

def plot_drift_check(cex_df):
    """
    A1: dimensionless drift-to-diffusion ratio |μ|·τ / (σ·√τ).

    μ and σ are the mean and std of log returns at the **native bar frequency**
    of ``cex_df``; τ is expressed in the same units via the median bar spacing
    of the index (so hourly data + τ in seconds is consistent; no annualised
    μ mixed with τ in hours).
    """
    log_ret = np.log(cex_df['close']).diff().dropna()
    mu = float(log_ret.mean())
    sig = float(log_ret.std())
    if not np.isfinite(sig) or sig <= 0:
        sig = 1e-12

    if len(cex_df.index) < 3:
        bar_sec = 3600.0
    else:
        dt = pd.Series(cex_df.index).diff().median()
        if hasattr(dt, 'total_seconds'):
            bar_sec = float(dt.total_seconds())
        else:
            bar_sec = 86400.0
        if not np.isfinite(bar_sec) or bar_sec <= 0:
            bar_sec = 86400.0

    tau_s = np.array([12.0, 60.0, 300.0, 900.0, 1800.0, 3600.0])
    tau_bars = tau_s / bar_sec
    ratio = np.abs(mu) * tau_bars / (sig * np.sqrt(tau_bars + 1e-20))

    labels = ['12 s', '1 m', '5 m', '15 m', '30 m', '1 h']
    x = np.arange(len(tau_s))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(x, ratio, color='steelblue', width=0.65, edgecolor='k', linewidth=0.4)
    ax.axhline(0.05, color='red', ls='--', lw=1.8, label='5% threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(r'$|\hat\mu|\,\tau / (\hat\sigma\sqrt{\tau})$  (same bar freq.)')
    ax.set_xlabel('Horizon τ')
    ax.set_title('A1: Drift negligibility (native bar → τ in seconds)')
    ax.set_ylim(0, max(0.06, float(np.nanmax(ratio)) * 1.15))
    for rect, val in zip(bars, ratio):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig


def plot_returns_histogram_qq(px):
    """A2: DEX log-return histogram + normal fit + QQ (large fonts for thesis)."""
    ret = px['ret_dex'].replace([np.inf, -np.inf], np.nan).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(ret, bins=60, density=True, alpha=0.65, color='steelblue', label='Empirical')
    x = np.linspace(ret.min(), ret.max(), 200)
    axes[0].plot(x, stats.norm.pdf(x, ret.mean(), ret.std()), 'r-', lw=2, label='Normal')
    axes[0].set_title('DEX log-returns (A2)', fontsize=13)
    axes[0].set_xlabel('log-return', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].legend(fontsize=11)
    stats.probplot(ret, dist='norm', plot=axes[1])
    axes[1].set_title('QQ-plot vs Normal', fontsize=13)
    axes[1].tick_params(labelsize=11)
    plt.tight_layout()
    return fig


def plot_returns_rolling_vol(px, roll_h=ROLL_H):
    """A3: rolling annualised σ only (full width for thesis)."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    roll_sig = px['ret_dex'].rolling(roll_h).std() * np.sqrt(365 * 24)
    roll_sig.dropna().plot(ax=ax, color='darkorange', lw=1.2)
    ax.set_title(f'Rolling σ ({roll_h}h window, annualised) — A3', fontsize=13)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('σ (ann.)', fontsize=12)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    return fig


def plot_returns_normality(px, roll_h=ROLL_H):
    """A2/A3 (legacy): three panels in one row — prefer split helpers for PDF."""
    ret = px['ret_dex'].replace([np.inf, -np.inf], np.nan).dropna()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(ret, bins=60, density=True, alpha=0.65, color='steelblue', label='Empirical')
    x = np.linspace(ret.min(), ret.max(), 200)
    axes[0].plot(x, stats.norm.pdf(x, ret.mean(), ret.std()), 'r-', lw=2, label='Normal')
    axes[0].set_title('DEX log-returns (A2)', fontsize=12)
    axes[0].legend(fontsize=10)
    stats.probplot(ret, dist='norm', plot=axes[1])
    axes[1].set_title('QQ-plot vs Normal', fontsize=12)
    roll_sig = px['ret_dex'].rolling(roll_h).std() * np.sqrt(365 * 24)
    roll_sig.dropna().plot(ax=axes[2], color='darkorange')
    axes[2].set_title(f'Rolling σ ({roll_h}h, ann.) — A3', fontsize=12)
    for ax in axes:
        ax.tick_params(labelsize=10)
    plt.tight_layout()
    return fig


def plot_correlation(px, roll_h=ROLL_H):
    """A4: rolling ρ(CEX, DEX) — single large panel + median/corridor (thesis)."""
    roll_rho = px['ret_cex'].rolling(roll_h, min_periods=roll_h).corr(px['ret_dex'])
    s = roll_rho.dropna()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(s.index, s.values, color='purple', lw=0.9, label=r'$\hat\rho$')
    med = float(s.median())
    ax.axhline(med, color='crimson', ls='--', lw=1.2, label=f'median={med:.3f}')
    lo, hi = float(s.quantile(0.05)), float(s.quantile(0.95))
    ax.axhspan(lo, hi, color='gray', alpha=0.15, label='5–95% band')
    ax.set_title('Rolling ρ(CEX, DEX) (A4)', fontsize=13)
    ax.set_ylabel(r'$\hat\rho_{cd}$', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.tick_params(labelsize=10)
    ax.margins(x=0.01)
    plt.tight_layout()
    return fig


def plot_lambda_fit(g_probe, lam_emp, kappa, beta_, title='λ(g) = κ·g^β fit'):
    """Calibration: empirical λ(g) data points and power-law fit."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(g_probe, lam_emp, s=20, color='steelblue', label='Empirical λ(g)')
    g_fit = np.linspace(g_probe.min(), g_probe.max(), 100)
    ax.plot(g_fit, kappa * g_fit ** beta_, 'r-', lw=2, label=f'κ={kappa:.1f}, β={beta_:.2f}')
    ax.set_xlabel('gas price (gwei)'); ax.set_ylabel('λ (per hour)')
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    return fig


# ── Model / HJB ──────────────────────────────────────────────────────────────

def plot_J_grid(g_test, J_grid, g_star_cf, g_star_grid):
    """4a-1: FOC verification plot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g_test, J_grid, 'b-', lw=2, label='J(g)')
    ax.axvline(g_star_cf,   color='red',   ls='--', lw=1.5, label=f'g*_cf={g_star_cf:.3f}')
    ax.axvline(g_star_grid, color='green', ls=':',  lw=1.5, label=f'g*_grid={g_star_grid:.3f}')
    ax.set_xlabel('g (gwei)'); ax.set_ylabel('J(g)')
    ax.set_title(f'J(g) grid check  |Δg*| = {abs(g_star_cf - g_star_grid):.4f} gwei')
    ax.legend(); plt.tight_layout()
    return fig


def plot_hjb_solution(u_hjb, phi_hjb, g_star_hjb, params=None,
                      u_1d=None, phi_1d=None, g_1d=None):
    """4b: HJB value function and policy g*(u)."""
    sp = np.expm1(u_hjb) * 1e4
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(sp, phi_hjb, 'b-', lw=2.5, label='HJB φ(0,u)')
    axes[0].axhline(1.0, color='gray', ls='--', alpha=0.5, label='φ=1 (terminal)')
    if u_1d is not None and phi_1d is not None:
        axes[0].plot(np.expm1(u_1d) * 1e4, phi_1d, 'm:', lw=2, alpha=0.8, label='1D (cmp)')
    axes[0].set_xlabel('spread (bps)'); axes[0].set_ylabel('φ(0,u)')
    axes[0].set_title('Value function φ'); axes[0].legend(fontsize=9)

    axes[1].plot(sp, g_star_hjb, 'b-', lw=2.5, label='HJB g*(u)')
    if u_1d is not None and g_1d is not None:
        axes[1].plot(np.expm1(u_1d) * 1e4, g_1d, 'm:', lw=2, alpha=0.8, label='1D (cmp)')
    if params is not None:
        axes[1].axhline(params['g_lo'], color='red', ls='--', lw=1.2, label='g_lo (floor)')
    axes[1].set_xlabel('spread (bps)'); axes[1].set_ylabel('g* (gwei)')
    axes[1].set_title('Candidate gas policy g*(u) (grid argmax)'); axes[1].legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_gamma_sigma_sensitivity(gammas, sigmas, g_mat_gamma, g_mat_sigma, g_lo=None):
    """4b-2: g*(γ) and g*(σ) sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for j, gam in enumerate(gammas):
        axes[0].plot(g_mat_gamma[j], label=f'γ={gam}')
    if g_lo is not None:
        axes[0].axhline(g_lo, ls='--', color='red', lw=1.2, label='g_lo')
    axes[0].set_xlabel('u index'); axes[0].set_ylabel('g* (gwei)')
    axes[0].set_title('g*(u) by risk aversion γ'); axes[0].legend(fontsize=8)

    for j, sig in enumerate(sigmas):
        axes[1].plot(g_mat_sigma[j], label=f'σ={sig}')
    if g_lo is not None:
        axes[1].axhline(g_lo, ls='--', color='red', lw=1.2, label='g_lo')
    axes[1].set_xlabel('u index'); axes[1].set_ylabel('g* (gwei)')
    axes[1].set_title('g*(u) by volatility σ'); axes[1].legend(fontsize=8)
    plt.tight_layout()
    return fig


# ── Backtest ──────────────────────────────────────────────────────────────────

def plot_backtest(bt_dict, title_suffix=''):
    """
    5.1: backtest panel — cumulative PnL, PnL distribution, PnL vs τ.
    bt_dict: {'HJB': df, 'CF': df, 'Naive': df}
    """
    colors = {'HJB': 'royalblue', 'CF': 'darkorange', 'Naive': 'gray'}
    styles = {'HJB': '--', 'CF': '-', 'Naive': '-'}
    widths = {'HJB': 2.0, 'CF': 1.5, 'Naive': 1.5}
    valid = {k: v for k, v in bt_dict.items() if v is not None and not v.empty}
    if not valid:
        return None

    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, :2])
    for name, bt in valid.items():
        bt['cum_pnl'].plot(ax=ax0, label=name, color=colors.get(name, 'k'),
                           lw=widths.get(name, 1.5), ls=styles.get(name, '-'))
    ax0.axhline(0, color='black', ls=':')
    ax0.legend(); ax0.set_title(f'Cumulative PnL {title_suffix}')

    ax1 = fig.add_subplot(gs[0, 2])
    pooled = pd.concat([bt['net_pnl'] for bt in valid.values()])
    q1, q99 = float(pooled.quantile(0.01)), float(pooled.quantile(0.99))
    for name, bt in valid.items():
        ax1.hist(bt['net_pnl'].clip(q1, q99), bins=45, alpha=0.45, density=True,
                 color=colors.get(name, 'k'), label=name)
    ax1.axvline(0, color='black', ls=':', lw=1)
    ax1.set_xlim(q1, q99)
    ax1.set_yscale('log')
    ax1.legend(fontsize=9)
    ax1.set_title('PnL density (1–99% clip, log y)')

    ax2 = fig.add_subplot(gs[1, 0])
    for name, bt in valid.items():
        ax2.scatter(bt['tau_s'].clip(0, 600), bt['net_pnl'],
                    alpha=0.3, s=5, color=colors.get(name, 'k'), label=name)
    ax2.axhline(0, color='black', ls=':')
    ax2.set_xlabel('τ (s)'); ax2.set_ylabel('PnL'); ax2.set_title('PnL vs execution delay')

    ax3 = fig.add_subplot(gs[1, 1])
    names   = list(valid.keys())
    sharpes = [bt['net_pnl'].mean() / bt['net_pnl'].std()
               if bt['net_pnl'].std() > 0 else 0 for bt in valid.values()]
    ax3.bar(names, sharpes, color=[colors.get(n, 'k') for n in names])
    ax3.axhline(0, color='black', ls=':'); ax3.set_title('Per-trade Sharpe ratio')
    ax3.set_ylabel('Sharpe')

    ax4 = fig.add_subplot(gs[1, 2])
    means = [bt['g'].mean() for bt in valid.values()]
    ax4.bar(names, means, color=[colors.get(n, 'k') for n in names])
    ax4.set_title('Mean gas bid (gwei)'); ax4.set_ylabel('gwei')

    return fig


def plot_regime_comparison(regime_df):
    """Visualise ΔSharpe across regimes and Q for the three strategies."""
    if regime_df.empty:
        return None

    pivot = regime_df.pivot_table(
        index='regime', columns='strategy', values='delta_sharpe_vs_naive')
    if pivot.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pivot.plot(kind='bar', ax=axes[0], color=['royalblue', 'darkorange', 'gray'],
               alpha=0.85, edgecolor='k', linewidth=0.5)
    axes[0].axhline(0, color='black', ls=':')
    axes[0].set_xlabel('Gas regime'); axes[0].set_ylabel('ΔSharpe vs Naive')
    axes[0].set_title('ΔSharpe by gas regime'); axes[0].legend(title='Strategy')
    axes[0].tick_params(axis='x', rotation=20)

    pivot2 = regime_df.pivot_table(index='regime', columns='strategy', values='sharpe')
    if not pivot2.empty:
        pivot2.plot(kind='bar', ax=axes[1], alpha=0.85, edgecolor='k', linewidth=0.5)
        axes[1].axhline(0, color='black', ls=':')
        axes[1].set_xlabel('Gas regime'); axes[1].set_ylabel('Per-trade Sharpe')
        axes[1].set_title('Sharpe by regime'); axes[1].legend(title='Strategy')
        axes[1].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    return fig


def plot_q_regime_heatmap(grid_df, pos_color_percentile: float = 72.0):
    """
    Heatmap of ΔSharpe(CF vs Naive) on Q × regime grid.

    Цветовая шкала: центр в нуле; положительный хвост «сжимается» по перцентилю,
    чтобы умеренные +ΔSharpe не оказывались у красного из‑за выбросов вроде +1.7.
    Значения выше cap отображаются насыщенным зелёным (как и cap).
    """
    if grid_df.empty:
        return None
    pivot = grid_df.pivot_table(index='regime', columns='Q_ETH', values='Delta_Sharpe')
    if pivot.empty:
        return None
    vals = np.asarray(pivot.values, dtype=float)
    neg_floor = float(min(-0.12, np.nanmin(vals)))
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.size:
        pos_cap = float(np.nanpercentile(pos, pos_color_percentile))
        pos_cap = max(pos_cap, 0.06)
        pos_cap = min(pos_cap, float(np.nanmax(vals)))
    else:
        pos_cap = 0.1
    norm = TwoSlopeNorm(vmin=neg_floor, vcenter=0.0, vmax=pos_cap)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(vals, aspect='auto', cmap='RdYlGn', norm=norm)
    cb = plt.colorbar(im, ax=ax, label='ΔSharpe (CF − Naive)')
    cb.ax.tick_params(labelsize=9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{q}' for q in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, rotation=0)
    ax.set_xlabel('Q (ETH)')
    ax.set_title('ΔSharpe: Closed-form vs Naive  (Q × gas regime)')
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color='black', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_sensitivity_surface_g_tau(gammas, Qs, g_raw_surf, tau_surf):
    """6.2a: γ × Q — unclipped g* and E[τ] only (readable thesis figure)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    kw = dict(aspect='auto', origin='lower',
              extent=[Qs[0], Qs[-1], gammas[0], gammas[-1]])
    im0 = axes[0].imshow(g_raw_surf, **kw, cmap='viridis')
    plt.colorbar(im0, ax=axes[0], label='g* unclipped (gwei)')
    axes[0].set_xlabel('Q (ETH)', fontsize=12)
    axes[0].set_ylabel('γ', fontsize=12)
    axes[0].set_title('Unclipped g*(γ, Q)', fontsize=12)

    im1 = axes[1].imshow(tau_surf, **kw, cmap='coolwarm')
    plt.colorbar(im1, ax=axes[1], label='E[τ] (s)')
    axes[1].set_xlabel('Q (ETH)', fontsize=12)
    axes[1].set_ylabel('γ', fontsize=12)
    axes[1].set_title('Expected delay E[τ](g*)', fontsize=12)
    for ax in axes:
        ax.tick_params(labelsize=10)
    plt.tight_layout()
    return fig


def plot_sensitivity_surface_sharpe(gammas, Qs, sh_surf):
    """6.2b: Sharpe proxy surface alone."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    kw = dict(aspect='auto', origin='lower',
              extent=[Qs[0], Qs[-1], gammas[0], gammas[-1]])
    im = ax.imshow(sh_surf, **kw, cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='Sharpe proxy')
    ax.set_xlabel('Q (ETH)', fontsize=12)
    ax.set_ylabel('γ', fontsize=12)
    ax.set_title('Sharpe proxy at g*(γ, Q)', fontsize=12)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    return fig


def plot_sensitivity_surface(gammas, Qs, g_raw_surf, tau_surf, sh_surf):
    """6.2: γ × Q sensitivity — three panels (legacy); prefer split plots for PDF."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    kw = dict(aspect='auto', origin='lower',
              extent=[Qs[0], Qs[-1], gammas[0], gammas[-1]])
    im0 = axes[0].imshow(g_raw_surf, **kw, cmap='viridis')
    plt.colorbar(im0, ax=axes[0], label='g* unclipped (gwei)')
    axes[0].set_xlabel('Q (ETH)'); axes[0].set_ylabel('γ')
    axes[0].set_title('Unclipped g*(γ, Q)')

    im1 = axes[1].imshow(tau_surf, **kw, cmap='coolwarm')
    plt.colorbar(im1, ax=axes[1], label='E[τ] (s)')
    axes[1].set_xlabel('Q (ETH)'); axes[1].set_ylabel('γ')
    axes[1].set_title('Expected delay E[τ](g*)')

    im2 = axes[2].imshow(sh_surf, **kw, cmap='RdYlGn')
    plt.colorbar(im2, ax=axes[2], label='Sharpe proxy')
    axes[2].set_xlabel('Q (ETH)'); axes[2].set_ylabel('γ')
    axes[2].set_title('Sharpe proxy at g*(γ, Q)')

    plt.tight_layout()
    return fig


def plot_2d_hjb_heatmap(xg, yg, phi_2d, g_star_2d):
    """
    2D HJB: heatmap of φ in (log S^C, log S^D) space only.

    The g*(x,y) panel is omitted: on the calibrated grid it is visually uniform at the gas
    floor and repeatedly confused readers / looked like a rendering bug. ``g_star_2d`` is kept
    in the signature for notebook/cache compatibility.
    """
    spread_bps_x = np.expm1(xg - xg.mean()) * 1e4
    spread_bps_y = np.expm1(yg - yg.mean()) * 1e4
    kw_extent = dict(
        origin='lower', aspect='auto',
        extent=[spread_bps_x[0], spread_bps_x[-1],
                spread_bps_y[0], spread_bps_y[-1]],
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.2))
    im0 = ax.imshow(phi_2d, interpolation='bilinear', cmap='Blues', **kw_extent)
    plt.colorbar(im0, ax=ax, label='φ(0, x, y)')
    ax.set_xlabel('log S^C (bps from center)')
    ax.set_ylabel('log S^D (bps from center)')
    ax.set_title('2D value function φ(x, y)')
    ax.grid(True, alpha=0.22, linestyle=':', linewidth=0.6)
    plt.tight_layout()
    return fig

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from arb.config import OUTPUT_DIR


def plot_hft_stylized_facts_part1(px, rho_param):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    lp = np.log(px['cex_close'])
    trend = (lp - lp.shift(60 * 24)).dropna() / (60 * 24)
    axes[0].hist(trend, bins=60, color='steelblue', edgecolor='white')
    axes[0].axvline(0, color='red', lw=1.5)
    axes[0].set_title('A1: drift proxy (1m)')

    ret = px['ret_cex'].dropna() * 100
    axes[1].hist(ret.clip(-2, 2), bins=80, density=True, color='steelblue', alpha=0.7)
    xx = np.linspace(-2, 2, 200)
    axes[1].plot(xx, stats.norm.pdf(xx, ret.mean(), ret.std()), 'r-', lw=2)
    axes[1].set_title('A2: CEX 1m returns (%)')

    stats.probplot(ret.clip(-3, 3), plot=axes[2])
    axes[2].set_title('A3: QQ-plot')
    plt.tight_layout()
    return fig


def plot_hft_stylized_facts_part2(px, arb, rho_param):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(arb['abs_spread_bps'].clip(upper=50), bins=60, color='darkorange')
    axes[0].set_title('A4: spread (arb swaps)')
    axes[1].hist(
        arb['gas_price_gwei'].clip(upper=arb['gas_price_gwei'].quantile(0.99)),
        bins=60,
        color='green',
    )
    axes[1].set_title('A5: gas (arb txs)')
    step = max(1, len(px) // 5000)
    axes[2].scatter(
        px['ret_cex'].iloc[::step] * 100,
        px['ret_dex'].iloc[::step] * 100,
        alpha=0.15,
        s=6,
    )
    axes[2].set_title(f'A6: CEX–DEX (ρ={rho_param:.3f})')
    plt.tight_layout()
    return fig


def plot_hft_gas_alignment(arb_bids, cap_q=0.97):
    cap = float(arb_bids['gas_price_gwei'].quantile(cap_q))
    g = arb_bids['gas_price_gwei'].to_numpy(dtype=float)
    g = g[np.isfinite(g)]
    x_hi = max(3.0, min(cap, float(np.percentile(g, 99.0)), 5.0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(g[g <= x_hi], bins=56, range=(0, x_hi), alpha=0.55, color='black', density=True)
    ax.axvline(arb_bids['g_cf'].median(), color='steelblue', lw=2, label='CF')
    if arb_bids['g_hjb'].notna().any():
        ax.axvline(arb_bids['g_hjb'].median(), color='green', lw=2, ls='--', label='HJB')
    ax.axvline(arb_bids['g_naive'].iloc[0], color='red', lw=2, ls=':', label='Naive')
    ax.set(xlabel='Gas (gwei)', ylabel='Density', title='Model vs on-chain gas')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_hft_fig15_hjb_1d(u_hjb, phi_hjb, g_star_hjb, g_lo, naive_gwei):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(u_hjb * 10000, phi_hjb)
    axes[0].set(xlabel='u (bps)', ylabel='φ', title='Value function')
    axes[1].plot(u_hjb * 10000, g_star_hjb, color='green', label='HJB')
    axes[1].axhline(g_lo, color='red', ls=':', label='floor')
    axes[1].axhline(naive_gwei, color='orange', ls='--', label='naive')
    axes[1].set(xlabel='spread (bps)', ylabel='g* (gwei)', title='Gas policy')
    axes[1].legend()
    plt.tight_layout()
    return fig


def save_hft_figure(fig, stem):
    path = OUTPUT_DIR / 'figures' / f'{stem}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    return path

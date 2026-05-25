import numpy as np
import pandas as pd

from arb.config import GAS_USED

from .config_hft import MIN_AMOUNT_ETH, MIN_SPREAD_BPS


def match_cex_price(swaps, cex_1m):
    swaps = swaps.copy()
    swaps['dt_1m'] = swaps['dt'].dt.floor('1min')
    cex_close = cex_1m[['close']].rename(columns={'close': 'cex_price'})
    df = swaps.merge(cex_close, left_on='dt_1m', right_index=True, how='left')
    df = df.sort_values('timestamp')
    df['cex_price'] = df['cex_price'].ffill(limit=5)
    df['spread_usd'] = df['cex_price'] - df['dex_price']
    df['spread_bps'] = df['spread_usd'] / df['cex_price'].replace(0, np.nan) * 10_000
    return df.drop(columns=['dt_1m'])


def classify_arb_swaps(
    swaps,
    cex_1m,
    min_spread_bps=MIN_SPREAD_BPS,
    min_amount_eth=MIN_AMOUNT_ETH,
):
    df = match_cex_price(swaps, cex_1m)
    df['is_correcting'] = (
        ((df['direction'] == 1) & (df['spread_usd'] > 0))
        | ((df['direction'] == -1) & (df['spread_usd'] < 0))
    )
    df['abs_spread_bps'] = df['spread_bps'].abs()
    df['is_arb'] = (
        df['is_correcting']
        & (df['abs_spread_bps'] >= min_spread_bps)
        & (df['amount_eth'] >= min_amount_eth)
        & df['cex_price'].notna()
        & df['dex_price'].notna()
        & (df['dex_price'] > 0)
    )
    df['gross_pnl_usd'] = df['spread_usd'].abs() * df['amount_eth']
    df['gas_cost_eth'] = GAS_USED * df['gas_price_gwei'] * 1e-9
    df['gas_cost_usd'] = df['gas_cost_eth'] * df['cex_price']
    df['net_pnl_usd'] = df['gross_pnl_usd'] - df['gas_cost_usd']
    df['abs_spread_usd'] = df['spread_usd'].abs()
    return df[df['is_arb']].copy().reset_index(drop=True)


def describe_arb_swaps(arb):
    if arb.empty:
        return pd.DataFrame()
    stats = {
        'n_swaps': len(arb),
        'date_range': f'{arb["dt"].min():%Y-%m-%d} → {arb["dt"].max():%Y-%m-%d}',
        'mean_spread_bps': round(arb['abs_spread_bps'].mean(), 2),
        'median_spread_bps': round(arb['abs_spread_bps'].median(), 2),
        'mean_amount_eth': round(arb['amount_eth'].mean(), 2),
        'median_gas_gwei': round(arb['gas_price_gwei'].median(), 2),
        'mean_gas_gwei': round(arb['gas_price_gwei'].mean(), 2),
        'mean_gross_pnl': round(arb['gross_pnl_usd'].mean(), 2),
        'mean_gas_cost': round(arb['gas_cost_usd'].mean(), 2),
        'mean_net_pnl': round(arb['net_pnl_usd'].mean(), 2),
        'pct_profitable': round((arb['net_pnl_usd'] > 0).mean() * 100, 1),
    }
    return pd.Series(stats).to_frame('value')

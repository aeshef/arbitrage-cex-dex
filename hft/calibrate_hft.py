import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from arb.calibrate import calibrate_all
from arb.config import Q_ETH, DATA_DIR
from arb.data import fetch_block_range_parallel, load_block_history

from .config_hft import START_DATE, END_DATE


def build_px_hft(swaps, cex_1m):
    if swaps.empty or cex_1m.empty:
        return pd.DataFrame()

    dex_raw = (
        swaps.set_index('dt')['dex_price']
        .sort_index()
        .loc[lambda s: (s > 100) & (s < 1e6)]
    )
    dex_1m = dex_raw.resample('1min').last().ffill(limit=5)
    dex_1m.name = 'dex_mid'
    px = pd.concat([cex_1m['close'].rename('cex_close'), dex_1m], axis=1).dropna()
    if len(px) < 60:
        warnings.warn(f'build_px_hft: only {len(px)} aligned bars')
        return pd.DataFrame()

    px['ret_cex'] = np.log(px['cex_close']).diff()
    px['ret_dex'] = np.log(px['dex_mid']).diff()
    px['spread_usd'] = px['cex_close'] - px['dex_mid']
    px['spread_bps'] = px['spread_usd'] / px['cex_close'] * 10_000
    return px.dropna(subset=['ret_cex', 'ret_dex'])


def gas_oracle_from_blocks(blocks):
    bf_col = 'baseFeePerGas' if 'baseFeePerGas' in blocks.columns else 'baseFee_gwei'
    if bf_col not in blocks.columns or blocks.empty:
        return {'base_fee': 2.0, 'safe_gwei': 2.5, 'propose_gwei': 3.0, 'fast_gwei': 5.0}
    bf = blocks[bf_col].dropna()
    return {
        'base_fee': float(bf.median()),
        'safe_gwei': float(bf.quantile(0.50) + 0.1),
        'propose_gwei': float(bf.quantile(0.70) + 0.2),
        'fast_gwei': float(bf.quantile(0.90) + 0.5),
    }


def _prepare_swaps_for_slip(swaps):
    sw = swaps.sort_values('timestamp').copy()
    sw['exec_price'] = sw['dex_price']
    sw['amountUSD'] = sw['amount_usd']
    return sw.loc[sw['amount_usd'] > 100, ['exec_price', 'amountUSD']].dropna()


def calibrate_hft(swaps, cex_1m, blocks, Q=Q_ETH):
    px = build_px_hft(swaps, cex_1m)
    if px.empty:
        raise ValueError('empty panel — check swaps and CEX data')

    oracle = gas_oracle_from_blocks(blocks)
    slip_input = _prepare_swaps_for_slip(swaps)

    px_1h = px[['cex_close', 'dex_mid']].resample('1h').last().dropna()
    px_1h['ret_cex'] = np.log(px_1h['cex_close']).diff()
    px_1h['ret_dex'] = np.log(px_1h['dex_mid']).diff()
    px_1h['spread_usd'] = px_1h['cex_close'] - px_1h['dex_mid']
    px_1h['spread_bps'] = px_1h['spread_usd'] / px_1h['cex_close'] * 10_000
    px_1h = px_1h.dropna(subset=['ret_cex', 'ret_dex'])
    if len(px_1h) < 60:
        px_1h = px

    params = calibrate_all(px_1h, blocks, oracle, swaps=slip_input, Q=Q)
    params['data_freq'] = '1m'
    return params, px


def fetch_blocks_hft(start_date=START_DATE, end_date=END_DATE,
                     n_samples=500, verbose=True):
    cache_dir = DATA_DIR / 'hft_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'blocks_hft_{start_date}_{end_date}.parquet'
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    blocks = load_block_history(n_blocks=n_samples, use_cache=False)
    if blocks.empty:
        import datetime
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        days = (end - start).days
        anchor_date = os.environ.get('ETH_BLOCK_ANCHOR_DATE', '2026-01-01')
        anchor_block = int(os.environ.get('ETH_BLOCK_ANCHOR', '21700000'))
        blocks_per_day = int(os.environ.get('ETH_BLOCKS_PER_DAY', '7200'))
        anchor = datetime.datetime.strptime(anchor_date, '%Y-%m-%d')
        start_block = anchor_block + int((start - anchor).days * blocks_per_day)
        blocks = fetch_block_range_parallel(
            start_block,
            total_range=days * 7200,
            n_samples=min(n_samples, 500),
        )
    if blocks.empty:
        blocks = load_block_history(n_blocks=200, use_cache=True)

    if not blocks.empty:
        blocks.to_parquet(cache_file)
        if verbose:
            print(len(blocks))
    return blocks

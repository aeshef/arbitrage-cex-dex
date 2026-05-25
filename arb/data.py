import json
import time
import warnings
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    DATA_DIR,
    THE_GRAPH_API_KEY,
    ETHERSCAN_KEY,
    POOL_ADDRESS,
    CEX_SYMBOL,
    PERP_SYMBOL,
)

UA = {'User-Agent': 'cex-dex-arb/1.0'}
UNI3_SG = (
    f'https://gateway.thegraph.com/api/{THE_GRAPH_API_KEY}/subgraphs/id/'
    '5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV'
)
UNI3_SG_HOSTED = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'

_RPC_LIST = [
    'https://eth.llamarpc.com',
    'https://rpc.ankr.com/eth',
    'https://cloudflare-eth.com',
    'https://ethereum-rpc.publicnode.com',
    'https://1rpc.io/eth',
]


def _load_cache(name):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, reader in (
        ('.parquet', pd.read_parquet),
        ('.csv', lambda p: pd.read_csv(p, index_col=0, parse_dates=True)),
    ):
        path = DATA_DIR / f'{name}{suffix}'
        if not path.exists():
            continue
        df = reader(path)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        return df
    return None


def _save_cache(df, name):
    if df is None or df.empty:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / f'{name}.parquet')


def http_get(url, params=None, timeout=30, extra_headers=None):
    headers = {**UA, **(extra_headers or {})}
    qs = ('?' + urlencode(params)) if params else ''
    req = Request(url + qs, headers=headers, method='GET')
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def graphql_post(url, query, variables=None, timeout=60):
    payload = json.dumps({'query': query, 'variables': variables or {}}).encode()
    req = Request(
        url,
        data=payload,
        headers={**UA, 'Content-Type': 'application/json'},
        method='POST',
    )
    with urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    if 'errors' in out:
        raise RuntimeError(out['errors'])
    return out['data']


def _graphql_post_with_fallback(query, variables=None, timeout=60):
    err = None
    for endpoint in (UNI3_SG, UNI3_SG_HOSTED):
        try:
            return graphql_post(endpoint, query, variables, timeout)
        except Exception as exc:
            err = exc
            warnings.warn(f'GraphQL {endpoint[:48]} failed: {exc}')
    raise RuntimeError('all GraphQL endpoints failed') from err


def _rpc_call(rpc_url, method, params, timeout=10):
    payload = json.dumps({'jsonrpc': '2.0', 'method': method, 'params': params, 'id': 1}).encode()
    req = Request(
        rpc_url,
        data=payload,
        headers={**UA, 'Content-Type': 'application/json'},
        method='POST',
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def binance_klines(symbol=CEX_SYMBOL, interval='1m', limit=1000):
    rows = http_get(
        'https://api.binance.com/api/v3/klines',
        {'symbol': symbol, 'interval': interval, 'limit': limit},
    )
    df = pd.DataFrame(
        rows,
        columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_vol', 'trades', 'taker_base', 'taker_quote', '_',
        ],
    )
    df['ts'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    return df.set_index('ts')[['open', 'high', 'low', 'close', 'volume', 'trades']].astype(float)


def binance_perp_klines(symbol=PERP_SYMBOL, interval='1h', limit=500):
    rows = http_get(
        'https://fapi.binance.com/fapi/v1/klines',
        {'symbol': symbol, 'interval': interval, 'limit': limit},
    )
    df = pd.DataFrame(
        rows,
        columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_vol', 'trades', 'taker_base', 'taker_quote', '_',
        ],
    )
    df['ts'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    return df.set_index('ts')[['open', 'high', 'low', 'close', 'volume']].astype(float)


def binance_klines_paginated(
    symbol=CEX_SYMBOL,
    interval='1h',
    start_str='2021-01-01',
    end_str=None,
    max_retries=3,
    page_timeout=45,
    use_cache=True,
):
    end_key = pd.Timestamp(end_str, tz='UTC').strftime('%Y%m%d') if end_str else 'now'
    cache_key = f'cex_{symbol}_{interval}_{start_str.replace("-", "")}_to_{end_key}'
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached

    start_ms = int(pd.Timestamp(start_str, tz='UTC').timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_str, tz='UTC').timestamp() * 1000) if end_str else None
    all_rows = []
    while True:
        params = {'symbol': symbol, 'interval': interval, 'limit': 1000, 'startTime': start_ms}
        if end_ms:
            params['endTime'] = end_ms
        rows = None
        for attempt in range(max_retries):
            try:
                rows = http_get(
                    'https://api.binance.com/api/v3/klines',
                    params,
                    timeout=page_timeout,
                )
                break
            except OSError:
                if attempt == max_retries - 1:
                    rows = []
                time.sleep(1.5 ** attempt)
        if not rows:
            break
        all_rows.extend(rows)
        last_open_ms = rows[-1][0]
        if end_ms and last_open_ms >= end_ms:
            break
        start_ms = last_open_ms + 1
        if len(rows) < 1000:
            break
        time.sleep(0.12)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_vol', 'trades', 'taker_base', 'taker_quote', '_',
        ],
    )
    df['ts'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df = (
        df.set_index('ts')[['open', 'high', 'low', 'close', 'volume']]
        .astype(float)
        .sort_index()
    )
    df = df[~df.index.duplicated(keep='last')]
    if use_cache:
        _save_cache(df, cache_key)
    return df


POOL_HOURLY_Q = '''
query PoolHourly($pool: String!, $ts_gt: Int!) {
  poolHourDatas(
    where: { pool: $pool, periodStartUnix_gt: $ts_gt }
    orderBy: periodStartUnix orderDirection: asc
    first: 1000
  ) { periodStartUnix open high low close token0Price token1Price
      volumeUSD tvlUSD feesUSD liquidity sqrtPrice tick }
}
'''

POOL_SWAPS_Q = '''
query PoolSwaps($pool: String!, $ts_gt: BigInt!) {
  swaps(
    where: { pool: $pool, timestamp_gt: $ts_gt }
    orderBy: timestamp orderDirection: asc
    first: 1000
  ) { timestamp amount0 amount1 amountUSD sqrtPriceX96 tick logIndex
      transaction { blockNumber gasUsed gasPrice } }
}
'''


def load_pool_hourly(pool=POOL_ADDRESS, pages=30, use_cache=True):
    cache_key = f'dex_thegraph_hourly_{pool[:10]}'
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached

    rows = []
    ts_cursor = 0
    for _ in range(pages):
        data = _graphql_post_with_fallback(
            POOL_HOURLY_Q, {'pool': pool.lower(), 'ts_gt': ts_cursor}
        )
        chunk = data.get('poolHourDatas', [])
        rows.extend(chunk)
        if len(chunk) < 1000:
            break
        ts_cursor = int(chunk[-1]['periodStartUnix'])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['ts'] = pd.to_datetime(df['periodStartUnix'].astype(int), unit='s', utc=True)
    df = df.set_index('ts').sort_index()
    for col in (
        'open', 'high', 'low', 'close', 'token0Price', 'token1Price',
        'volumeUSD', 'tvlUSD', 'feesUSD', 'liquidity',
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'token0Price' in df.columns:
        df['dex_mid'] = df['token0Price']
    elif 'close' in df.columns:
        df['dex_mid'] = df['close']
    if use_cache:
        _save_cache(df, cache_key)
    return df


def load_pool_swaps(pool=POOL_ADDRESS, pages=50, use_cache=True):
    cache_key = f'dex_swaps_{pool[:10]}'
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached

    rows = []
    ts_cursor = 0
    for _ in range(pages):
        data = _graphql_post_with_fallback(
            POOL_SWAPS_Q, {'pool': pool.lower(), 'ts_gt': str(ts_cursor)}
        )
        chunk = data.get('swaps', [])
        rows.extend(chunk)
        if len(chunk) < 1000:
            break
        ts_cursor = int(chunk[-1]['timestamp'])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['ts'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
    df = df.set_index('ts').sort_index()
    for col in ('amount0', 'amount1', 'amountUSD'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'sqrtPriceX96' in df.columns:
        sqp = pd.to_numeric(df['sqrtPriceX96'], errors='coerce')
        p_raw = (sqp / 2**96) ** 2
        df['exec_price'] = 1e12 / p_raw.replace(0, np.nan)
    if 'transaction' in df.columns:
        df['blockNumber'] = df['transaction'].apply(
            lambda x: int(x['blockNumber'], 16)
            if isinstance(x, dict) and x.get('blockNumber') else np.nan
        )
        df['gasPrice_gwei'] = df['transaction'].apply(
            lambda x: int(x['gasPrice'], 16) / 1e9
            if isinstance(x, dict) and x.get('gasPrice') else np.nan
        )
    if use_cache:
        _save_cache(df, cache_key)
    return df


def build_dex_1min_from_swaps(swaps_df):
    if swaps_df.empty or 'exec_price' not in swaps_df.columns:
        return pd.DataFrame()
    sw = swaps_df[['exec_price']].copy()
    sw['exec_price'] = pd.to_numeric(sw['exec_price'], errors='coerce')
    sw = sw[(sw['exec_price'] > 100) & (sw['exec_price'] < 1e6)].sort_index()
    if sw.empty:
        return pd.DataFrame()
    p = sw['exec_price']
    dex_1m = pd.DataFrame({
        'open': p.resample('1min').first(),
        'high': p.resample('1min').max(),
        'low': p.resample('1min').min(),
        'close': p.resample('1min').last(),
        'n_swaps': p.resample('1min').count(),
    })
    dex_1m['dex_mid'] = (dex_1m['open'] + dex_1m['close']) / 2
    dex_1m['dex_mid'] = dex_1m['dex_mid'].ffill()
    return dex_1m.dropna(subset=['dex_mid'])


def load_gecko_hourly(
    pool_network='eth',
    pool_address=POOL_ADDRESS,
    limit=1000,
    pages=6,
    use_cache=True,
):
    cache_key = f'dex_gecko_hourly_{pool_address[:10]}_p{pages}'
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached

    url = (
        f'https://api.geckoterminal.com/api/v2/networks/{pool_network}'
        f'/pools/{pool_address}/ohlcv/hour'
    )
    hdrs = {'Accept': 'application/json;version=20230302'}
    all_raw = []
    before_ts = None
    for _ in range(pages):
        params = {'limit': min(limit, 1000), 'currency': 'usd'}
        if before_ts is not None:
            params['before_timestamp'] = before_ts
        resp = http_get(url, params, extra_headers=hdrs)
        raw = resp['data']['attributes']['ohlcv_list']
        if not raw:
            break
        all_raw.extend(raw)
        before_ts = raw[-1][0]
        if len(raw) < min(limit, 1000):
            break
        time.sleep(0.3)

    if not all_raw:
        return pd.DataFrame()

    df = pd.DataFrame(all_raw, columns=['ts_unix', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts_unix'], unit='s', utc=True)
    df = df.set_index('ts').sort_index().drop_duplicates()
    df['dex_mid'] = df['close'].astype(float)
    if use_cache:
        _save_cache(df, cache_key)
    return df


def etherscan_gas_oracle():
    data = http_get(
        'https://api.etherscan.io/v2/api',
        {
            'chainid': '1',
            'module': 'gastracker',
            'action': 'gasoracle',
            'apikey': ETHERSCAN_KEY,
        },
    )
    r = data.get('result', {})
    return {
        'safe_gwei': float(r.get('SafeGasPrice', 5.0)),
        'propose_gwei': float(r.get('ProposeGasPrice', 8.0)),
        'fast_gwei': float(r.get('FastGasPrice', 15.0)),
        'base_fee': float(r.get('suggestBaseFee', 4.0)),
    }


def load_block_history(n_blocks=200, use_cache=True):
    cache_key = f'blocks_recent_{n_blocks}'
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached

    latest = None
    for rpc in _RPC_LIST:
        try:
            r = _rpc_call(rpc, 'eth_blockNumber', [])
            latest = int(r['result'], 16)
            break
        except OSError:
            continue
    if latest is None:
        return pd.DataFrame()

    records = []
    step = max(1, n_blocks // 150)
    for blk in range(latest - n_blocks, latest, step):
        for rpc in _RPC_LIST:
            try:
                r = _rpc_call(rpc, 'eth_getBlockByNumber', [hex(blk), False])
                res = r.get('result') or {}
                if res and 'baseFeePerGas' in res:
                    records.append({
                        'block': blk,
                        'baseFeePerGas': int(res['baseFeePerGas'], 16) / 1e9,
                        'timestamp': int(res['timestamp'], 16),
                    })
                    break
            except OSError:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index('block')
    df['ts'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    if use_cache:
        _save_cache(df, cache_key)
    return df


def _fetch_one_block(blk, rpc_list=_RPC_LIST):
    payload = {
        'jsonrpc': '2.0',
        'method': 'eth_getBlockByNumber',
        'params': [hex(blk), False],
        'id': blk,
    }
    data = json.dumps(payload).encode()
    for rpc in rpc_list:
        try:
            req = Request(
                rpc,
                data=data,
                headers={**UA, 'Content-Type': 'application/json'},
                method='POST',
            )
            with urlopen(req, timeout=12) as resp:
                res = json.loads(resp.read().decode()).get('result', {})
            if res and 'baseFeePerGas' in res:
                return {
                    'block': blk,
                    'baseFee_gwei': int(res['baseFeePerGas'], 16) / 1e9,
                    'timestamp': int(res['timestamp'], 16),
                }
        except OSError:
            continue
    return None


def fetch_block_range_parallel(
    start_block,
    total_range=200_000,
    n_samples=300,
    n_workers=10,
    use_cache=True,
):
    cache_key = f'blocks_{start_block}_{total_range}_{n_samples}'
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached

    step = max(1, total_range // n_samples)
    blks = list(range(start_block, start_block + total_range, step))
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_fetch_one_block, b): b for b in blks}
        for fut in as_completed(futs):
            row = fut.result()
            if row:
                results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values('block').reset_index(drop=True)
    df['ts'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    if use_cache:
        _save_cache(df, cache_key)
    return df


def build_aligned(cex_h, dex_h, perp_h=None):
    df = pd.DataFrame(index=cex_h.index)
    df['cex_close'] = cex_h['close']
    assert not dex_h.empty and 'dex_mid' in dex_h.columns
    df = df.join(dex_h[['dex_mid']], how='left')
    for col in ('liquidity', 'tvlUSD', 'volumeUSD'):
        if col in dex_h.columns:
            df = df.join(dex_h[[col]], how='left')
    if perp_h is not None and not perp_h.empty:
        df = df.join(perp_h[['close']].rename(columns={'close': 'perp_close'}), how='left')
    df = df.dropna(subset=['cex_close'])
    df['dex_mid'] = df['dex_mid'].replace(0, np.nan).ffill(limit=3).bfill(limit=3)
    df['cex_close'] = df['cex_close'].replace(0, np.nan)
    df = df.dropna(subset=['dex_mid', 'cex_close'])
    df['ret_cex'] = np.log(df['cex_close']).diff()
    df['ret_dex'] = np.log(df['dex_mid']).diff()
    if 'perp_close' in df.columns:
        df['ret_perp'] = np.log(df['perp_close']).diff()
    df['spread_usd'] = df['cex_close'] - df['dex_mid']
    df['spread_bps'] = 1e4 * df['spread_usd'] / df['cex_close'].replace(0, np.nan)
    q75 = df['spread_usd'].abs().rolling(24).quantile(0.75)
    df['is_opp'] = df['spread_usd'].abs() > q75
    return df

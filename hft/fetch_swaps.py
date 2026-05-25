import datetime
import os
import time

import numpy as np
import pandas as pd
import requests

from .config_hft import (
    CACHE_DIR,
    GRAPH_GATEWAY,
    GRAPH_HOSTED,
    POOL_ADDRESS,
    TOKEN0_DECIMALS,
    TOKEN1_DECIMALS,
)

_QUERY = """
query Swaps($pool: String!, $ts_lo: BigInt!, $ts_hi: BigInt!, $last_id: ID!) {
  swaps(
    first: 1000
    orderBy: id
    orderDirection: asc
    where: {
      pool: $pool
      timestamp_gte: $ts_lo
      timestamp_lt:  $ts_hi
      id_gt:         $last_id
    }
  ) {
    id
    timestamp
    amount0
    amount1
    amountUSD
    sqrtPriceX96
    tick
    sender
    origin
    transaction { id gasUsed gasPrice }
  }
}
"""

_DEC_ADJUST = 10 ** (TOKEN1_DECIMALS - TOKEN0_DECIMALS)


def _sqrt_price_to_eth_usd(sqrt_x96):
    sq = int(sqrt_x96)
    price_raw = (sq / (2 ** 96)) ** 2
    if price_raw == 0:
        return float('nan')
    return _DEC_ADJUST / price_raw


def _gql(url, variables, timeout=30):
    r = requests.post(url, json={'query': _QUERY, 'variables': variables}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if 'errors' in j:
        return None
    return j.get('data')


def fetch_pool_swaps(start_date, end_date, pool=POOL_ADDRESS, verbose=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = CACHE_DIR / f'swaps_{pool[:8]}_{start_date}_{end_date}.parquet'
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    ts_lo = int(datetime.datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    ts_hi = int(datetime.datetime.strptime(end_date, '%Y-%m-%d').timestamp()) + 86400
    records = []
    last_id = ''
    page = 0
    while True:
        variables = {
            'pool': pool.lower(),
            'ts_lo': str(ts_lo),
            'ts_hi': str(ts_hi),
            'last_id': last_id,
        }
        data = _gql(GRAPH_GATEWAY, variables)
        if data is None:
            data = _gql(GRAPH_HOSTED, variables)
        if data is None:
            break

        swaps = data.get('swaps', [])
        if not swaps:
            break

        for s in swaps:
            tx = s.get('transaction', {})
            amt0 = float(s['amount0'])
            amt1 = float(s['amount1'])
            eth_price = _sqrt_price_to_eth_usd(s['sqrtPriceX96'])
            if np.isnan(eth_price) and amt1 != 0:
                eth_price = abs(amt0) / abs(amt1)
            records.append({
                'id': s['id'],
                'timestamp': int(s['timestamp']),
                'tx_hash': tx.get('id', ''),
                'dex_price': eth_price,
                'amount_eth': abs(amt1),
                'amount_usd': abs(float(s.get('amountUSD', 0))),
                'gas_used': int(tx.get('gasUsed', 0)),
                'gas_price_wei': int(tx.get('gasPrice', 0)),
                'direction': 1 if amt1 < 0 else -1,
                'sender': s.get('sender', ''),
                'origin': s.get('origin', ''),
            })

        page += 1
        last_id = swaps[-1]['id']
        if verbose and page % 10 == 0:
            print(page, len(records))
        if len(swaps) < 1000:
            break
        time.sleep(0.15)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df['gas_price_gwei'] = df['gas_price_wei'] / 1e9
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.to_parquet(cache_file, index=False)
    if verbose:
        print(len(df))
    return df

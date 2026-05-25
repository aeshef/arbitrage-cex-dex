import datetime
import os
import time

import pandas as pd
import requests

from .config_hft import BINANCE_KLINE_URL, CACHE_DIR, CEX_SYMBOL


def fetch_binance_1m(start_date, end_date, symbol=CEX_SYMBOL, verbose=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = CACHE_DIR / f'binance_1m_{symbol}_{start_date}_{end_date}.parquet'
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    ts_lo = int(datetime.datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    ts_hi = int(
        (datetime.datetime.strptime(end_date, '%Y-%m-%d').timestamp() + 86400) * 1000
    )
    rows = []
    cursor = ts_lo
    page = 0

    while cursor < ts_hi:
        params = {
            'symbol': symbol,
            'interval': '1m',
            'startTime': cursor,
            'endTime': min(cursor + 999 * 60_000, ts_hi),
            'limit': 1000,
        }
        r = requests.get(BINANCE_KLINE_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        for k in data:
            rows.append({
                'ts_ms': int(k[0]),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
            })
        page += 1
        cursor = int(data[-1][0]) + 60_000
        if verbose and page % 50 == 0:
            print(page, len(rows))
        if len(data) < 1000:
            break
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['dt'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True)
    df = df.set_index('dt').sort_index()
    df = df[~df.index.duplicated()]
    df.to_parquet(cache_file)
    if verbose:
        print(len(df))
    return df

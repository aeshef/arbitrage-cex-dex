from arb.config import DATA_DIR, POOL_ADDRESS, Q_ETH, THE_GRAPH_API_KEY, _env

GRAPH_GATEWAY = (
    f'https://gateway.thegraph.com/api/{THE_GRAPH_API_KEY}/subgraphs/id/'
    '5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV'
)
GRAPH_HOSTED = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'

TOKEN0_DECIMALS = 6
TOKEN1_DECIMALS = 18

START_DATE = _env('HFT_START_DATE', '2026-01-01')
END_DATE = _env('HFT_END_DATE', '2026-04-30')

CEX_SYMBOL = _env('CEX_SYMBOL', 'ETHUSDT')
BINANCE_INTERVAL = _env('BINANCE_INTERVAL', '1m')
BINANCE_KLINE_URL = 'https://api.binance.com/api/v3/klines'

MIN_SPREAD_BPS = float(_env('MIN_SPREAD_BPS', '3'))
MIN_AMOUNT_ETH = float(_env('MIN_AMOUNT_ETH', '1'))

CACHE_DIR = DATA_DIR / 'hft_cache'

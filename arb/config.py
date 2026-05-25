import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'config'
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'


def load_env_file(path=None):
    """Load ROOT/.env without overriding variables already in the environment."""
    path = Path(path) if path else ROOT / '.env'
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, val = line.partition('=')
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


load_env_file()


def _env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, '') else default


def _env_float(name, default=None):
    v = _env(name)
    return float(v) if v is not None else default


def _env_int(name, default=None):
    v = _env(name)
    return int(v) if v is not None else default


def _load_json_config(file_env, json_env, default_name):
    raw = _env(json_env)
    if raw:
        return json.loads(raw)
    path = Path(_env(file_env) or CONFIG_DIR / default_name)
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing {default_name}: set {file_env} or {json_env}, '
            f'or copy config/{default_name.replace(".json", ".example.json")} '
            f'to config/{default_name}'
        )
    return json.loads(path.read_text())


def load_gas_regimes():
    return _load_json_config('GAS_REGIMES_FILE', 'GAS_REGIMES_JSON', 'gas_regimes.json')


def fallback_regime(regimes):
    name = _env('GAS_REGIME_FALLBACK') or next(iter(regimes))
    if name not in regimes:
        raise KeyError(f'GAS_REGIME_FALLBACK={name!r} not in gas_regimes')
    return name, regimes[name]


THE_GRAPH_API_KEY = _env('THE_GRAPH_API_KEY', '')
ETHERSCAN_KEY = _env('ETHERSCAN_KEY', '')

POOL_ADDRESS = _env(
    'POOL_ADDRESS',
    '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',
)
CEX_SYMBOL = _env('CEX_SYMBOL', 'ETHUSDT')
PERP_SYMBOL = _env('PERP_SYMBOL', CEX_SYMBOL)

Q_ETH = _env_float('Q_ETH', 10.0)
GAS_USED = _env_int('GAS_USED', 150_000)
GAMMA = _env_float('GAMMA', 3.0)
NAIVE_GAS_FIELD = _env('NAIVE_GAS_FIELD', 'fast_gwei')

ROLL_H = _env_int('ROLL_H', 24)
WF_TRAIN = _env_int('WF_TRAIN', 14 * 24)
WF_TEST = _env_int('WF_TEST', 7 * 24)

N_Z_HJB = _env_int('N_Z_HJB', 2000)
N_T_HJB = _env_int('N_T_HJB', 3000)
N_X_HJB_XY = _env_int('N_X_HJB_XY', 200)
N_Y_HJB_XY = _env_int('N_Y_HJB_XY', 200)
N_T_HJB_XY = _env_int('N_T_HJB_XY', 1000)
G_SCAN = _env_int('G_SCAN', 150)
N_Z_HJB_HEAVY = _env_int('N_Z_HJB_HEAVY', 4000)
N_T_HJB_HEAVY = _env_int('N_T_HJB_HEAVY', 3000)
N_T_HJB_GLOB = _env_int('N_T_HJB_GLOB', 50)
T_HORIZON = _env_float('T_HORIZON', 1.0)

HJB_PDE_MODE = _env('HJB_PDE_MODE', 'both')
HJB_2D_ACTIVE = _env('HJB_2D_ACTIVE', 'true').lower() in ('1', 'true', 'yes')
HJB_CONVERGENCE_CHECK = _env('HJB_CONVERGENCE_CHECK', 'true').lower() in ('1', 'true', 'yes')

N_MC_PATHS = _env_int('N_MC_PATHS', 60_000)
_q_grid = _env('Q_GRID')
Q_GRID = [int(x) for x in _q_grid.split(',')] if _q_grid else [1, 2, 5, 10, 20]

SLIP_ALPHA_DEFAULT = _env_float('SLIP_ALPHA_DEFAULT', 4.60)
SLIP_BETA_DEFAULT = _env_float('SLIP_BETA_DEFAULT', 0.049)

GAS_REGIMES = load_gas_regimes()
SYNTH_REGIMES = GAS_REGIMES


def load_historical_blocks():
    return _load_json_config(
        'HISTORICAL_BLOCKS_FILE',
        'HISTORICAL_BLOCKS_JSON',
        'historical_blocks.json',
    )


NB_ARTIFACT_DIR = str(OUTPUT_DIR)

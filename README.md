# arbitrage-cex-dex

Код к ВКР (НИУ ВШЭ, 2026): арбитраж CEX–DEX (Uniswap v3 ETH/USDC, Binance), ставка газа через HJB и бэктест.

## Что где

| Путь | Содержание |
|------|------------|
| `arb/` | Часовые данные: загрузка, калибровка λ(g), HJB, бэктест |
| `hft/` | Минутные свопы + CEX, проверка на уровне сделок |
| `notebooks/pipeline.ipynb` | Основной часовой прогон |
| `notebooks/hft.ipynb` | HFT-ветка |
| `config/*.example.json` | Шаблоны режимов газа и блоков |

`data/` и `outputs/` — кэш и графики, в git не попадают.

## Установка

```bash
cd arbitrage-cex-dex
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp config/gas_regimes.example.json config/gas_regimes.json
cp config/historical_blocks.example.json config/historical_blocks.json
```

В `.env` — `THE_GRAPH_API_KEY`, `ETHERSCAN_KEY`. Остальное см. `.env.example`.

Запуск: из корня репозитория открыть ноутбук в Jupyter; в первой ячейке `sys.path` указывает на родительский каталог (`notebooks/` → корень проекта).

## Зависимости

`requirements.txt`

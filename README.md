# Buffered ETF Backtest

Backtest buffered ETF strategies (protection/cap and partial-gain) using daily price data and optional annual yield data.

## Usage

**Typical use** (daily price file + annual gain/yield file):

```bash
python main.py sp500-daily.csv sp500-annual-gain-yield-short.csv
```

Run with only the daily file (yield assumed 0):

```bash
python main.py sp500-daily.csv
```

Limit the backtest to a date range:

```bash
python main.py sp500-daily.csv -from 1990-01-01 -to 2020-12-31
```

Help:

```bash
python main.py --help
```

## Arguments

| Argument       | Required | Description |
|----------------|----------|-------------|
| `daily_file`   | Yes      | Path to daily price CSV. Accepts 2-column `date`,`price` or `date`,`adj_close`. Dates in `YYYY-MM-DD`. |
| `annual_yield` | No       | Path to annual gain/yield CSV with `year`, `pricegain`, and `yield` columns. If omitted or empty, yield is assumed 0. |
| `-from YYYY-MM-DD` | No   | Include only data on or after this date. |
| `-to YYYY-MM-DD`   | No   | Include only data on or before this date. |

## Tax treatment

Tax is applied **each year**, not at the end of the backtest. For every year, the dividend/yield component of that year’s return is multiplied by `(1 - tax_rate)` when computing the buy-and-hold outcome; price gain is not taxed. The default tax rate is set in `main.py` (`TAX_RATE`); pass a different rate via the script’s tax option when available.

#! /usr/bin/env python3

import argparse
import csv
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time  # Import the time module
from collections import defaultdict

# Constants
START_MONEY = 1
#TAX_RATE = 0.371
TAX_RATE = 0.0
DEFAULT_PERCENTILES = [25, 50, 75]
DEFAULT_SAMPLE_TIMES = 5

class AllYearsInfo:
    """Class to store and compare investment performance metrics."""
    def __init__(self, end_money: float, max_drawdown: float, gain_per: float, annualized: float, bucket_max_drawdown: float = None):
        self.end_money = end_money
        self.max_drawdown = max_drawdown
        self.gain_per = gain_per
        self.annualized = annualized
        self.bucket_max_drawdown = bucket_max_drawdown  # worst drawdown in percentile bucket (multiverse only)

    def __lt__(self, other):
        return self.end_money < other.end_money

    def __eq__(self, other):
        return self.end_money == other.end_money

class Investment:
    """Class to store and calculate investment results."""
    def __init__(self, start_money: float = 1, tax_rate: float = TAX_RATE):
        self.start_money = start_money
        self.tax_rate = tax_rate
        self.max_drawdown = 0
        self.max_money = 0

    def calc_one_year(self, old_money: float, data: dict, protection: float, cap: float) -> float:
        """Calculate investment results for one year with optional protection and cap.

        Args:
            old_money: Starting amount of money
            data: Dictionary containing price gain and yield data
            protection: Downside protection percentage
            cap: Maximum gain cap (-1 for no cap)

        Returns:
            float: New amount of money after calculations
        """
        buyhold_mon = old_money
        buyhold_mon *= data['pricegain'] + data['yield'] * (1 - self.tax_rate)

        if buyhold_mon < old_money:
            lost = 1 - buyhold_mon / old_money
            new_money = old_money - old_money * max(lost - protection, 0)
        elif cap != -1:
            gained = buyhold_mon / old_money - 1
            gained = min(gained, cap)
            new_money = old_money + old_money * gained
        else:
            new_money = buyhold_mon

        if new_money > self.max_money:
            self.max_money = new_money

        drawdown = 1 - new_money / self.max_money
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        return new_money

    def calc_one_year_partial_gain(self, old_money: float, data: dict, loss_threshold: float, gain_fraction: float) -> float:
        """Calculate investment results for one year with complete loss protection after threshold and partial gains.

        Args:
            old_money: Starting amount of money
            data: Dictionary containing price gain and yield data
            loss_threshold: Complete loss protection threshold (e.g., 0.15 means losses capped at 15%)
            gain_fraction: Fraction of buy-and-hold gains captured (e.g., 0.5 means 50% of gains)

        Returns:
            float: New amount of money after calculations
        """
        buyhold_mon = old_money
        buyhold_mon *= data['pricegain'] + data['yield'] * (1 - self.tax_rate)

        if buyhold_mon < old_money:
            # Loss scenario: complete protection after loss_threshold
            lost = 1 - buyhold_mon / old_money
            # If loss exceeds threshold, cap it at threshold
            capped_loss = min(lost, loss_threshold)
            new_money = old_money - old_money * capped_loss
        else:
            # Gain scenario: only capture gain_fraction of the gains
            # gain_fraction=-1 means "no cap" = 100% of gains (same as cap=-1 in calc_one_year)
            if gain_fraction == -1:
                new_money = buyhold_mon
            else:
                gained = buyhold_mon / old_money - 1
                partial_gain = gained * gain_fraction
                new_money = old_money + old_money * partial_gain

        if new_money > self.max_money:
            self.max_money = new_money

        drawdown = 1 - new_money / self.max_money
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        return new_money

    def calc_all_years(self, data: list, protection: float, cap: float, verbose: bool = False) -> AllYearsInfo:
        """Calculate investment results over all years with protection and cap buffered ETF.

        Args:
            data: List of year data dictionaries
            protection: Downside protection percentage
            cap: Maximum gain cap (-1 for no cap)
            verbose: If True, print gain/loss and max drawdown so far for each year (skip when called from multiverse)

        Returns:
            AllYearsInfo: Investment performance metrics
        """
        self.max_drawdown = 0
        self.max_money = 0

        cur_money = self.start_money
        if verbose:
            rows = []
        for i, year_data in enumerate(data):
            prev_money = cur_money
            cur_money = self.calc_one_year(cur_money, year_data, protection, cap)
            if verbose:
                year_gain_pct = (cur_money - prev_money) / prev_money * 100
                year_label = year_data.get('year', i + 1)
                rows.append((year_label, year_gain_pct, self.max_drawdown * 100))
        if verbose and rows:
            print('  {:>4}  {:>10}  {:>14}'.format('Year', 'Gain/Loss', 'Max DD so far'))
            print('  ' + '-' * 32)
            for year_label, year_gain_pct, max_dd in rows:
                print('  {:>4}  {:>+9.2f}%  {:>13.2f}%'.format(year_label, year_gain_pct, max_dd))

        gain = cur_money / self.start_money
        annualized_gain = pow(gain, 1 / (len(data) - 1)) - 1

        return AllYearsInfo(cur_money, self.max_drawdown, gain, annualized_gain)

    def calc_all_years_partial_gain(self, data: list, loss_threshold: float, gain_fraction: float, verbose: bool = False) -> AllYearsInfo:
        """Calculate investment results over all years with partial gain buffered ETF.

        Args:
            data: List of year data dictionaries
            loss_threshold: Complete loss protection threshold
            gain_fraction: Fraction of buy-and-hold gains captured
            verbose: If True, print gain/loss and max drawdown so far for each year (skip when called from multiverse)

        Returns:
            AllYearsInfo: Investment performance metrics
        """
        self.max_drawdown = 0
        self.max_money = 0

        cur_money = self.start_money
        if verbose:
            rows = []
        for i, year_data in enumerate(data):
            prev_money = cur_money
            cur_money = self.calc_one_year_partial_gain(cur_money, year_data, loss_threshold, gain_fraction)
            if verbose:
                year_gain_pct = (cur_money - prev_money) / prev_money * 100
                year_label = year_data.get('year', i + 1)
                rows.append((year_label, year_gain_pct, self.max_drawdown * 100))
        if verbose and rows:
            print('  {:>4}  {:>10}  {:>14}'.format('Year', 'Gain/Loss', 'Max DD so far'))
            print('  ' + '-' * 32)
            for year_label, year_gain_pct, max_dd in rows:
                print('  {:>4}  {:>+9.2f}%  {:>13.2f}%'.format(year_label, year_gain_pct, max_dd))

        gain = cur_money / self.start_money
        annualized_gain = pow(gain, 1 / (len(data) - 1)) - 1

        return AllYearsInfo(cur_money, self.max_drawdown, gain, annualized_gain)

def read_annual_data(filename: str) -> list:
    data = []

    with open(filename, mode='r') as file:
        csvfile = csv.DictReader(file)

        for line_dict in csvfile:
            line_dict = dict(line_dict)
            data.append({
                'year': int(line_dict['year']),
                'pricegain': float(line_dict['pricegain']) + 1,
                'yield': float(line_dict['yield'])
            })

    return data


def truncate_to_full_years(month_data: list) -> list:
    """Truncate monthly data to a multiple of 12 months (complete years).
    Drops the last several months so all data is used in year chunks.
    """
    n = (len(month_data) // 12) * 12
    return month_data[:n] if n > 0 else month_data


# Supports CSV with "date,price", "date,adj_close", or "Date,Open,High,Low,Close,Volume" columns
def read_monthly_data(filename: str) -> list:
    """Read monthly data from CSV file.

    Args:
        filename: Path to CSV with date and price column.
                 Accepts: "date","price"; "date","adj_close"; or "Date","Close" (OHLCV format).

    Returns:
        List of dictionaries with 'date' and 'gain' for all months
    """
    data = []

    with open(filename, mode='r') as file:
        csvfile = csv.DictReader(file)

        prev_adj_close = None
        prev_month = None
        last_day = None # Last day of the last month
        last_day_adj_close = None # price on 'last_day'
        for line_dict in csvfile:
            line_dict = dict(line_dict)
            # Support price, adj_close, or Close column; skip rows with empty price
            raw_price = (line_dict.get('price') or line_dict.get('adj_close') or line_dict.get('Close') or '').strip()
            if not raw_price:
                continue
            cur_adj_close = float(raw_price)
            date_str = (line_dict.get('date') or line_dict.get('Date') or '').strip()
            if not date_str:
                continue
            date = datetime.strptime(date_str, '%Y-%m-%d')
            if prev_month is None:
                prev_month = date.month
            elif date.month != prev_month:
                # only keep the last day of each month
                if prev_adj_close is not None:
                    gain = last_day_adj_close / prev_adj_close
                    data.append({
                        'date': last_day,
                        'gain': gain
                    })
                prev_adj_close = last_day_adj_close
                prev_month = date.month
            last_day_adj_close = cur_adj_close  # Update to the last day of the month
            last_day = date

        # Add the last month
        if last_day_adj_close is not None and prev_adj_close is not None:
            gain = last_day_adj_close / prev_adj_close
            data.append({
                'date': date,
                'gain': gain
            })

    return data

def month_to_year_data(month_data: list) -> list:
    """Convert monthly data to yearly data format by calculating yearly gains from 12-month periods.
    Doesn't alter the original list.

    Args:
        month_data: List of dictionaries with 'date' and 'gain' keys

    Returns:
        List of dictionaries with 'pricegain' and 'yield' keys
    """
    yearly_data = []

    # Process data in chunks of 12 months
    for i in range(0, len(month_data) - 11, 12):  # Step by 12, ensure we have 12 months left
        year_chunk = month_data[i:i+12]

        start_money = 1000
        money = start_money
        for month in year_chunk:
            money *= month['gain']

        price_gain = money / start_money

        yearly_data.append({
            'pricegain': price_gain,
            'yield': 0.0
        })

    return yearly_data

def calc_multiverse_sample(month_data: list, protection: float, cap: float) -> AllYearsInfo:
    """Helper function to calculate a single sample for calc_multiverse.
    Expects month_data to already have yield applied (do not shuffle before yield).
    """
    random.shuffle(month_data)
    year_data = month_to_year_data(month_data)
    invest = Investment(START_MONEY, TAX_RATE)
    return invest.calc_all_years(year_data, protection, cap)

def calc_multiverse(month_data: list, annual_data: list, protection: float, cap: float,
                    sample_times: int = 5, want: list = DEFAULT_PERCENTILES,
                    tax_rate: float = TAX_RATE) -> list:
    """Calculate multiverse results. Yield is applied to monthly data first, before any re-ordering."""
    # Apply yield to monthly data BEFORE any shuffling
    month_with_yield = adjust_monthly_gain_with_yield(month_data, annual_data, tax_rate)
    month_with_yield = month_with_yield.copy()  # copy to avoid modifying when workers shuffle

    end_moneys = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calc_multiverse_sample, month_with_yield, protection, cap) for _ in range(sample_times)]
        for future in as_completed(futures):
            end_moneys.append(future.result())

    end_moneys.sort()
    out = []

    for per in want:
        nth = min(int(per / 100 * sample_times), sample_times - 1)
        # Bucket for this percentile: [per-25, per] so 50→25–50%, 75→50–75%, etc
        start_idx = max(0, int((per - 25) / 100 * sample_times))
        end_idx = max(start_idx + 1, int(per / 100 * sample_times))
        end_idx = min(end_idx, sample_times)
        bucket = end_moneys[start_idx:end_idx]
        avg_drawdown = sum(s.max_drawdown for s in bucket) / len(bucket)
        worst_drawdown = max(s.max_drawdown for s in bucket)
        rep = end_moneys[nth]
        out.append(AllYearsInfo(rep.end_money, avg_drawdown, rep.gain_per, rep.annualized, bucket_max_drawdown=worst_drawdown))

    return out

def calc_multiverse_sample_partial_gain(month_data: list, loss_threshold: float, gain_fraction: float) -> AllYearsInfo:
    """Helper function to calculate a single sample for calc_multiverse_partial_gain.
    Expects month_data to already have yield applied (do not shuffle before yield).
    """
    random.shuffle(month_data)
    year_data = month_to_year_data(month_data)
    invest = Investment(START_MONEY, TAX_RATE)
    return invest.calc_all_years_partial_gain(year_data, loss_threshold, gain_fraction)

def calc_multiverse_partial_gain(month_data: list, annual_data: list, loss_threshold: float, gain_fraction: float,
                                 sample_times: int = 5, want: list = DEFAULT_PERCENTILES,
                                 tax_rate: float = TAX_RATE) -> list:
    """Calculate multiverse results for partial gain buffered ETF.
    Yield is applied to monthly data first, before any re-ordering.

    Args:
        month_data: List of monthly data dictionaries
        annual_data: Annual data with pricegain and yield for yield adjustment
        loss_threshold: Complete loss protection threshold
        gain_fraction: Fraction of buy-and-hold gains captured
        sample_times: Number of random samples to generate
        want: List of percentiles to return

    Returns:
        List of AllYearsInfo objects for requested percentiles
    """
    # Apply yield to monthly data BEFORE any shuffling
    month_with_yield = adjust_monthly_gain_with_yield(month_data, annual_data, tax_rate)
    month_with_yield = month_with_yield.copy()  # copy to avoid modifying when workers shuffle

    end_moneys = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calc_multiverse_sample_partial_gain, month_with_yield, loss_threshold, gain_fraction) for _ in range(sample_times)]
        for future in as_completed(futures):
            end_moneys.append(future.result())

    end_moneys.sort()
    out = []

    for per in want:
        nth = min(int(per / 100 * sample_times), sample_times - 1)
        # Bucket for this percentile: [per-25, per] so 50→25–50%, 75→50–75%, etc
        start_idx = max(0, int((per - 25) / 100 * sample_times))
        end_idx = max(start_idx + 1, int(per / 100 * sample_times))
        end_idx = min(end_idx, sample_times)
        bucket = end_moneys[start_idx:end_idx]
        avg_drawdown = sum(s.max_drawdown for s in bucket) / len(bucket)
        worst_drawdown = max(s.max_drawdown for s in bucket)
        rep = end_moneys[nth]
        out.append(AllYearsInfo(rep.end_money, avg_drawdown, rep.gain_per, rep.annualized, bucket_max_drawdown=worst_drawdown))

    return out

def calc_and_print(data: list, protection: float, cap: float, tax_rate: float = TAX_RATE, verbose: bool = True):
    print_cap = 'no' if cap == -1 else f'{cap * 100:.3f}%'
    print(f'\n*** Results for {protection * 100:.3f}% protection, {print_cap} cap ***')

    investment = Investment(START_MONEY, tax_rate)
    info = investment.calc_all_years(data, protection, cap, verbose=verbose)

    #print(f'Starting money: ${START_MONEY}\nEnding money: ${end_money:.2f}')
    print(f'Max drawdown: {info.max_drawdown * 100:.3f}%\nAnnualized gain: {info.annualized * 100:.3f}%')

def calc_and_print_partial_gain(data: list, loss_threshold: float, gain_fraction: float, tax_rate: float = TAX_RATE, verbose: bool = True):
    """Calculate and print results for partial gain buffered ETF.

    Args:
        data: List of year data dictionaries
        loss_threshold: Complete loss protection threshold (e.g., 0.15 for 15%)
        gain_fraction: Fraction of buy-and-hold gains captured (e.g., 0.5 for 50%)
        tax_rate: Tax rate for yield calculations
        verbose: If True, print gain/loss and max drawdown for each year
    """
    print(f'\n*** Results for {loss_threshold * 100:.3f}% loss threshold, {gain_fraction * 100:.3f}% of gains ***')

    investment = Investment(START_MONEY, tax_rate)
    info = investment.calc_all_years_partial_gain(data, loss_threshold, gain_fraction, verbose=verbose)

    print(f'Max drawdown: {info.max_drawdown * 100:.3f}%\nAnnualized gain: {info.annualized * 100:.3f}%')

def adjust_monthly_gain_with_yield(data_no_div: list, data_with_div: list, tax_rate: float) -> list:
    """Adjust the monthly gain in data_no_div with the yield in data_with_div.

    Args:
        data_no_div: List of dictionaries with 'date' and 'gain' keys (monthly data)
        data_with_div: List of dictionaries with 'pricegain' and 'yield' keys (annual data)

    Returns:
        List of dictionaries with adjusted 'gain' values
    """
    adjusted_data = data_no_div.copy()
    year_index = 0

    for i in range(len(adjusted_data)):
        month = adjusted_data[i]['date'].month
        year = adjusted_data[i]['date'].year

        if year_index < len(data_with_div) and year == adjusted_data[i]['date'].year:
            annual_gain = data_with_div[year_index]['pricegain']
            annual_yield = data_with_div[year_index]['yield']

            annual_yield_ratio = (annual_gain + annual_yield * (1 - tax_rate)) / annual_gain
            monthly_yield = annual_yield_ratio ** (1 / 12)
            adjusted_data[i]['gain'] *= monthly_yield

        if month == 12:  # Move to the next year after December
            year_index += 1

    return adjusted_data

def print_price_gain_with_yield(data):
    for i in range(len(data)):
        gain = data[i]['pricegain']
        yield_ = data[i]['yield']
        print(f'{i}: {gain} {yield_} {gain + yield_ * (1 - TAX_RATE)}')

def parse_date_arg(value: str, is_from: bool) -> tuple:
    """Parse -from or -to argument. Year-only (e.g. 1980) becomes 1980-01-01 (from) or 1980-12-31 (to).

    Returns:
        tuple: (datetime, normalized_str) for use in filtering
    """
    if not value:
        return (datetime.min, None) if is_from else (datetime.max, None)
    value = value.strip()
    if len(value) == 4 and value.isdigit():
        year = int(value)
        if is_from:
            dt = datetime(year, 1, 1)
            norm = f'{year}-01-01'
        else:
            dt = datetime(year, 12, 31)
            norm = f'{year}-12-31'
        return (dt, norm)
    dt = datetime.strptime(value, '%Y-%m-%d')
    return (dt, value)


def filter_by_date_range(month_data: list, from_date: datetime, to_date: datetime) -> list:
    """Filter monthly data to only include months within the given date range.

    Args:
        month_data: List of dicts with 'date' (last day of month) and 'gain'
        from_date: Inclusive start date (YYYY-MM-DD)
        to_date: Inclusive end date (YYYY-MM-DD)

    Returns:
        Filtered list of monthly data
    """
    return [m for m in month_data if from_date <= m['date'] <= to_date]


def filter_annual_by_year_range(annual_data: list, from_year: int, to_year: int) -> list:
    """Filter annual data to only include years within the given range.

    Args:
        annual_data: List of dicts with 'year', 'pricegain', 'yield'
        from_year: Inclusive start year
        to_year: Inclusive end year

    Returns:
        Filtered list of annual data
    """
    return [a for a in annual_data if from_year <= a['year'] <= to_year]


def _get_col(line_dict: dict, *candidates: str) -> str:
    """Get value from dict with case-insensitive key match."""
    keys_lower = {k.lower(): k for k in line_dict}
    for c in candidates:
        if c.lower() in keys_lower:
            return line_dict[keys_lower[c.lower()]].strip()
    return ''


def read_daily_ohlc(filename: str) -> list:
    """Read daily OHLC data from CSV file.

    Expects columns: date, open/Open, High/high, Low/low, Close/adj_close.

    Returns:
        List of dicts with 'date', 'open', 'high', 'low', 'close'
    """
    data = []
    with open(filename, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            date_str = (row.get('date') or row.get('Date') or '').strip()
            if not date_str:
                continue
            close_val = _get_col(row, 'adj_close', 'Close', 'close')
            if not close_val:
                continue
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                open_val = float(_get_col(row, 'open', 'Open') or close_val)
                high_val = float(_get_col(row, 'High', 'high') or close_val)
                low_val = float(_get_col(row, 'Low', 'low') or close_val)
                close_val = float(close_val)
            except (ValueError, TypeError):
                continue
            data.append({
                'date': dt,
                'open': open_val,
                'high': high_val,
                'low': low_val,
                'close': close_val,
            })
    return data


def aggregate_ohlc(daily_data: list, freq: str) -> list:
    """Aggregate daily OHLC to monthly, quarterly, or annual bars.

    Args:
        daily_data: List of dicts with 'date', 'open', 'high', 'low', 'close'
        freq: 'monthly', 'quarterly', or 'annual'

    Returns:
        List of dicts with 'date', 'open', 'high', 'low', 'close'
    """
    if not daily_data:
        return []
    daily_data = sorted(daily_data, key=lambda x: x['date'])
    groups = defaultdict(list)
    for d in daily_data:
        dt = d['date']
        if freq == 'monthly':
            key = (dt.year, dt.month)
        elif freq == 'quarterly':
            q = (dt.month - 1) // 3 + 1
            key = (dt.year, q)
        else:
            key = (dt.year,)
        groups[key].append(d)
    result = []
    for key in sorted(groups.keys()):
        bars = groups[key]
        first, last = bars[0], bars[-1]
        result.append({
            'date': last['date'],
            'open': first['open'],
            'high': max(b['high'] for b in bars),
            'low': min(b['low'] for b in bars),
            'close': last['close'],
        })
    return result


def filter_daily_by_date_range(daily_data: list, from_date: datetime, to_date: datetime) -> list:
    """Filter daily OHLC data to the given date range."""
    return [d for d in daily_data if from_date <= d['date'] <= to_date]


def _apply_yield_to_ohlc(ohlc: list, annual_data: list, freq: str, tax_rate: float) -> list:
    """Convert price OHLC to total-return (dividend-adjusted) OHLC.

    Uses cumulative total return index. O, H, L, C are scaled so close includes dividends.
    """
    if not ohlc or not annual_data:
        return ohlc
    year_to_annual = {a['year']: a for a in annual_data}
    base = ohlc[0]['close']
    cum_tr = [base]
    for i in range(1, len(ohlc)):
        bar = ohlc[i]
        prev_bar = ohlc[i - 1]
        year = bar['date'].year
        ann = year_to_annual.get(year)
        if ann:
            total_return = ann['pricegain'] + ann['yield'] * (1 - tax_rate)
            if freq == 'annual':
                tr_gain = total_return
            else:
                price_gain = bar['close'] / prev_bar['close']
                annual_yield_ratio = total_return / ann['pricegain']
                adj = annual_yield_ratio ** (1 / 12) if freq == 'monthly' else annual_yield_ratio ** (1 / 4)
                tr_gain = price_gain * adj
        else:
            tr_gain = bar['close'] / prev_bar['close']
        cum_tr.append(cum_tr[-1] * tr_gain)
    result = []
    for i, bar in enumerate(ohlc):
        o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
        scale = cum_tr[i] / c if c > 0 else 1
        result.append({
            'date': bar['date'],
            'open': cum_tr[i - 1] if i > 0 else base,
            'high': h * scale,
            'low': l * scale,
            'close': cum_tr[i],
        })
    return result


def plot_price_candlestick(daily_file: str, from_date: datetime, to_date: datetime,
                          annual_yield_file: str = '', tax_rate: float = TAX_RATE,
                          ymin: float = None, ymax: float = None) -> None:
    """Plot price OHLC candles for the date range.

    Uses monthly data if range <= 20 years, quarterly if > 20 and <= 50 years,
    annual if > 50 years. X-axis: year. Y-axis: price relative to close of first
    period (indexed to 1.0). Green=up, Red=down.
    When annual_yield_file is provided, plots total return (price + dividends).
    Use -ymin and -ymax to set y-axis limits (in relative units, e.g. 0.8 = 80%% of first close).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    daily_data = read_daily_ohlc(daily_file)
    if not daily_data:
        print('Error: No OHLC data found in file.')
        return
    daily_data = filter_daily_by_date_range(daily_data, from_date, to_date)
    if not daily_data:
        print('Error: No data in the specified date range.')
        return

    years_span = (to_date.year - from_date.year) + (to_date.month - from_date.month) / 12 + (to_date.day - from_date.day) / 365
    if years_span > 50:
        freq = 'annual'
    elif years_span > 20:
        freq = 'quarterly'
    else:
        freq = 'monthly'

    ohlc = aggregate_ohlc(daily_data, freq)
    with_dividends = False
    if annual_yield_file.strip():
        annual_data = read_annual_data(annual_yield_file.strip())
        from_year = from_date.year
        to_year = to_date.year
        annual_data = filter_annual_by_year_range(annual_data, from_year, to_year)
        if annual_data:
            ohlc = _apply_yield_to_ohlc(ohlc, annual_data, freq, tax_rate)
            with_dividends = True

    # Scale y-axis relative to close price of first period being plotted
    first_close = ohlc[0]['close']
    ohlc_rel = []
    for bar in ohlc:
        ohlc_rel.append({
            'date': bar['date'],
            'open': bar['open'] / first_close,
            'high': bar['high'] / first_close,
            'low': bar['low'] / first_close,
            'close': bar['close'] / first_close,
        })
    ohlc = ohlc_rel

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Price (relative to first {freq} close = 1.0)')
    title = f'Price + dividends ({freq})' if with_dividends else f'Price ({freq})'
    ax.set_title(f'{title} — {from_date.strftime("%Y-%m-%d")} to {to_date.strftime("%Y-%m-%d")}')

    width = 0.8
    for i, bar in enumerate(ohlc):
        x = i
        o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
        color = 'green' if c >= o else 'red'
        # Wick from low to high
        ax.plot([x, x], [l, h], color=color, linewidth=1)
        # Body
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < (h - l) * 0.01:
            body_height = (h - l) * 0.02
            body_bottom = (o + c) / 2 - body_height / 2
        ax.add_patch(Rectangle((x - width / 2, body_bottom), width, body_height,
                               facecolor=color, edgecolor=color))

    if ohlc:
        year_to_first_idx = {}
        for i, bar in enumerate(ohlc):
            y = bar['date'].year
            if y not in year_to_first_idx:
                year_to_first_idx[y] = i
        indices = sorted(year_to_first_idx.values())
        ax.set_xticks(indices)
        ax.set_xticklabels([ohlc[i]['date'].strftime('%Y') for i in indices])

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    else:
        ax.autoscale()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Buffered ETF backtest')
    parser.add_argument('daily_file', help='Path to daily (or monthly) price CSV file')
    parser.add_argument('annual_yield', nargs='?', default='',
                        help='Path to annual gain/yield CSV file; if omitted or empty, yield is assumed 0')
    parser.add_argument('-from', '--from-date', dest='from_date', metavar='YYYY[-MM-DD]',
                        help='Include only data on or after this date (year only, e.g. 1980, becomes YYYY-01-01)')
    parser.add_argument('-to', '--to-date', dest='to_date', metavar='YYYY[-MM-DD]',
                        help='Include only data on or before this date (year only, e.g. 1980, becomes YYYY-12-31)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Do not print each year\'s gain')
    parser.add_argument('-s', '--samples', type=int, default=10000,
                        metavar='N', help='Number of multiverse samples (default: 10000)')
    parser.add_argument('-S', '--skip-multiverse', action='store_true',
                        help='Skip multiverse calculations (run only our universe)')
    parser.add_argument('-plot', '--plot', dest='plot', action='store_true',
                        help='Plot price OHLC candlesticks for the -from to -to range')
    parser.add_argument('-ymin', type=float, default=None,
                        help='Y-axis minimum (relative to first period close, e.g. 0.8 = 80%%)')
    parser.add_argument('-ymax', type=float, default=None,
                        help='Y-axis maximum (relative to first period close, e.g. 1.2 = 120%%)')
    args = parser.parse_args()

    daily_file = args.daily_file
    month_data = read_monthly_data(daily_file)

    from_dt, from_norm = parse_date_arg(args.from_date or '', is_from=True)
    to_dt, to_norm = parse_date_arg(args.to_date or '', is_from=False)
    month_data = filter_by_date_range(month_data, from_dt, to_dt)
    if not month_data:
        print('Error: No data remains after applying date range filter.')
        return

    if args.plot:
        plot_price_candlestick(daily_file, from_dt, to_dt,
                              annual_yield_file=args.annual_yield or '', tax_rate=TAX_RATE,
                              ymin=args.ymin, ymax=args.ymax)

    # Truncate to complete years (multiple of 12 months)
    month_data = truncate_to_full_years(month_data)

    if args.annual_yield.strip():
        annual_data = read_annual_data(args.annual_yield.strip())
        if not annual_data:
            annual_data = month_to_year_data(month_data)
        elif from_norm is not None or to_norm is not None:
            from_year = int(from_norm[:4]) if from_norm else 0
            to_year = int(to_norm[:4]) if to_norm else 9999
            annual_data = filter_annual_by_year_range(annual_data, from_year, to_year)
    else:
        annual_data = month_to_year_data(month_data)

    samples = args.samples
    #protection, cap = (1, 0.1064)
    #protection, cap = (0.09, 0.183)
    #protection, cap = (0, -1)

    #calc_and_print(annual_data, protection, cap) # our universe

    percentiles = [5, 25, 50, 75, 95]

    # Test different protection and cap configurations
    protection_cap_cases = [
        (0, -1),
        (1, 0.1064),
        (0.09, 0.183)
    ]

    verbose = not args.quiet
    for protection, cap in protection_cap_cases:
        calc_and_print(annual_data, protection, cap, verbose=verbose) # our universe

        if not args.skip_multiverse:
            # Measure execution time of the following line
            start_time = time.time()
            result = calc_multiverse(month_data, annual_data, protection, cap,
                                     sample_times=samples, want=percentiles)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'\nExecution time: {execution_time:.2f} seconds for {samples} multiverse')

            print(f'\nprotection={protection}, cap={cap}, {len(month_data)} months data ***')
            print('Perctl\tGain\tDrawdown\tWorstDD')
            for i, verse in enumerate(result):
                gain = verse.annualized * 100
                drawdown = verse.max_drawdown * 100
                worst_dd = verse.bucket_max_drawdown * 100 if verse.bucket_max_drawdown is not None else drawdown
                print(f'{percentiles[i]}\t{gain:.2f}%\t{drawdown:.2f}%\t{worst_dd:.2f}%')

    # Partial gain buffered ETF results
    loss_threshold = 0.1  # 15% loss threshold
    gain_fraction = 0.70    # 50% of gains

    calc_and_print_partial_gain(annual_data, loss_threshold, gain_fraction, verbose=verbose) # our universe

    if not args.skip_multiverse:
        # Measure execution time of the following line
        start_time = time.time()
        result_partial = calc_multiverse_partial_gain(month_data, annual_data, loss_threshold, gain_fraction,
                                                      sample_times=samples, want=percentiles)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'\nExecution time: {execution_time:.2f} seconds for {samples} multiverse')

        print(f'\nloss_threshold={loss_threshold}, gain_fraction={gain_fraction}, {len(month_data)} months data ***')
        print('Perctl\tGain\tDrawdown\tWorstDD')
        for i, verse in enumerate(result_partial):
            gain = verse.annualized * 100
            drawdown = verse.max_drawdown * 100
            worst_dd = verse.bucket_max_drawdown * 100 if verse.bucket_max_drawdown is not None else drawdown
            print(f'{percentiles[i]}\t{gain:.2f}%\t{drawdown:.2f}%\t{worst_dd:.2f}%')

if __name__ == "__main__":
    main()
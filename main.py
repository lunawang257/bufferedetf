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
    """Class to store and calculate investment results. Expects data with total return per period (gain)."""
    def __init__(self, start_money: float = 1):
        self.start_money = start_money
        self.max_drawdown = 0
        self.max_money = 0

    def calc_one_year(self, old_money: float, price_gain: float, protection: float, cap: float) -> float:
        """Calculate investment results for one year with optional protection and cap.

        Args:
            old_money: Starting amount of money
            price_gain: Total return multiplier for the year
            protection: Downside protection percentage
            cap: Maximum gain cap (-1 for no cap)

        Returns:
            float: New amount of money after calculations
        """
        buyhold_mon = old_money * price_gain

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

    def calc_one_year_partial_gain(self, old_money: float, price_gain: float, loss_threshold: float, gain_fraction: float) -> float:
        """Calculate investment results for one year with complete loss protection after threshold and partial gains.

        Args:
            old_money: Starting amount of money
            price_gain: Total return multiplier for the year
            loss_threshold: Complete loss protection threshold (e.g., 0.15 means losses capped at 15%)
            gain_fraction: Fraction of buy-and-hold gains captured (e.g., 0.5 means 50% of gains)

        Returns:
            float: New amount of money after calculations
        """
        buyhold_mon = old_money * price_gain

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
            data: List of dicts with 'gain' (total return multiplier) and optional 'year' for labeling
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
            cur_money = self.calc_one_year(cur_money, year_data['gain'], protection, cap)
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
            data: List of dicts with 'gain' (total return multiplier) and optional 'year' for labeling
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
            cur_money = self.calc_one_year_partial_gain(cur_money, year_data['gain'], loss_threshold, gain_fraction)
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


def _partial_gain_multiplier(price_gain: float, loss_threshold: float, gain_fraction: float) -> float:
    """Return the multiplier (new_money/old_money) for partial-gain buffered ETF with given cumulative return.
    Pure function: no Investment instance, for use in mark-to-market valuations."""
    if price_gain < 1:
        lost = 1 - price_gain
        capped_loss = min(lost, loss_threshold)
        return 1 - capped_loss
    if gain_fraction == -1:
        return price_gain
    gained = price_gain - 1
    partial_gain = gained * gain_fraction
    return 1 + partial_gain


def _read_annual_raw(filename: str) -> list:
    """Read annual CSV. Supports two formats:
    1. With headers: year,pricegain,yield
    2. Headerless: date,yield (e.g. 2023-12-31,0.0150)
    Returns list of dicts with year, yield, and optionally pricegain,
    sorted by year ascending.
    """
    data = []
    with open(filename, mode='r') as file:
        first_line = file.readline().strip()
        file.seek(0)

        if 'year' in first_line.lower():
            csvfile = csv.DictReader(file)
            for line_dict in csvfile:
                line_dict = dict(line_dict)
                data.append({
                    'year': int(line_dict['year']),
                    'pricegain': float(line_dict['pricegain']) + 1,
                    'yield': float(line_dict['yield'])
                })
        else:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 2:
                    continue
                date_str = row[0].strip()
                yield_val = row[1].strip()
                try:
                    dt = datetime.strptime(date_str, '%Y-%m-%d')
                    data.append({
                        'year': dt.year,
                        'yield': float(yield_val)
                    })
                except (ValueError, TypeError):
                    continue
    data.sort(key=lambda x: x['year'])
    return data


def truncate_to_full_years(month_data: list) -> list:
    """Truncate monthly data to a multiple of 12 months (complete years).
    Drops the last several months so all data is used in year chunks.
    """
    n = (len(month_data) // 12) * 12
    return month_data[:n] if n > 0 else month_data


# Supports CSV with "date,price", "date,adj_close", or "Date,Open,High,Low,Close,Volume" columns
def read_monthly_data(filename: str, annual_yield_file: str = '', tax_rate: float = TAX_RATE) -> list:
    """Read monthly data from CSV file. If annual_yield_file is given, applies yield (and tax) to monthly gains.

    Args:
        filename: Path to CSV with date and price column.
        annual_yield_file: Optional path to annual gain/yield CSV; if set, monthly gains are adjusted.
        tax_rate: Applied to yield when annual_yield_file is set (only used here).

    Returns:
        List of dictionaries with 'date' and 'gain'
    """
    data = []

    with open(filename, mode='r') as file:
        csvfile = csv.DictReader(file)

        prev_adj_close = None
        prev_month = None
        last_day = None
        last_day_adj_close = None
        for line_dict in csvfile:
            line_dict = dict(line_dict)
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
                if prev_adj_close is not None:
                    gain = last_day_adj_close / prev_adj_close
                    data.append({'date': last_day, 'gain': gain})
                prev_adj_close = last_day_adj_close
                prev_month = date.month
            last_day_adj_close = cur_adj_close
            last_day = date

        if last_day_adj_close is not None and prev_adj_close is not None:
            gain = last_day_adj_close / prev_adj_close
            data.append({'date': date, 'gain': gain})

    if annual_yield_file.strip():
        raw_annual = _read_annual_raw(annual_yield_file.strip())
        has_pricegain = raw_annual and 'pricegain' in raw_annual[0]
        if not has_pricegain:
            year_gains = {}
            for m in data:
                y = m['date'].year
                year_gains.setdefault(y, 1.0)
                year_gains[y] *= m['gain']
            for entry in raw_annual:
                entry['pricegain'] = year_gains.get(entry['year'], 1.0)
        year_map = {entry['year']: entry for entry in raw_annual}
        for i in range(len(data)):
            year = data[i]['date'].year
            if year in year_map:
                ag = year_map[year]['pricegain']
                ay = year_map[year]['yield']
                ratio = (ag + ay * (1 - tax_rate)) / ag
                data[i]['gain'] *= ratio ** (1 / 12)

    return data

def month_to_year_data(month_data: list) -> list:
    """Convert monthly data to yearly gain list (12-month compounded gain per year)."""
    yearly_data = []
    for i in range(0, len(month_data) - 11, 12):
        year_chunk = month_data[i:i+12]
        start_money = 1000
        money = start_money
        for month in year_chunk:
            money *= month['gain']
        price_gain = money / start_money
        year = year_chunk[-1]['date'].year
        yearly_data.append({'year': year, 'gain': price_gain})
    return yearly_data

def calc_multiverse_sample(month_data: list, protection: float, cap: float) -> AllYearsInfo:
    """Helper for calc_multiverse: one random ordering of months."""
    random.shuffle(month_data)
    gain_list = month_to_year_data(month_data)
    invest = Investment(START_MONEY)
    return invest.calc_all_years(gain_list, protection, cap)

def calc_multiverse(month_data: list, protection: float, cap: float,
                    sample_times: int = 5, want: list = DEFAULT_PERCENTILES) -> list:
    """Calculate multiverse results over random orderings of the given monthly data."""
    end_moneys = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calc_multiverse_sample, month_data.copy(), protection, cap) for _ in range(sample_times)]
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
    """Helper for calc_multiverse_partial_gain: one random ordering of months."""
    random.shuffle(month_data)
    gain_list = month_to_year_data(month_data)
    invest = Investment(START_MONEY)
    return invest.calc_all_years_partial_gain(gain_list, loss_threshold, gain_fraction)

def calc_multiverse_partial_gain(month_data: list, loss_threshold: float, gain_fraction: float,
                                 sample_times: int = 5, want: list = DEFAULT_PERCENTILES) -> list:
    """Calculate multiverse results for partial gain buffered ETF."""
    end_moneys = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calc_multiverse_sample_partial_gain, month_data.copy(), loss_threshold, gain_fraction) for _ in range(sample_times)]
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

def calc_and_print(data: list, protection: float, cap: float, verbose: bool = True):
    """data: list of dicts with 'gain' and optional 'year'."""
    print_cap = 'no' if cap == -1 else f'{cap * 100:.3f}%'
    print(f'\n*** Results for {protection * 100:.3f}% protection, {print_cap} cap ***')

    investment = Investment(START_MONEY)
    info = investment.calc_all_years(data, protection, cap, verbose=verbose)

    #print(f'Starting money: ${START_MONEY}\nEnding money: ${end_money:.2f}')
    print(f'Max drawdown: {info.max_drawdown * 100:.3f}%\nAnnualized gain: {info.annualized * 100:.3f}%')

def calc_and_print_partial_gain(data: list, loss_threshold: float, gain_fraction: float, verbose: bool = True):
    """Calculate and print results for partial gain buffered ETF.

    Args:
        data: List of dicts with 'gain' and optional 'year'
        loss_threshold: Complete loss protection threshold (e.g., 0.15 for 15%)
        gain_fraction: Fraction of buy-and-hold gains captured (e.g., 0.5 for 50%)
        verbose: If True, print gain/loss and max drawdown for each year
    """
    print(f'\n*** Results for {loss_threshold * 100:.3f}% loss threshold, {gain_fraction * 100:.3f}% of gains ***')

    investment = Investment(START_MONEY)
    info = investment.calc_all_years_partial_gain(data, loss_threshold, gain_fraction, verbose=verbose)

    print(f'Max drawdown: {info.max_drawdown * 100:.3f}%\nAnnualized gain: {info.annualized * 100:.3f}%')


def calc_pipelines(month_data: list, num_pipelines: int, loss_threshold: float, gain_fraction: float,
                  start_money: float, verbose: bool = False) -> tuple:
    """Pipelines strategy: deploy 1/num_pipelines of portfolio value num_pipelines times per year into partial-gain buffered ETFs;
    each position is held 12 months then sold and proceeds reinvested.

    Returns:
        tuple: (AllYearsInfo for overall portfolio, list of per-slot dicts with keys
               'month', 'annualized', 'max_drawdown', 'num_periods')
    """
    if not (1 <= num_pipelines <= 12):
        raise ValueError('num_pipelines must be between 1 and 12')
    deployment_months = sorted({1 + (12 * k) // num_pipelines for k in range(num_pipelines)})
    month_to_slot = {m: i for i, m in enumerate(deployment_months)}

    cash = start_money
    pipelines = []  # list of (initial_value, start_idx, cumulative_gain, slot_idx)
    max_money = start_money
    max_drawdown = 0.0

    slot_returns = [[] for _ in range(num_pipelines)]

    num_months = len(month_data)
    verbose_rows = [] if verbose else None
    prev_year_end_value = start_money

    for i in range(num_months):
        # 1) Start of month: close pipelines that have reached 12 months
        to_remove = [j for j, (_, start_idx, _, _) in enumerate(pipelines) if i - start_idx == 12]
        for j in reversed(to_remove):
            init, _, cum, slot = pipelines[j]
            multiplier = _partial_gain_multiplier(cum, loss_threshold, gain_fraction)
            slot_returns[slot].append(multiplier)
            cash += init * multiplier
            del pipelines[j]

        # 2) If deployment month: value all at start-of-month, then deploy (1/num_pipelines)*total
        month = month_data[i]['date'].month
        if month in month_to_slot:
            total = cash
            for (init, start_idx, cum, _) in pipelines:
                total += init * _partial_gain_multiplier(cum, loss_threshold, gain_fraction)
            deploy = (1 / num_pipelines) * total
            cash -= deploy
            slot = month_to_slot[month]
            pipelines.append((deploy, i, 1.0, slot))

        # 3) Apply this month's gain to all pipelines
        gain_i = month_data[i]['gain']
        for j in range(len(pipelines)):
            init, start_idx, cum, slot = pipelines[j]
            pipelines[j] = (init, start_idx, cum * gain_i, slot)

        # 4) End-of-month portfolio value; only update drawdown at year-end (consistent with partial-gain)
        port_value = cash
        for (init, start_idx, cum, _) in pipelines:
            port_value += init * _partial_gain_multiplier(cum, loss_threshold, gain_fraction)
        if (i + 1) % 12 == 0:
            if port_value > max_money:
                max_money = port_value
            dd = 1 - port_value / max_money
            if dd > max_drawdown:
                max_drawdown = dd

        if verbose and (i + 1) % 12 == 0:
            year_label = month_data[i]['date'].year
            year_gain_pct = (port_value - prev_year_end_value) / prev_year_end_value * 100
            verbose_rows.append((year_label, year_gain_pct, max_drawdown * 100))
            prev_year_end_value = port_value

    end_money = cash
    for (init, start_idx, cum, _) in pipelines:
        end_money += init * _partial_gain_multiplier(cum, loss_threshold, gain_fraction)

    gain = end_money / start_money
    num_years = num_months / 12
    annualized = (pow(gain, 1 / (num_years - 1)) - 1) if num_years > 1 else 0.0

    if verbose and verbose_rows:
        print('  {:>4}  {:>10}  {:>14}'.format('Year', 'Gain/Loss', 'Max DD so far'))
        print('  ' + '-' * 32)
        for year_label, gain_pct, max_dd in verbose_rows:
            print('  {:>4}  {:>+9.2f}%  {:>13.2f}%'.format(year_label, gain_pct, max_dd))

    # Compute per-slot annualized gain and max drawdown
    slot_infos = []
    for s in range(num_pipelines):
        returns = slot_returns[s]
        if not returns:
            slot_infos.append({'month': deployment_months[s], 'annualized': 0.0, 'max_drawdown': 0.0, 'num_periods': 0})
            continue
        compounded = 1.0
        peak = 1.0
        slot_max_dd = 0.0
        for r in returns:
            compounded *= r
            if compounded > peak:
                peak = compounded
            dd = 1 - compounded / peak
            if dd > slot_max_dd:
                slot_max_dd = dd
        n = len(returns)
        slot_ann = (pow(compounded, 1 / n) - 1) if n > 0 else 0.0
        slot_infos.append({
            'month': deployment_months[s],
            'annualized': slot_ann,
            'max_drawdown': slot_max_dd,
            'num_periods': n,
        })

    return AllYearsInfo(end_money, max_drawdown, gain, annualized), slot_infos


def calc_and_print_pipelines(month_data: list, num_pipelines: int, loss_threshold: float, gain_fraction: float,
                             verbose: bool = True) -> None:
    """Run pipelines strategy and print per-pipeline and overall max drawdown and annualized gain."""
    print(f'\n*** Pipelines (num_pipelines={num_pipelines}): {loss_threshold * 100:.3f}% loss threshold, {gain_fraction * 100:.3f}% of gains ***')
    info, slot_infos = calc_pipelines(month_data, num_pipelines, loss_threshold, gain_fraction, START_MONEY, verbose=verbose)

    print(f'\n  {"Pipeline":>8}  {"Deploy Mo":>9}  {"Annualized":>10}  {"Max DD":>10}  {"Periods":>7}')
    print('  ' + '-' * 49)
    for i, si in enumerate(slot_infos):
        print(f'  {i+1:>8}  {si["month"]:>9}  {si["annualized"]*100:>+9.3f}%  {si["max_drawdown"]*100:>9.3f}%  {si["num_periods"]:>7}')

    print(f'\nMax drawdown: {info.max_drawdown * 100:.3f}%\nAnnualized gain: {info.annualized * 100:.3f}%')


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


def filter_gain_list_by_year_range(month_data: list, from_year: int, to_year: int) -> list:
    """Filter monthly gain list to only include months whose year is within the given range.

    Args:
        month_data: List of dicts with 'date' and 'gain' (same format as read_monthly_data returns).
        from_year: Inclusive start year.
        to_year: Inclusive end year.

    Returns:
        Filtered list of monthly data in the same format.
    """
    return [m for m in month_data if from_year <= m['date'].year <= to_year]


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


def _apply_gain_list_to_ohlc(ohlc: list, gain_list: list, freq: str) -> list:
    """Scale price OHLC to total-return using gain list (year -> gain).
    Preserves actual bar-to-bar price variation; distributes the dividend
    adjustment (total_return / price_return) evenly within each year."""
    if not ohlc or not gain_list:
        return ohlc
    year_to_total_gain = {g['year']: g['gain'] for g in gain_list}

    year_bars = defaultdict(list)
    for bar in ohlc:
        year_bars[bar['date'].year].append(bar)

    prev_year_end_close = None
    year_div_factor = {}
    for y in sorted(year_bars.keys()):
        bars = year_bars[y]
        last_close = bars[-1]['close']
        if prev_year_end_close is not None and prev_year_end_close > 0:
            price_return = last_close / prev_year_end_close
            total_gain = year_to_total_gain.get(y)
            if total_gain is not None and price_return > 0:
                n = len(bars)
                year_div_factor[y] = (total_gain / price_return) ** (1 / n)
        prev_year_end_close = last_close

    base = ohlc[0]['close']
    cum_tr = [base]
    for i in range(1, len(ohlc)):
        bar = ohlc[i]
        prev_bar = ohlc[i - 1]
        price_gain = bar['close'] / prev_bar['close'] if prev_bar['close'] > 0 else 1.0
        div_factor = year_div_factor.get(bar['date'].year, 1.0)
        cum_tr.append(cum_tr[-1] * price_gain * div_factor)

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
                          gain_list: list = None, ymin: float = None, ymax: float = None) -> None:
    """Plot price OHLC candles. If gain_list is provided (monthly format, same as read_monthly_data),
    scales to total return."""
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
    if gain_list:
        plot_gains_monthly = filter_gain_list_by_year_range(gain_list, from_date.year, to_date.year)
        if plot_gains_monthly:
            plot_gains_yearly = month_to_year_data(truncate_to_full_years(plot_gains_monthly))
            if plot_gains_yearly:
                ohlc = _apply_gain_list_to_ohlc(ohlc, plot_gains_yearly, freq)
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
        all_years = sorted(year_to_first_idx.keys())
        min_gap = max(len(ohlc) // 30, 2)
        ticks, labels = [], []
        for y in all_years:
            idx = year_to_first_idx[y]
            if not ticks or idx - ticks[-1] >= min_gap:
                ticks.append(idx)
                labels.append(str(y))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, len(ohlc) - 0.5)

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    else:
        ax.autoscale(axis='y')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Buffered ETF backtest')
    parser.add_argument('daily_file', help='Path to daily (or monthly) price CSV file')
    parser.add_argument('annual_yield', nargs='?', default='',
                        help='Path to annual CSV file; if omitted, gain is derived from price data only')
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
    parser.add_argument('--pipelines-n', type=int, default=4, dest='num_pipelines', metavar='N',
                        help='Pipelines strategy: deployments per year 1-12 (default: 4)')
    args = parser.parse_args()

    daily_file = args.daily_file
    month_data = read_monthly_data(daily_file, args.annual_yield or '', TAX_RATE)

    from_dt, from_norm = parse_date_arg(args.from_date or '', is_from=True)
    to_dt, to_norm = parse_date_arg(args.to_date or '', is_from=False)
    month_data = filter_by_date_range(month_data, from_dt, to_dt)
    if not month_data:
        print('Error: No data remains after applying date range filter.')
        return

    month_data = truncate_to_full_years(month_data)
    gain_list = month_to_year_data(month_data)

    if args.plot:
        plot_gains_monthly = filter_gain_list_by_year_range(month_data, from_dt.year, to_dt.year)
        plot_price_candlestick(daily_file, from_dt, to_dt, gain_list=plot_gains_monthly,
                               ymin=args.ymin, ymax=args.ymax)

    samples = args.samples
    #protection, cap = (1, 0.1064)
    #protection, cap = (0.09, 0.183)
    #protection, cap = (0, -1)

    # calc_and_print(gain_list, protection, cap)  # our universe

    percentiles = [5, 25, 50, 75, 95]

    # Test different protection and cap configurations
    protection_cap_cases = [
        (0, -1),
        (1, 0.1064),
        (0.09, 0.183)
    ]

    verbose = not args.quiet
    for protection, cap in protection_cap_cases:
        calc_and_print(gain_list, protection, cap, verbose=verbose)  # our universe

        if not args.skip_multiverse:
            # Measure execution time of the following line
            start_time = time.time()
            result = calc_multiverse(month_data, protection, cap,
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

    calc_and_print_partial_gain(gain_list, loss_threshold, gain_fraction, verbose=verbose)  # our universe

    if not args.skip_multiverse:
        # Measure execution time of the following line
        start_time = time.time()
        result_partial = calc_multiverse_partial_gain(month_data, loss_threshold, gain_fraction,
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

    # Pipelines strategy (same loss_threshold, gain_fraction as partial-gain)
    num_pipelines = args.num_pipelines
    if not (1 <= num_pipelines <= 12):
        print(f'Error: --pipelines-n must be between 1 and 12 (got {num_pipelines})')
    else:
        calc_and_print_pipelines(month_data, num_pipelines, loss_threshold, gain_fraction, verbose=verbose)

if __name__ == "__main__":
    main()
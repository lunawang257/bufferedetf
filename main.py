#! /usr/bin/env python3

import argparse
import csv
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time  # Import the time module

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

# Supports CSV with either "date,price" or "date,adj_close" columns
def read_monthly_data(filename: str) -> list:
    """Read monthly data from CSV file.
    
    Args:
        filename: Path to CSV with date and price column.
                 Accepts either "date", "price" (2 columns) or "date", "adj_close".
        
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
            # Support both "price" and "adj_close" column names; skip rows with empty price
            raw_price = (line_dict.get('price') or line_dict.get('adj_close') or '').strip()
            if not raw_price:
                continue
            cur_adj_close = float(raw_price)
            date_str = (line_dict.get('date') or '').strip()
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
    """Helper function to calculate a single sample for calc_multiverse."""
    random.shuffle(month_data)
    year_data = month_to_year_data(month_data)
    invest = Investment(START_MONEY, TAX_RATE)
    return invest.calc_all_years(year_data, protection, cap)

def calc_multiverse(month_data: list, protection: float, cap: float, sample_times: int = 5, want: list = DEFAULT_PERCENTILES) -> list:
    end_moneys = []
    month_data = month_data.copy()  # copy to avoid modifying the original list

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calc_multiverse_sample, month_data, protection, cap) for _ in range(sample_times)]
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
    """Helper function to calculate a single sample for calc_multiverse_partial_gain."""
    random.shuffle(month_data)
    year_data = month_to_year_data(month_data)
    invest = Investment(START_MONEY, TAX_RATE)
    return invest.calc_all_years_partial_gain(year_data, loss_threshold, gain_fraction)

def calc_multiverse_partial_gain(month_data: list, loss_threshold: float, gain_fraction: float, sample_times: int = 5, want: list = DEFAULT_PERCENTILES) -> list:
    """Calculate multiverse results for partial gain buffered ETF.
    
    Args:
        month_data: List of monthly data dictionaries
        loss_threshold: Complete loss protection threshold
        gain_fraction: Fraction of buy-and-hold gains captured
        sample_times: Number of random samples to generate
        want: List of percentiles to return
    
    Returns:
        List of AllYearsInfo objects for requested percentiles
    """
    end_moneys = []
    month_data = month_data.copy()  # copy to avoid modifying the original list

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calc_multiverse_sample_partial_gain, month_data, loss_threshold, gain_fraction) for _ in range(sample_times)]
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

def calc_and_print(data: list, protection: float, cap: float, tax_rate: float = TAX_RATE):
    print_cap = 'no' if cap == -1 else f'{cap * 100:.3f}%'
    print(f'\n*** Results for {protection * 100:.3f}% protection, {print_cap} cap ***')

    investment = Investment(START_MONEY, tax_rate)
    info = investment.calc_all_years(data, protection, cap, verbose=True)
    
    #print(f'Starting money: ${START_MONEY}\nEnding money: ${end_money:.2f}')
    print(f'Max drawdown: {info.max_drawdown * 100:.3f}%\nAnnualized gain: {info.annualized * 100:.3f}%')

def calc_and_print_partial_gain(data: list, loss_threshold: float, gain_fraction: float, tax_rate: float = TAX_RATE):
    """Calculate and print results for partial gain buffered ETF.
    
    Args:
        data: List of year data dictionaries
        loss_threshold: Complete loss protection threshold (e.g., 0.15 for 15%)
        gain_fraction: Fraction of buy-and-hold gains captured (e.g., 0.5 for 50%)
        tax_rate: Tax rate for yield calculations
    """
    print(f'\n*** Results for {loss_threshold * 100:.3f}% loss threshold, {gain_fraction * 100:.3f}% of gains ***')

    investment = Investment(START_MONEY, tax_rate)
    info = investment.calc_all_years_partial_gain(data, loss_threshold, gain_fraction, verbose=True)
    
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

def main():
    parser = argparse.ArgumentParser(description='Buffered ETF backtest')
    parser.add_argument('daily_file', help='Path to daily (or monthly) price CSV file')
    parser.add_argument('annual_yield', nargs='?', default='',
                        help='Path to annual gain/yield CSV file; if omitted or empty, yield is assumed 0')
    args = parser.parse_args()

    daily_file = args.daily_file
    month_data = read_monthly_data(daily_file)

    if args.annual_yield.strip():
        annual_data = read_annual_data(args.annual_yield.strip())
        if not annual_data:
            annual_data = month_to_year_data(month_data)
    else:
        annual_data = month_to_year_data(month_data)

    month_with_yield = adjust_monthly_gain_with_yield(month_data, annual_data, TAX_RATE)

    samples = 10000
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

    for protection, cap in protection_cap_cases:
        calc_and_print(annual_data, protection, cap) # our universe

        # Measure execution time of the following line
        start_time = time.time()
        result = calc_multiverse(month_with_yield, protection, cap,
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

    calc_and_print_partial_gain(annual_data, loss_threshold, gain_fraction) # our universe

    # Measure execution time of the following line
    start_time = time.time()
    result_partial = calc_multiverse_partial_gain(month_with_yield, loss_threshold, gain_fraction,
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
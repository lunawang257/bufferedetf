import csv
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
START_MONEY = 1
TAX_RATE = 0.371
DEFAULT_PERCENTILES = [25, 50, 75]
DEFAULT_SAMPLE_TIMES = 5

class AllYearsInfo:
    """Class to store and compare investment performance metrics."""
    def __init__(self, end_money: float, max_drawdown: float, gain_per: float, annualized: float):
        self.end_money = end_money
        self.max_drawdown = max_drawdown
        self.gain_per = gain_per
        self.annualized = annualized

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
    
    def calc_all_years(self, data: list, protection: float, cap: float) -> AllYearsInfo:
        self.max_drawdown = 0
        self.max_money = 0

        cur_money = self.start_money
        for year_data in data:
            cur_money = self.calc_one_year(cur_money, year_data, protection, cap)

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
                'pricegain': float(line_dict['pricegain']) + 1,
                'yield': float(line_dict['yield'])
            })
    
    return data

# date, adjusted close (adj_close)
def read_monthly_data(filename: str) -> list:
    """Read monthly data from CSV file.
    
    Args:
        filename: Path to CSV file with date, open, adj_close columns
        
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
            cur_adj_close = float(line_dict['adj_close'])
            date = datetime.strptime(line_dict['date'], '%Y-%m-%d')
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
        nth = int(per / 100 * sample_times)
        out.append(end_moneys[nth])
    
    return out

def calc_and_print(data: list, protection: float, cap: float, tax_rate: float = TAX_RATE):
    print_cap = 'no' if cap == -1 else f'{cap * 100:.3f}%'
    print(f'\n*** Results for {protection * 100:.3f}% protection, {print_cap} cap ***')

    investment = Investment(START_MONEY, tax_rate)
    info = investment.calc_all_years(data, protection, cap)
    
    #print(f'Starting money: ${START_MONEY}\nEnding money: ${end_money:.2f}')
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
    annual_data = read_annual_data('sp500-annual-gain-yield-short.csv')
    '''calc_and_print(data, 1, -1)
    calc_and_print(annual_data, 0, -1)
    calc_and_print(data, 0, 0)
    calc_and_print(data, 1, 0)
    calc_and_print(data, 1, 0.1064)
    calc_and_print(data, 0.09, 0.183)'''

    daily_file = 'sp500-daily.csv'
    daily_file = 'sp500-daily-mini.csv' # 3 year data for testing

    month_data = read_monthly_data(daily_file)
    '''result = calc_multiverse(month_data, 0, -1)
    print('\n*** Multiverse results for no protection, no cap ***')
    for verse in result:
        print(verse.annualized, verse.max_drawdown)'''

    yield_month = adjust_monthly_gain_with_yield(month_data, annual_data, TAX_RATE)
    yield_month_to_year = month_to_year_data(yield_month)
    annual_data = read_annual_data('sp500-annual-mini.csv') # with yield

    print('\nCompare raw data')
    print('From monthly data (yield combined)')
    print_price_gain_with_yield(yield_month_to_year)
    print('\nFrom annual data')
    print_price_gain_with_yield(annual_data)

    calc_and_print(yield_month_to_year, 0, -1, tax_rate=0)
    calc_and_print(annual_data, 0, -1)

if __name__ == "__main__":
    main()
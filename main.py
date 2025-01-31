import csv
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
START_MONEY = 1
TAX_RATE = 0.5
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
    def __init__(self, start_money: float = 1, tax_rate: float = 0.5):
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
        buyhold_mon *= data['pricegain']
        dividend = buyhold_mon * data['yield']
        buyhold_mon += dividend - dividend * self.tax_rate
        
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
        for line_dict in csvfile:
            line_dict = dict(line_dict)
            date = datetime.strptime(line_dict['date'], '%Y-%m-%d')
            if prev_month is None:
                prev_month = date.month
            elif date.month != prev_month:
                # only keep the first day of each month
                cur_adj_close = float(line_dict['adj_close'])
                if prev_adj_close is None:
                    gain = 1 # first month has special value 1 means no change
                else:
                    gain = cur_adj_close / prev_adj_close
                data.append({
                    'date': date,
                    'gain': gain
                })
                prev_adj_close = cur_adj_close
                prev_month = date.month
    
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

def calc_multiverse(month_data: list, sample_times: int = 5, want: list = [25, 50, 75]) -> list:
    end_moneys = []
    month_data = month_data.copy()  # copy to avoid modifying the original list

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calc_multiverse_sample, month_data, 0, -1) for _ in range(sample_times)]
        for future in as_completed(futures):
            end_moneys.append(future.result())

    end_moneys.sort()
    out = []

    for per in want:
        nth = int(per / 100 * sample_times)
        out.append(end_moneys[nth])
    
    return out

def calc_and_print(data: list, protection: float, cap: float):
    print_cap = 'no' if cap == -1 else f'{cap * 100:.3f}%'
    print(f'\n*** Results for {protection * 100:.3f}% protection, {print_cap} cap ***')

    investment = Investment(START_MONEY, TAX_RATE)
    info = investment.calc_all_years(data, protection, cap)
    
    #print(f'Starting money: ${START_MONEY}\nEnding money: ${end_money:.2f}')
    print(f'Max drawdown: {info.max_drawdown * 100:.3f}%\nAnnualized gain: {info.annualized * 100:.3f}%')

def main():
    data = read_annual_data('sp500-annual-gain-yield-short.csv')
    calc_and_print(data, 0, -1)
    calc_and_print(data, 1, -1)
    calc_and_print(data, 0, 0)
    calc_and_print(data, 1, 0)
    calc_and_print(data, 1, 0.1064)
    calc_and_print(data, 0.09, 0.183)

    data_file = 'sp500-short.csv'
    data_file = 'sp500-short-mini.csv' # 3 year data for testing

    month_data = read_monthly_data(data_file)
    result = calc_multiverse(month_data)
    for verse in result:
        print(verse.annualized, verse.max_drawdown)

    data2 = read_monthly_data(data_file)
    data2 = month_to_year_data(data2)
    calc_and_print(data2, 0, -1)
    data3 = read_annual_data('sp500-annual-mini.csv')
    calc_and_print(data3, 0, -1)



if __name__ == "__main__":
    main()
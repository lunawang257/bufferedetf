import csv
import random
from datetime import datetime

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
        buyhold_mon += buyhold_mon * data['pricegain']
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

        gain = cur_money / self.start_money - 1
        annualized_gain = pow(gain, 1 / (len(data) - 1)) - 1

        return AllYearsInfo(cur_money, self.max_drawdown, gain, annualized_gain)

def read_annual_data(filename: str) -> list:
    data = []

    with open(filename, mode='r') as file:
        csvfile = csv.DictReader(file)

        for line_dict in csvfile:
            line_dict = dict(line_dict)

            for key, value in line_dict.items():
                if key == 'year':
                    line_dict[key] = int(value)
                else:
                    line_dict[key] = float(value)

            data.append(line_dict)
    
    return data

# date, adjusted close (adj_close)
def read_monthly_data(filename: str) -> list:
    data = []

    with open(filename, mode='r') as file:
        csvfile = csv.DictReader(file)

        for line_dict in csvfile:
            line_dict = dict(line_dict)

            line_dict['date'] = datetime.strptime(line_dict['date'], '%Y-%m-%d')

            if line_dict['date'].month == 1:
                data.append( {'date': line_dict['date'], 'adj_close': float(line_dict['adj_close'])} )
    
    return data

def calc_multiverse(data: list, want: list = [25, 50, 75], sample_times: int = 5) -> list:
    end_moneys = []
    data = data.copy()
    for i in range(sample_times):
        random.shuffle(data)
        
        # debugging
        '''for year in data:
            print(year['year'], end=' ')
        print()
        print()'''

        end_moneys.append(calc_all_years(data, 0, -1))

    end_moneys.sort()
    out = []

    for per in want:
        nth = int(per / 100 * sample_times)
        out.append(end_moneys[nth])
    
    return out

def calc_and_print(data, protection, cap):
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
    calc_and_print(data, 1, 0.1064)
    calc_and_print(data, 0.09, 0.183)

    result = calc_multiverse(data)
    for verse in result:
        print(verse.annualized)

if __name__ == "__main__":
    main()
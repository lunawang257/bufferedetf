import csv
import random
import datetime

max_drawdown = 0
max_money = 0
START_MONEY = 1
TAX_RATE = 0.5

class all_years_info:
    def __init__(self, end_money, max_drawdown, gain_per, annualized):
        self.end_money = end_money
        self.max_drawdown = max_drawdown
        self.gain_per = gain_per
        self.annualized = annualized

    def __lt__(self, other):
        return self.end_money < other.end_money
    
    def __eq__(self, other):
        return self.end_money == other.end_money

def read_annual_data(filename):
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
def read_monthly_data(filename):
    data = []

    with open(filename, mode='r') as file:
        csvfile = csv.DictReader(file)

        for line_dict in csvfile:
            line_dict = dict(line_dict)

            line_dict['date'] = datetime.strptime(line_dict['date'], "%Y-%m-%d")

            if line_dict['date'].month == 1:
                data.append( {'date': line_dict['date'], 'adj_close': float(line_dict['adj_close'])} )
    
    return data

def calc_multiverse(data, want=[25, 50, 75], sample_times=5):
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
        nth = int(per / 100 * sample) # TODO 0 or 1 indexing??
        out.append(end_moneys[nth])
    
    return out

def calc_one_year(old_money, data: dict, protection, cap):
    global max_drawdown
    global max_money

    buyhold_mon = old_money
    buyhold_mon += buyhold_mon * data['pricegain']
    dividend = buyhold_mon * data['yield']
    buyhold_mon += dividend - dividend * TAX_RATE
    
    if buyhold_mon < old_money:
        # might need protection
        lost = 1 - buyhold_mon / old_money
        new_money = old_money - old_money * max(lost - protection, 0)
    elif cap != -1:
        # might hit cap
        gained = buyhold_mon / old_money - 1
        gained = min(gained, cap)
        new_money = old_money + old_money * gained
    else:
        new_money = buyhold_mon

    if new_money > max_money:
        max_money = new_money
    drawdown = 1 - new_money / max_money

    if drawdown > max_drawdown:
        max_drawdown = drawdown

    return new_money

# cap = -1 means no cap
def calc_all_years(data, protection, cap):
    global max_drawdown
    global max_money
    max_drawdown = 0
    max_money = 0

    cur_money = START_MONEY
    for year_data in data:
        cur_money = calc_one_year(cur_money, year_data, protection, cap)

    gain = cur_money / START_MONEY - 1
    annualized_gain = pow(gain, 1 / (len(data) - 1)) - 1

    return all_years_info(cur_money, max_drawdown, gain, annualized_gain)

def calc_and_print(data, protection, cap):
    print_cap = 'no' if cap == -1 else f'{cap * 100:.3f}%'
    print(f'\n*** Results for {protection * 100:.3f}% protection, {print_cap} cap ***')
    info = calc_all_years(data, protection, cap)
    
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
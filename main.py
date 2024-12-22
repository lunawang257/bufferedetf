import csv

max_drawdown = 0
max_money = 0
START_MONEY = 100000
TAX_RATE = 0.5

def read_data(filename):
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

def calc_one_year(money, data: dict, type):
    global max_drawdown
    global max_money

    new_money = money
    new_money += new_money * data['pricegain']
    dividend = new_money * data['yield']
    new_money += dividend - dividend * TAX_RATE

    if new_money > max_money:
        max_money = new_money
    drawdown = 1 - new_money / max_money

    if drawdown > max_drawdown:
        max_drawdown = drawdown

    return new_money

def calc_all_years(data):
    cur_money = START_MONEY
    for year_data in data:
        cur_money = calc_one_year(cur_money, year_data, None)

    return cur_money


def main():
    data = read_data('sp500-annual-gain-yield-short.csv')
    end_money = calc_all_years(data)
    print(f'Starting money: ${START_MONEY}\nEnding money: ${end_money:.2f}\nMax drawdown: {max_drawdown * 100:.3f}%')

if __name__ == "__main__":
    main()
import pytest
from main import read_monthly_data, adjust_monthly_gain_with_yield, month_to_year_data, read_annual_data, print_price_gain_with_yield, calc_and_print, TAX_RATE

def test_args():
    annual_data = read_annual_data('sp500-annual-gain-yield-short.csv')
    calc_and_print(annual_data, 1, -1)
    calc_and_print(annual_data, 0, -1)
    calc_and_print(annual_data, 0, 0)
    calc_and_print(annual_data, 1, 0)
    calc_and_print(annual_data, 1, 0.1064)
    calc_and_print(annual_data, 0.09, 0.183)

def test_adjust_yield():
    daily_file = 'sp500-daily-mini.csv' # 3 year data for testing
    month_data = read_monthly_data(daily_file)
    annual_data = read_annual_data('sp500-annual-mini.csv') # with yield
    yield_month = adjust_monthly_gain_with_yield(month_data, annual_data, TAX_RATE)
    yield_month_to_year = month_to_year_data(yield_month)

    print('\nCompare raw data')
    print('From monthly data (yield combined)')
    print_price_gain_with_yield(yield_month_to_year)
    print('\nFrom annual data')
    print_price_gain_with_yield(annual_data)

    calc_and_print(yield_month_to_year, 0, -1, tax_rate=0)
    calc_and_print(annual_data, 0, -1)
    
    # make sure yield_month_to_year and annual_data are close enough
    for i in range(len(yield_month_to_year)):
        gain1 = yield_month_to_year[i]['pricegain']
        gain2 = annual_data[i]['pricegain'] + annual_data[i]['yield']
        assert gain1 - gain2 < 0.0001

if __name__ == "__main__":
    pytest.main()

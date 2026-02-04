try:
    import pytest
except ImportError:
    pytest = None

from main import read_monthly_data, adjust_monthly_gain_with_yield, month_to_year_data, read_annual_data, print_price_gain_with_yield, calc_and_print, calc_and_print_partial_gain, calc_multiverse_partial_gain, TAX_RATE

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
        assert gain1 - gain2 < 0.000001

def test_partial_gain_buffered_etf():
    """Test the new type of buffered ETF with complete loss protection after threshold and partial gains."""
    annual_data = read_annual_data('sp500-annual-gain-yield-short.csv')
    
    # Test with different loss thresholds and gain fractions
    print('\n=== Testing Partial Gain Buffered ETF ===')
    calc_and_print_partial_gain(annual_data, loss_threshold=0.15, gain_fraction=0.5)  # 15% loss cap, 50% of gains
    calc_and_print_partial_gain(annual_data, loss_threshold=0.20, gain_fraction=0.6)  # 20% loss cap, 60% of gains
    calc_and_print_partial_gain(annual_data, loss_threshold=0.10, gain_fraction=0.4)  # 10% loss cap, 40% of gains
    
    # Test with multiverse for one configuration
    daily_file = 'sp500-daily-mini.csv'
    month_data = read_monthly_data(daily_file)
    annual_data_mini = read_annual_data('sp500-annual-mini.csv')
    month_with_yield = adjust_monthly_gain_with_yield(month_data, annual_data_mini, TAX_RATE)
    
    print('\n=== Multiverse Test for Partial Gain Buffered ETF ===')
    percentiles = [25, 50, 75]
    result = calc_multiverse_partial_gain(month_with_yield, loss_threshold=0.15, gain_fraction=0.5,
                                         sample_times=100, want=percentiles)
    
    print(f'\nloss_threshold=0.15, gain_fraction=0.5, {len(month_data)} months data ***')
    print('Perctl\tGain\tDrawdown')
    for i, verse in enumerate(result):
        gain = verse.annualized * 100
        drawdown = verse.max_drawdown * 100
        print(f'{percentiles[i]}\t{gain:.2f}%\t{drawdown:.2f}%')

if __name__ == "__main__":
    if pytest:
        pytest.main()
    else:
        # Run tests directly if pytest is not available
        test_args()
        test_adjust_yield()
        test_partial_gain_buffered_etf()

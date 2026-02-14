#! /usr/bin/env python3
try:
    import pytest
except ImportError:
    pytest = None

import copy

from main import (
    read_monthly_data,
    adjust_monthly_gain_with_yield,
    month_to_year_data,
    read_annual_data,
    print_price_gain_with_yield,
    calc_and_print,
    calc_and_print_partial_gain,
    calc_multiverse_partial_gain,
    calc_multiverse_sample_partial_gain,
    TAX_RATE,
)

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

    print('\n=== Multiverse Test for Partial Gain Buffered ETF ===')
    percentiles = [25, 50, 75]
    result = calc_multiverse_partial_gain(month_data, annual_data_mini, loss_threshold=0.15, gain_fraction=0.5,
                                         sample_times=100, want=percentiles)
    
    print(f'\nloss_threshold=0.15, gain_fraction=0.5, {len(month_data)} months data ***')
    print('Perctl\tGain\tDrawdown\tWorstDD')
    for i, verse in enumerate(result):
        gain = verse.annualized * 100
        drawdown = verse.max_drawdown * 100
        worst_dd = verse.bucket_max_drawdown * 100 if verse.bucket_max_drawdown is not None else drawdown
        print(f'{percentiles[i]}\t{gain:.2f}%\t{drawdown:.2f}%\t{worst_dd:.2f}%')


def test_multiverse_identical_when_buy_and_hold():
    """Use loss_threshold=1 to simulate buy-and-hold. Verify shuffling doesn't change results.

    With loss_threshold=1 we accept all losses; with gain_fraction=1 we capture full gains.
    This is pure buy-and-hold: total return = product of all monthly gains.
    Shuffling only reorders which 12 months form each synthetic year, but the product
    of ALL months is invariant. All multiverse samples must yield identical end_money.
    """
    daily_file = 'sp500-daily-mini.csv'
    month_data = read_monthly_data(daily_file)
    annual_data = read_annual_data('sp500-annual-mini.csv')
    month_with_yield = adjust_monthly_gain_with_yield(month_data, annual_data, TAX_RATE)

    n = (len(month_with_yield) // 12) * 12
    month_with_yield = month_with_yield[:n]
    assert n >= 12, "Need at least 12 months of data"

    # loss_threshold=1 simulates buy-and-hold: take all losses, full gains
    loss_threshold = 1.0
    gain_fraction = 1.0

    # Run many samples; each shuffle produces a different month ordering
    results = []
    for _ in range(50):
        data_copy = copy.deepcopy(month_with_yield)
        info = calc_multiverse_sample_partial_gain(data_copy, loss_threshold, gain_fraction)
        results.append(info.end_money)

    # Shuffling must not change results: all samples identical
    assert all(abs(r - results[0]) < 1e-10 for r in results), (
        f"Shuffling must not change buy-and-hold results; got differing end_money: {results[:5]}..."
    )


def test_gain_fraction_minus_one_no_cap():
    """gain_fraction=-1 should mean 'no cap' (100% of gains), like cap=-1 in calc_one_year.

    Uses Investment.calc_one_year (protection=0, cap=-1) to compute buy-and-hold result
    over shuffled year data. Verifies calc_one_year_partial_gain with loss_threshold=1,
    gain_fraction=-1 produces identical results.
    """
    import random
    from main import Investment

    daily_file = 'sp500-daily-mini.csv'
    month_data = read_monthly_data(daily_file)
    annual_data = read_annual_data('sp500-annual-mini.csv')
    month_with_yield = adjust_monthly_gain_with_yield(month_data, annual_data, TAX_RATE)
    n = (len(month_with_yield) // 12) * 12
    month_with_yield = month_with_yield[:n]

    # Shuffle and convert to year data
    data_copy = copy.deepcopy(month_with_yield)
    random.shuffle(data_copy)
    year_data = month_to_year_data(data_copy)

    inv_bh = Investment(1.0, 0.0)
    inv_partial = Investment(1.0, 0.0)

    # Buy-and-hold via calc_one_year (protection=0, cap=-1)
    money_bh = 1.0
    for yd in year_data:
        money_bh = inv_bh.calc_one_year(money_bh, yd, protection=0, cap=-1)

    # Same via calc_one_year_partial_gain (loss_threshold=1, gain_fraction=-1 = no cap)
    money_partial = 1.0
    for yd in year_data:
        money_partial = inv_partial.calc_one_year_partial_gain(
            money_partial, yd, loss_threshold=1.0, gain_fraction=-1
        )

    assert money_partial > 0, "gain_fraction=-1 (no cap) should give positive return"
    assert abs(money_partial - money_bh) < 1e-10, (
        f"gain_fraction=-1 should match calc_one_year B&H: got {money_partial}, expected {money_bh}"
    )


def test_loss_threshold_zero_gain_fraction_minus_one_sensible():
    """With loss_threshold=0, gain_fraction=-1: floor + full gains. Results can differ by shuffle.

    With lt=0 we never lose (floor). With gf=-1 we get full gains. This is path-dependent:
    different month orderings produce different synthetic years, so multiverse samples can differ.
    This test just ensures we get sensible positive returns, not the buggy inverted ones.
    """
    daily_file = 'sp500-daily-mini.csv'
    month_data = read_monthly_data(daily_file)
    annual_data = read_annual_data('sp500-annual-mini.csv')
    month_with_yield = adjust_monthly_gain_with_yield(month_data, annual_data, TAX_RATE)
    n = (len(month_with_yield) // 12) * 12
    month_with_yield = month_with_yield[:n]

    results = []
    for _ in range(10):
        data_copy = copy.deepcopy(month_with_yield)
        info = calc_multiverse_sample_partial_gain(data_copy, loss_threshold=0.0, gain_fraction=-1)
        results.append(info.end_money)

    # All should be positive and in a reasonable range (no inverted/negative from bug)
    assert all(r > 0 for r in results), f"lt=0, gf=-1 should give positive returns, got {results}"


if __name__ == "__main__":
    if pytest:
        pytest.main()
    else:
        # Run tests directly if pytest is not available
        test_args()
        test_adjust_yield()
        test_partial_gain_buffered_etf()

from typing import List
from datetime import datetime
from algos.RollingAverages import RollingAverages
from repos import CoinBaseRepo

class ExchangeRateItem:
    def __init__(self, unix : float, date : datetime, low: float, high : float, open : float, close : float, volume: float):
        self.unix = unix
        self.date = date
        self.low = low
        self.high = high
        self.open = open
        self.close = close
        self.volume = volume

        
def investment_strategy(exchange_rates: List[ExchangeRateItem], window_length: int = 10) -> (float, float):
    """
    Simulate an investment strategy based on the moving average of Bitcoin prices.

    Parameters:
    exchange_rates (List[ExchangeRateItem]): List of daily exchange rates.
    window_length (int): Size of the window for the moving average calculation.

    Returns:
    tuple: Total amount of Bitcoin purchased and remaining balance in Euros.
    """
    total_euro_spent = 0
    balance_euro = 0
    total_bitcoin_purchased = 0

    # Extract close prices from exchange rates
    close_prices = [item.close for item in exchange_rates]

    # Iterate through each day's exchange rate
    for idx, rate_item in enumerate(exchange_rates):
        # Add 100 Euro to balance if it's the first of the month
        if rate_item.date.day == 1:
            balance_euro += 100
            total_euro_spent += 100

        # Only calculate moving average if there are enough data points (at least 'window_length' items)
        if idx >= window_length - 1:
            moving_avg = RollingAverages.single_moving_average(close_prices[idx+1-window_length:idx+1])

            # If the current close price is below the moving average and balance is available
            if rate_item.close < moving_avg and balance_euro > 0:
                bitcoin_to_purchase = balance_euro / rate_item.close
                total_bitcoin_purchased += bitcoin_to_purchase
                #print(f"bought {bitcoin_to_purchase} btc, now {total_bitcoin_purchased} at date {rate_item.date}")
                balance_euro = 0  # Reset balance after purchase

    return total_bitcoin_purchased, balance_euro, total_euro_spent

start_filter = datetime(2016, 4, 1)
end_filter = datetime(2023, 8, 24)
dictlist_btc = CoinBaseRepo.read_csv_to_dict('repos/BTC_EUR.csv')
sample_exchange_rates = CoinBaseRepo.get_exchange_rate_items(start_filter, end_filter, dictlist_btc)

for window_length in [1, 2, 5, 10, 20, 40, 80]:
    total_bitcoin, remaining_balance, total_euro_spent= investment_strategy(sample_exchange_rates, window_length=window_length)
    total_money_now = sample_exchange_rates[-1].close * total_bitcoin
    print(window_length, total_bitcoin, remaining_balance, total_euro_spent, total_money_now)

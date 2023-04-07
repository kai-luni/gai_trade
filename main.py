from datetime import datetime, timedelta
import sys
import os
import time
from algos.RollingAverages import RollingAverages

#add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from api.CoinBaseRepo import CoinBaseRepo
from api.FearGreedRepo import FearGreedRepo 

from algos.PearsonCorrelationCoefficient import PCC




#if main
if __name__ == "__main__":
    # start = datetime(2019, 1, 1)
    # end = datetime(2019, 2, 1)
    # while start < datetime.now():
    #     CoinBaseRepo.fetch_daily_data('ETH/EUR', start, end)
    #     start = start + timedelta(days=30)
    #     end = end + timedelta(days=30)
    #     #sleep 1 second
    #     time.sleep(1)
    #     print(start)


    start_filter = datetime(2018, 2, 1)
    end_filter = datetime(2023, 4, 6)
    dictlist_btc = CoinBaseRepo.read_csv_to_dict('api/BTC_EUR.csv')
    exchange_items_btc = CoinBaseRepo.get_exchange_rate_items(start_filter, end_filter, dictlist_btc)
    dictlist_eth = CoinBaseRepo.read_csv_to_dict('api/ETH_EUR.csv')
    exchange_items_eth = CoinBaseRepo.get_exchange_rate_items(start_filter, end_filter, dictlist_eth)

    #FearGreedRepo.get_data(10000, 'api/fear_greed_data.csv')
    greed_items = FearGreedRepo.read_csv_file(start_filter, end_filter, 'api/fear_greed_data.csv')
    print(f"Number of fear greed items: {len(greed_items)}")

    # Replace with your actual data
    prices_btc = [item.open for item in exchange_items_btc]
    prices_eth = [item.open for item in exchange_items_eth]
    fear_greed_index = [item["index"] for item in greed_items]

    # Get buy and sell signals
    signals_btc = RollingAverages.get_buy_sell_signals(prices_btc, fear_greed_index, 4, 90, 46, 55)
    signals_eth = RollingAverages.get_buy_sell_signals(prices_eth, fear_greed_index, 4, 90, 46, 55)

    # final_btc = RollingAverages.simulate_trading(prices_btc, signals_btc)
    # final_eth = RollingAverages.simulate_trading(prices_eth, signals_eth)
    # print(f"btc one: {final_btc}")
    # print(f"eth one: {final_eth}")
    signals_btc = RollingAverages.get_buy_sell_signals(prices_btc, fear_greed_index, 6, 100, 46, 55)
    signals_eth = RollingAverages.get_buy_sell_signals(prices_eth, fear_greed_index, 6, 100, 46, 55)
    final_btc = RollingAverages.simulate_trading(prices_btc, signals_btc)
    final_eth = RollingAverages.simulate_trading(prices_eth, signals_eth)
    print(f"btc one: {final_btc}")
    print(f"eth one: {final_eth}")
    #raise SystemExit(0)

    # print("Buy and sell signals:")
    # for signal, index in signals_btc:
    #     print(f"{signal.capitalize()} at index {exchange_items_btc[index].date}")

    # Define the range of values for each parameter
    short_window_values = range(3, 15)
    long_window_values = range(20, 101, 10)
    buy_threshold_values = range(1, 51, 5)
    sell_threshold_values = range(50, 101, 5)

    best_params = None
    best_final_bitcoin = 0

    # Iterate over all combinations of parameter values
    for short_window in short_window_values:
        for long_window in long_window_values:
            for buy_threshold in buy_threshold_values:
                for sell_threshold in sell_threshold_values:
                    # Get buy and sell signals for the current parameter combination
                    signals_btc = RollingAverages.get_buy_sell_signals(prices_btc, fear_greed_index, short_window, long_window, buy_threshold, sell_threshold)
                    signals_eth = RollingAverages.get_buy_sell_signals(prices_eth, fear_greed_index, short_window, long_window, buy_threshold, sell_threshold)

                    # Calculate the final amount of Bitcoin for the current parameter combination
                    final_bitcoin = RollingAverages.simulate_trading(prices_btc, signals_btc)
                    final_eth = RollingAverages.simulate_trading(prices_eth, signals_eth)

                    # Update the best parameter combination if the current one yields a higher final amount of Bitcoin
                    if final_bitcoin > best_final_bitcoin and final_eth > 1.1:
                        best_final_bitcoin = final_bitcoin
                        best_params = (short_window, long_window, buy_threshold, sell_threshold)

                        # Print the result for the current parameter combination
                        print(f"Short window: {short_window}, Long window: {long_window}, Buy threshold: {buy_threshold}, Sell threshold: {sell_threshold}, Final Bitcoin: {final_bitcoin:.8f} BTC")

    print(f"Best parameters: Short window: {best_params[0]}, Long window: {best_params[1]}, Buy threshold: {best_params[2]}, Sell threshold: {best_params[3]}, Final Bitcoin: {best_final_bitcoin:.8f} BTC")


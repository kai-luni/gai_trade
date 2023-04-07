import pandas as pd
from typing import List, Dict, Tuple

class RollingAverages:
    def moving_average(data: List[float], window: int) -> List[float]:
        return pd.Series(data).rolling(window=window).mean().tolist()

    def get_buy_sell_signals(prices: List[float], fear_greed_index: List[int], short_window: int, long_window: int, buy_threshold: int, sell_threshold: int) -> List[Tuple[str, int]]:
        short_mavg = RollingAverages.moving_average(prices, short_window)
        long_mavg = RollingAverages.moving_average(prices, long_window)

        signals = []
        position = None

        for i in range(long_window, len(prices)):
            # Check for a buy signal
            if short_mavg[i] > long_mavg[i] and fear_greed_index[i] <= buy_threshold and position != "buy":
                position = "buy"
                signals.append(("buy", i))

            # Check for a sell signal
            elif short_mavg[i] < long_mavg[i] and fear_greed_index[i] >= sell_threshold and position != "sell":
                position = "sell"
                signals.append(("sell", i))

        return signals
    
    def simulate_trading(prices: List[float], signals: List[Tuple[str, int]]) -> float:
        cash = 0.0
        bitcoin = 1.0

        for signal, index in signals:
            price = prices[index]

            if signal == "buy" and cash > 0:
                # Buy Bitcoin with all available cash
                bitcoin = cash / price
                cash = 0

            elif signal == "sell" and bitcoin > 0:
                # Sell all Bitcoin and convert to cash
                cash = bitcoin * price
                bitcoin = 0

        # Calculate the final amount of Bitcoin
        if cash > 0:
            final_bitcoin = cash / prices[-1] + bitcoin
        else:
            final_bitcoin = bitcoin

        return final_bitcoin

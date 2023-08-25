import pandas as pd
from typing import List, Tuple

class RollingAverages:
    """
    This class contains methods related to computing rolling averages and trading simulation based on them.
    """

    @staticmethod
    def moving_average(data: List[float], window: int) -> List[float]:
        """
        Calculate the moving average over a specified window for a given data set.

        Parameters:
        data (List[float]): List of floating point numbers representing the data set.
        window (int): Size of the window over which to calculate the moving average.

        Returns:
        List[float]: List of floating point numbers representing the moving averages.
        """
        return pd.Series(data).rolling(window=window).mean().tolist()

    @staticmethod
    def get_buy_sell_signals(prices: List[float], fear_greed_index: List[int], short_window: int, long_window: int, buy_threshold: int, sell_threshold: int) -> List[Tuple[str, int]]:
        """
        Generate buy and sell trading signals based on moving averages and fear-greed index.

        Parameters:
        prices (List[float]): List of floating point numbers representing prices.
        fear_greed_index (List[int]): List of integers representing the fear-greed index.
        short_window (int): Size of the short moving average window.
        long_window (int): Size of the long moving average window.
        buy_threshold (int): Threshold value for the fear-greed index to trigger a buy signal.
        sell_threshold (int): Threshold value for the fear-greed index to trigger a sell signal.

        Returns:
        List[Tuple[str, int]]: List of tuples representing buy and sell signals and their respective index positions.
        """
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

    @staticmethod
    def simulate_trading(prices: List[float], signals: List[Tuple[str, int]]) -> float:
        """
        Simulate trading based on given buy and sell signals, starting with 1 bitcoin and $0. 

        Parameters:
        prices (List[float]): List of floating point numbers representing prices.
        signals (List[Tuple[str, int]]): List of tuples representing buy and sell signals and their respective index positions.

        Returns:
        float: Final amount of bitcoin after executing the buy and sell signals.
        """
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
    
    @staticmethod
    def single_moving_average(data: List[float]) -> float:
        """
        Calculate the moving average over the entire data set and return the last moving average value.

        Parameters:
        data (List[float]): List of floating point numbers representing the data set.

        Returns:
        float: The last moving average value.
        """
        window = len(data)
        moving_avgs = pd.Series(data).rolling(window=window).mean().tolist()
        
        # Return the last moving average value
        return moving_avgs[-1]

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List


class PCC:
    @staticmethod
    def calculate_pcc(dates: List[datetime], stock_prices: List[float], fear_and_greed: List[int]) -> float:
        """
        Calculate the Pearson correlation coefficient (PCC) between daily percentage changes in stock prices and the fear and greed index.

        Args:
            dates (List[datetime]): List of dates for the data points.
            stock_prices (List[float]): List of stock prices corresponding to the dates.
            fear_and_greed (List[int]): List of fear and greed index values corresponding to the dates.

        Returns:
            float: The Pearson correlation coefficient (PCC) between daily percentage changes in stock prices and the fear and greed index.
        """
        # Check if all input lists have the same length
        if len(dates) != len(stock_prices) or len(dates) != len(fear_and_greed):
            raise ValueError("All input lists (dates, stock_prices, and fear_and_greed) must have the same length.")
        
        # Calculate daily percentage changes in stock prices
        stock_price_changes = [(stock_prices[i + 1] - stock_prices[i]) / stock_prices[i] * 100 for i in range(len(stock_prices) - 1)]

        # Remove the first date since there's no stock price change for the first day
        dates = dates[1:]
        fear_and_greed = fear_and_greed[1:]

        # Create a DataFrame
        data = {'Date': dates, 'StockPriceChange': stock_price_changes, 'FearAndGreed': fear_and_greed}
        df = pd.DataFrame(data)

        # Calculate the Pearson correlation coefficient
        correlation = np.corrcoef(df['StockPriceChange'], df['FearAndGreed'])[0, 1]

        return correlation

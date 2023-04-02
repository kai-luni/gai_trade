import pandas as pd
import numpy as np
from datetime import datetime
from typing import List


class PCC:
    @staticmethod
    def calculate_pcc(dates: List[datetime], stock_prices: List[float], fear_and_greed: List[int]) -> float:
        """
        Calculate the Pearson correlation coefficient (PCC) between stock prices and the fear and greed index.

        Args:
            dates (List[datetime]): List of dates for the data points.
            stock_prices (List[float]): List of stock prices corresponding to the dates.
            fear_and_greed (List[int]): List of fear and greed index values corresponding to the dates.

        Returns:
            float: The Pearson correlation coefficient (PCC) between stock prices and the fear and greed index.
        """
        # Check if all input lists have the same length
        if len(dates) != len(stock_prices) or len(dates) != len(fear_and_greed):
            raise ValueError("All input lists (dates, stock_prices, and fear_and_greed) must have the same length.")
        
        # Create a DataFrame
        data = {'Date': dates, 'StockPrice': stock_prices, 'FearAndGreed': fear_and_greed}
        df = pd.DataFrame(data)

        # Calculate the Pearson correlation coefficient
        correlation = np.corrcoef(df['StockPrice'], df['FearAndGreed'])[0, 1]

        return correlation

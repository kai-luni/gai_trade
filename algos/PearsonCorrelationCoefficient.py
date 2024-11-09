import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict

from dto.ExchangeRateItem import ExchangeRateItem


class PCC:
    @staticmethod
    def calculate_pcc(dates: List[datetime], stock_prices: List[float], fear_and_greed: List[int], fg_from: int, fg_to: int) -> float:
        """
        Calculate the Pearson correlation coefficient (PCC) between daily percentage changes in stock prices and the fear and greed index.

        Args:
            dates (List[datetime]): List of dates for the data points.
            stock_prices (List[float]): List of stock prices corresponding to the dates.
            fear_and_greed (List[int]): List of fear and greed index values corresponding to the dates.
            fg_from (int): Lower bound of the fear and greed index filter.
            fg_to (int): Upper bound of the fear and greed index filter.

        Returns:
            float: The Pearson correlation coefficient (PCC) between daily percentage changes in stock prices and the fear and greed index.
        """
        # Check if all input lists have the same length
        if len(dates) != len(stock_prices) or len(dates) != len(fear_and_greed):
            raise ValueError("All input lists (dates, stock_prices, and fear_and_greed) must have the same length.")

        dates, stock_price_changes, fear_and_greed = PCC.get_stock_changes_in_percent(dates, stock_prices, fear_and_greed)

        # Apply the fear_and_greed filter
        filtered_data = [(date, change, fg) for date, change, fg in zip(dates, stock_price_changes, fear_and_greed) if fg_from <= fg <= fg_to]
        #avoid exception when none found
        if len(filtered_data) == 0:
            return -10.
        filtered_dates, filtered_stock_price_changes, filtered_fear_and_greed = zip(*filtered_data)

        # Create a DataFrame
        data = {'Date': filtered_dates, 'StockPriceChange': filtered_stock_price_changes, 'FearAndGreed': filtered_fear_and_greed}
        df = pd.DataFrame(data)

        # Calculate the Pearson correlation coefficient
        correlation = np.corrcoef(df['StockPriceChange'], df['FearAndGreed'])[0, 1]

        return correlation
    
    def find_good_correlations(exchange_items: List[ExchangeRateItem], greed_items: List[Dict[str, int]]):
        """
        This function calculates the Pearson correlation coefficient between exchange rates and greed indexes
        for specified ranges of data and writes the results to a text file if the PCC value is greater than 0.4.

        Args:
            exchange_items (List[ExchangeRateItem]): A list of exchange rate items containing dates and opening prices.
            greed_items (List[Dict[str, int]]): A list of dictionaries containing greed indexes.

        Returns:
            None
        """

        # Extract dates, opening prices, and greed indexes from the input lists
        dates = [item.date for item in exchange_items]
        opens = [item.open for item in exchange_items]
        greed_indexes = [item["index"] for item in greed_items]

        # Open a file to write the results
        with open('results.txt', 'a') as file:
            # Loop through the specified ranges
            for i in range(100):
                for j in range(30):
                    # Calculate the PCC for the given range
                    pcc = PCC.calculate_pcc(dates, opens, greed_indexes, i, i+j)

                    # Write the result to the file if the PCC value is greater than 0.4
                    if pcc > 0.4:
                        file.write(f"Pearson correlation coefficient from {i} to {i+j} is {pcc}\n")     
    
    @staticmethod
    def get_stock_changes_in_percent(dates: List[datetime], stock_prices: List[float], fear_and_greed: List[int]):
                # Calculate daily percentage changes in stock prices
        stock_price_changes = [(stock_prices[i + 1] - stock_prices[i]) / stock_prices[i] * 100 for i in range(len(stock_prices) - 1)]

        # Remove the first date since there's no stock price change for the first day
        dates = dates[1:]
        fear_and_greed = fear_and_greed[1:]

        return dates, stock_price_changes, fear_and_greed

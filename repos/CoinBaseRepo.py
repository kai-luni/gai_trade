from ast import Dict
import csv
import time
import cryptocompare
import requests
import json
import os
from datetime import datetime, timedelta, timezone

from dto.ExchangeRateItem import ExchangeRateItem

class CoinBaseRepo:
    @staticmethod
    def fetch_daily_data(symbol, start, end, folder="repos"):
        """
        Fetch daily cryptocurrency data from CryptoCompare API, combine with existing data,
        and save the complete dataset to a CSV file.

        Parameters:
        symbol (str): The cryptocurrency symbol, e.g., 'BTC'.
        start (datetime): The start date for fetching data.
        end (datetime): The end date for fetching data.
        folder (str): The folder to save the CSV file.

        Returns:
        None
        """
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Calculate the number of days between start and end
        delta = end - start
        days = delta.days

        # Fetch historical daily data from CryptoCompare
        data = cryptocompare.get_historical_price_day(
            symbol,
            currency='EUR',
            limit=days,
            toTs=int(end.timestamp())
        )

        if data:
            # Define the CSV file path
            filename = f"{symbol}_EUR.csv"
            filepath = os.path.join(folder, filename)

            # Read existing data from CSV
            existing_data = CoinBaseRepo.read_csv_to_dict(filepath)
            existing_dates = CoinBaseRepo.get_exchange_rate_items(datetime(2010,1,1), datetime(2029,1,20), existing_data, throwExceptionMissingDay=False)

            # Prepare combined data, including existing and new entries
            all_data = []

            # Add existing data to all_data
            for item in existing_data:
                item_date = datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S')
                all_data.append([
                    int(item['unix']), float(item['low']), float(item['high']),
                    float(item['open']), float(item['close']), float(item['volume']),
                    item['date']
                ])

            # Add new data if it doesn't already exist in the current dates
            for entry in data:
                entry_date = datetime.fromtimestamp(entry['time'], tz=timezone.utc).date()
                if all(item.date.date() != entry_date for item in existing_dates):
                    datetime_string = datetime.fromtimestamp(entry['time'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    all_data.append([
                        entry['time'], entry['low'], entry['high'], entry['open'],
                        entry['close'], entry['volumefrom'], datetime_string
                    ])

            # Sort data by the timestamp (first element of each entry)
            all_data.sort(key=lambda x: x[0])

            # Rewrite the entire CSV file with the sorted data
            with open(filepath, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow(["unix", "low", "high", "open", "close", "volume", "date"])
                for row in all_data:
                    writer.writerow(row)

            print(f"Data has been saved to {filepath} with {len(all_data)} total entries.")
        else:
            print("No data retrieved from CryptoCompare API.")


    @staticmethod
    def get_exchange_rate_items(start_date: datetime, end_date: datetime, exchange_items: 'list[Dict]', throwExceptionMissingDay: bool = True) -> 'list[ExchangeRateItem]':
        """
        Filter exchange items by date range and convert them to ExchangeRateItem objects.
        Ensure that there is exactly one item per day, and raise an exception if an item is missing.

        Parameters:
        start_date (datetime): The start date for filtering exchange items.
        end_date (datetime): The end date for filtering exchange items.
        exchange_items (list): A list of dictionaries containing exchange rate data.

        Returns:
        list: A list of ExchangeRateItem objects filtered by the specified date range.
        """

        result = []
        days_difference = (end_date - start_date).days

        for i in range(days_difference + 1):
            current_date = start_date + timedelta(days=i)
            item_for_current_date = None

            for item in exchange_items:
                item_date = datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S')
                if item_date.date() == current_date.date():
                    item_for_current_date = item
                    break

            if item_for_current_date:
                exchange_rate_item = ExchangeRateItem(
                    unix=int(item_for_current_date['unix']),
                    date=item_date,
                    low=float(item_for_current_date['low']),
                    high=float(item_for_current_date['high']),
                    open=float(item_for_current_date['open']),
                    close=float(item_for_current_date['close']),
                    volume=float(item_for_current_date['volume'])
                )
                result.append(exchange_rate_item)
            elif throwExceptionMissingDay:
                raise Exception(f"Missing exchange rate data for date: {current_date.date()}")

        return result


    @staticmethod
    def read_csv_to_dict(filename) -> 'list[Dict]':
        """
        Read data from a CSV file and return a list of dictionaries.

        Parameters:
        filename (str): The name of the CSV file to read.

        Returns:
        list: A list of dictionaries containing the data from the CSV file.
        """

        data = []

        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')

            for row in reader:
                data.append(row)

        return data




if __name__ == "__main__":
    start = datetime(2024, 3, 1)
    end = datetime(2024, 4, 1)
    while start < datetime.now():
        CoinBaseRepo.fetch_daily_data('BTC', start, end)
        start = start + timedelta(days=30)
        end = end + timedelta(days=30)
        #sleep 1 second
        time.sleep(1)
        print(start)
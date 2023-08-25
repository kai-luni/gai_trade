from ast import Dict
import csv
import time
import requests
import json
import os
from datetime import datetime, timedelta, timezone as tz

from dto.ExchangeRateItem import ExchangeRateItem

class CoinBaseRepo:

    @staticmethod
    def fetch_daily_data(symbol, start, end, folder="api"):
        """
        Fetch daily currency data from Coinbase API and save it to a CSV file.

        Parameters:
        symbol (str): The trading pair in the format 'XXX/XXX', e.g. 'BTC/EUR'.
        start (datetime): The start date and time for fetching data.
        end (datetime): The end date and time for fetching data.

        Returns:
        None
        """
        # Split the symbol into its components and join them with a hyphen
        pair_split = symbol.split('/')
        symbol = pair_split[0] + '-' + pair_split[1]

        # Construct the API request URL
        url = f'https://api.pro.coinbase.com/products/{symbol}/candles?granularity=86400&start={start.astimezone(tz.utc).strftime("%Y-%m-%d %H:%M:%S")}&end={end.astimezone(tz.utc).strftime("%Y-%m-%d %H:%M:%S")}'

        response = requests.get(url)

        # Check to make sure the response from server is good
        if response.status_code == 200:
            # Create a CSV file or append to an existing one
            filename = f"{pair_split[0]}_{pair_split[1]}.csv"
            filepath = f"{folder}/{filename}"

            # Read existing data from the CSV file
            existing_data = {}
            if os.path.isfile(filepath):
                with open(filepath, "r") as csvfile:
                    reader = csv.reader(csvfile, delimiter=";")
                    next(reader)  # skip header
                    for row in reader:
                        if len(row) > 0:
                            existing_data[int(row[0])] = row

            # Update existing_data with new data from the API response
            candles = json.loads(response.text)
            for i in reversed(range(len(candles))):
                unix = int(candles[i][0])
                if unix not in existing_data:
                    datetime_obj = datetime.fromtimestamp(unix)
                    datetime_string = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
                    candle_data = [str(x) for x in candles[i]]
                    candle_data.append(datetime_string)
                    existing_data[unix] = candle_data

            # Sort the data by the unix timestamp
            sorted_data = sorted(existing_data.values(), key=lambda x: int(x[0]))

            # Write the sorted data to the CSV file
            with open(filepath, "w", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                writer.writerow(["unix", "low", "high", "open", "close", "volume", "date"])
                for row in sorted_data:
                    writer.writerow(row)
        else:
            print("Did not receive OK response from Coinbase API")


    @staticmethod
    def get_exchange_rate_items(start_date: datetime, end_date: datetime, exchange_items: 'list[Dict]') -> 'list[ExchangeRateItem]':
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
            else:
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





# start = datetime(2023, 4, 1)
# end = datetime(2023, 9, 29)
# while start < datetime.now():
#     CoinBaseRepo.fetch_daily_data('BTC/EUR', start, end)
#     start = start + timedelta(days=30)
#     end = end + timedelta(days=30)
#     #sleep 1 second
#     time.sleep(1)
#     print(start)
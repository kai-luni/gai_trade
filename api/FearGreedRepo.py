import csv
import json
import requests
from datetime import datetime

import pandas as pd

from dto.ObjectsGai import FearGreedItem

class FearGreedRepo:
    # Assuming you have an appropriate constructor and/or attributes for this class
    pass

    def get_data(limit=10, file_path="fear_greed_data.csv"):
        """Get daily fear greed index from alternative.me and append to a CSV file.

        Args:
            limit (int): days back from today. Defaults to 10.
            file_path (str): the path of the CSV file to append data to.
                
        Returns:
            None
        """

        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        response = requests.get(url)
        if response.status_code == 200:  # check to make sure the response from server is good
            candles = json.loads(response.text)["data"]

            # Read existing data from the CSV file
            existing_data = {}
            try:
                with open(file_path, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if len(row) > 0:
                            existing_data[int(row[0])] = row
            except FileNotFoundError:
                pass

            # Append new data to the CSV file
            with open(file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for candle in candles:
                    timestamp = int(candle["timestamp"])
                    if timestamp not in existing_data:
                        exchange_item = FearGreedItem()
                        exchange_item.unix = timestamp
                        exchange_item.index = int(candle["value"])
                        exchange_item.index_text = candle["value_classification"]
                        exchange_item.date = datetime.fromtimestamp(exchange_item.unix)
                        human_readable_date = exchange_item.date.strftime("%Y-%m-%d %H:%M:%S")

                        # Write the data to the CSV file
                        writer.writerow([exchange_item.unix, exchange_item.index, exchange_item.index_text, human_readable_date])
        else:
            print("Did not receive OK response from Coinbase API")


    def read_csv_file(start : datetime, end : datetime, file_path="fear_greed_data.csv"):
        """Reads the CSV file and returns the data as a list of dictionaries.
        Args:
            file_path (str): The path of the CSV file to read data from.
            start (str): The start date in the format 'YYYY-MM-DD' to filter the data.
            end (str): The end date in the format 'YYYY-MM-DD' to filter the data.

        Returns:
            list: A list of dictionaries containing the data.

        Raises:
            Exception: If there is a missing entry in the given date range.
        """

        data = []
        unique_dates = set()

        with open(file_path, "r") as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                if len(row) > 0:
                    timestamp = int(row[0])
                    date = datetime.fromtimestamp(timestamp).date()

                    if (start is None or date >= start.date()) and (end is None or date <= end.date()):
                        if date not in unique_dates:
                            unique_dates.add(date)
                            entry = {
                                "timestamp": timestamp,
                                "index": int(row[1]),
                                "index_text": row[2],
                                "human_readable_date": datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
                            }
                            data.append(entry)

        # Check for missing dates
        date_range = set(pd.date_range(start=start or min(unique_dates), end=end or max(unique_dates), freq='D').date)
        missing_dates = date_range - unique_dates
        if missing_dates:
            raise Exception(f"Missing data for the following dates: {', '.join(str(d) for d in missing_dates)}")

        return data


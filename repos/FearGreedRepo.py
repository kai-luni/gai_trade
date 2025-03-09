import csv
import json
import os
import requests
from datetime import datetime

import pandas as pd

from dto.ObjectsGai import FearGreedItem

class FearGreedRepo:
    # Assuming you have an appropriate constructor and/or attributes for this class
    pass

    def get_data(limit=10, file_path="repos/fear_greed_data.csv") -> 'list[dict]':
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
                    # Skip the header row
                    _ = next(reader, None)
                    for row in reader:
                        if len(row) > 0:
                            existing_data[int(row[0])] = row
            except FileNotFoundError:
                pass
            for candle in candles:
                timestamp = int(candle["timestamp"])
                if timestamp not in existing_data:
                    exchange_item = FearGreedItem()
                    exchange_item.unix = timestamp
                    exchange_item.index = int(candle["value"])
                    exchange_item.index_text = candle["value_classification"]
                    exchange_item.date = datetime.fromtimestamp(exchange_item.unix)
                    human_readable_date = exchange_item.date.strftime("%Y-%m-%d %H:%M:%S")

                    existing_data[exchange_item.unix] = [exchange_item.unix, exchange_item.index, exchange_item.index_text, human_readable_date]

            # Sort the data by the unix timestamp
            sorted_data = sorted(existing_data.values(), key=lambda x: int(x[0]))

            # Write the sorted data to the CSV file
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for row in sorted_data:
                    writer.writerow(row)
        else:
            print("Did not receive OK response from Coinbase API")



    def read_csv_file(start: datetime, end: datetime, file_path="repos/fear_greed_data.csv", check_data=False):
        """Reads the CSV file and returns the data as a list of FearGreedItem objects.
        
        Args:
            file_path (str): The path of the CSV file to read data from.
            start (datetime): The start date to filter the data.
            end (datetime): The end date to filter the data.
            check_data (bool): Whether to check for missing dates.

        Returns:
            list: A list of FearGreedItem objects.

        Raises:
            Exception: If there is a missing entry in the given date range.
        """
        data = []
        unique_dates = set()

        with open(file_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row if present

            for row in reader:
                if len(row) > 0:
                    timestamp = int(row[0])
                    date = datetime.fromtimestamp(timestamp).date()

                    if (start is None or date >= start.date()) and (end is None or date <= end.date()):
                        if date not in unique_dates:
                            unique_dates.add(date)
                            item = FearGreedItem(
                                unix=timestamp,
                                date=datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S"),
                                index=int(row[1]),
                                index_text=row[2]
                            )
                            data.append(item)
        
        if not check_data:
            return data

        # Check for missing dates
        date_range = set(pd.date_range(start=start or min(unique_dates), end=end or max(unique_dates), freq='D').date)
        missing_dates = date_range - unique_dates
        if missing_dates:
            raise Exception(f"Missing data for the following dates: {', '.join(str(d) for d in missing_dates)}")
        
        return data

if __name__ == "__main__":
    FearGreedRepo.get_data(1000)

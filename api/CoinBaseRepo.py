import csv
import requests
import json
import os
from datetime import datetime, timezone as tz

class CoinBaseRepo:
    def fetch_daily_data(symbol, start, end):
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
            write_or_append = "a" if os.path.isfile(filename) else "w"

            with open(filename, write_or_append) as outF:
                candles = json.loads(response.text)

                # Write the CSV header if creating a new file
                if write_or_append == "w":
                    outF.write("unix,low,high,open,close,volume,date\n")

                # Write the candle data in reversed order (oldest to newest)
                written_dates = set()
                for i in reversed(range(len(candles))):
                    datetime_obj = datetime.fromtimestamp(candles[i][0])

                    # Only insert one entry per day
                    if datetime_obj.date() not in written_dates:
                        datetime_string = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
                        candle_string = ";"
                        outF.write(candle_string.join([str(x) for x in candles[i]]))
                        outF.write(";" + datetime_string)
                        outF.write("\n")
                        written_dates.add(datetime_obj.date())
        else:
            print("Did not receive OK response from Coinbase API")

    def read_csv_to_dict(filename):
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

CoinBaseRepo.fetch_daily_data('BTC/EUR', datetime(2020, 1, 1), datetime(2020, 3, 17))
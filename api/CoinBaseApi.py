from datetime import datetime, timedelta
from dateutil import tz
import json
import os
import requests

print(os.getcwd())

from dto.ObjectsGai import ExchangeRateItem

class CoinBaseApi:
    def fetch_daily_data(symbol, start, end):
        pair_split = symbol.split('/')  # symbol must be in format XXX/XXX ie. BTC/EUR
        symbol = pair_split[0] + '-' + pair_split[1]
        url = f'https://api.pro.coinbase.com/products/{symbol}/candles?granularity=3600&start={start.astimezone(tz.UTC).strftime("%Y-%m-%d %H:%M:%S")}&end={end.astimezone(tz.UTC).strftime("%Y-%m-%d %H:%M:%S")}'
        print(url)
        response = requests.get(url)
        if response.status_code == 200:  # check to make sure the response from server is good
            filename = pair_split[0] + "csv"
            write_or_append = "a" if os.path.isfile(filename) else "w"
            outF = open(filename, write_or_append)
            candles = json.loads(response.text)
            if write_or_append == "w":
                outF.write("unix,low,high,open,close,volume,date\n")
            for i in reversed(range(len(candles))):
                datetime_string =  datetime.fromtimestamp(candles[i][0]).strftime('%Y-%m-%d %H:%M:%S')
                candle_string = ";"
                outF.write(candle_string.join([str(x) for x in candles[i]]))
                outF.write(";" + datetime_string)
                outF.write("\n")
            outF.close()
        else:
            print("Did not receieve OK response from Coinbase API")

    def getDataCoinbase(symbol : str, start : datetime, end : datetime, granularity : int = 86400):
        """Get daily currency data for a coin from coinbase

        Args:
            symbol (str): examples: LTC/EUR BTC/USD
            start (datetime): from date
            end (datetime): to date

        Returns:
            list: found ExchangeRateItems
        """
        pair_split = symbol.split('/')  # symbol must be in format XXX/XXX ie. BTC/EUR
        symbol = pair_split[0] + '-' + pair_split[1]
        url = f'https://api.pro.coinbase.com/products/{symbol}/candles?granularity={granularity}&start={start.astimezone(tz.UTC).strftime("%Y-%m-%d %H:%M:%S")}&end={end.astimezone(tz.UTC).strftime("%Y-%m-%d %H:%M:%S")}'
        response = requests.get(url)
        if response.status_code == 200:  # check to make sure the response from server is good
            return_list = []
            candles = json.loads(response.text)
            for candle in candles:
                #unix,low,high,open,close,volume,date
                candle.append(datetime.fromtimestamp(candle[0]).strftime('%Y-%m-%d %H:%M:%S'))

                exchange_item = ExchangeRateItem()
                exchange_item.unix = int(candle[0])
                exchange_item.low = float(candle[1])
                exchange_item.high = float(candle[2])
                exchange_item.open = float(candle[3])
                exchange_item.close = float(candle[4])
                exchange_item.volume = float(candle[5])
                exchange_item.date = datetime.fromtimestamp(candle[0])
                return_list.append(exchange_item)
            return return_list
        else:
            print("Did not receieve OK response from Coinbase API")
            return []

if __name__ == "__main__":
    start =  datetime(2021, 8, 1, 0, 0, 0, 0)
    end =  datetime(2021, 8, 1, 23, 59, 59, 0)
    data_entries : 'list[ExchangeRateItem]' = CoinBaseApi.getDataCoinbase("LTC/EUR", start, end, granularity=900)
    for entry in data_entries:
        print(entry.date)
from datetime import datetime
import json
import requests

from dto.ObjectsGai import FearGreedItem

class AlternativeFearGreedApi:
    def __init__(self, days_look_back = 10):
        self._all_entries : 'list[FearGreedItem]' = self.get_data(days_look_back)

    def get_data(self, limit = 10):
        """Get daily fear greed index from alternative.me

        Args:
            limit (int): days back from today. Defaults to 10.
            
        Returns:
            list: found FearGreedItems
        """

        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        response = requests.get(url)
        if response.status_code == 200:  # check to make sure the response from server is good
            return_list = []
            candles = json.loads(response.text)["data"]
            for candle in candles:
                exchange_item = FearGreedItem()
                exchange_item.unix = int(candle["timestamp"])
                exchange_item.index = int(candle["value"])
                exchange_item.index_text = candle["value_classification"]
                exchange_item.date = datetime.fromtimestamp(exchange_item.unix)
                return_list.append(exchange_item)
            return return_list
        else:
            print("Did not receieve OK response from Coinbase API")
            return []

    def get_entry_for_day(self, day : int, month : int, year : int):
        """find day entry for given date

        Args:
            day (int): day
            month (int): month
            year (int): year

        Returns:
            list: day entry: date, greed fear index, description
        """
        for day_entry in self._all_entries:
            if day_entry.date.day != day:
                continue
            if day_entry.date.month != month:
                continue
            if day_entry.date.year != year:
                continue
            return day_entry
        return None

if __name__ == "__main__":
    AlternativeFearGreedApi.get_data(365)
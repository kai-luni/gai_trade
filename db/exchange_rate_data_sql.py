
from datetime import datetime, timedelta
import sqlite3

from ObjectsGai import ExchangeRateItem
from CoinBaseApi import CoinBaseApi

class ExchangeRateData:
    table_name = "ExchangeRateData"

    def add_day_entry(day_data : ExchangeRateItem, full_path : str):
        """Add an ExchangeRateItem to db, can be empty db as well

        Args:
            day_data (ExchangeRateItem): represent one day of excange rate
            full_path (str): full path to db

        Returns:
            bool: success, false if exists already
        """
        conn = sqlite3.connect(full_path)
        conn.execute(ExchangeRateItem.get_sqlite_create_table_query())
        conn.commit()
        conn.close()

        exchange_items = ExchangeRateData.get_all_items(full_path)
        for i in range(len(exchange_items)):
            item = exchange_items[i]
            #found item to update
            if item.date.year == day_data.date.year and item.date.month == day_data.date.month and item.date.day == day_data.date.day:
                exchange_items[i] = day_data
                #item exists already
                return False
        ExchangeRateData.insert_item(day_data, full_path)
        return True
        

    def delete_all_items(full_path : str):
        """Delete all entries for this table

        Args:
            full_path (str): full path to db
        """
        conn = sqlite3.connect(full_path)
        # delete all rows from table
        conn.execute(f"DELETE FROM {ExchangeRateData.table_name};")
        conn.commit()
        conn.close()

    def fillDb(coin : str, currency : str, from_date : datetime, full_path : str):
        """Try to fill every day in db with coin/currency values.

        Args:
            coin (str): BTC, ETH...
            currency (str): EUR, USD...
            from_date (datetime): start from date
            path (str): full path to db

        Returns:
            bool: success
        """

        start =  from_date
        end =  datetime(from_date.year, from_date.month, from_date.day, 23, 59, 59, 0)
        while (start - timedelta(days=1)) < datetime.now():
            data = CoinBaseApi.getDataCoinbaseDaily(f"{coin}/{currency}", start, end)
            if len(data) == 0:
                print(f"No entry for {start}")
                start = start  + timedelta(days=1)
                end = end  + timedelta(days=1)
                continue
            if len(data) > 1:
                print(f"There was more than one entry for {start}")
                start = start  + timedelta(days=1)
                end = end  + timedelta(days=1)
                continue
            if not ExchangeRateData.add_day_entry(data[0], full_path):
                return False
            start = start  + timedelta(days=1)
            end = end  + timedelta(days=1)
        return True

    def get_all_items(full_path):
        """get all entries from db

        Args:
            full_path (str): path to db

        Returns:
            list[ExchangeRateItem]: entries from db
        """
        conn = sqlite3.connect(full_path)
        cursor = conn.execute(f"SELECT {ExchangeRateItem.get_sqlite_headers()} from ExchangeRateData order by unix asc")
        return_list = []
        for row in cursor:
            item = ExchangeRateItem()
            item.unix = row[0]
            item.low = row[1]
            item.high = row[2]
            item.open = row[3]
            item.close = row[4]
            item.volume = row[5]
            item.twenty_week_average = row[6]
            item.date = datetime.fromtimestamp(item.unix)
            return_list.append(item)
        return return_list

    def getDbFullPath(coin, currency, path):
        filename = f"{coin}_{currency}_echange_db.sqlite"
        return f"{path}{filename}"

    def insert_item(item : ExchangeRateItem, full_path):
        #TODO: NONE issue
        conn = sqlite3.connect(full_path)
        conn.execute(ExchangeRateItem.get_sqlite_create_table_query())
        conn.commit()
        query = f"INSERT INTO ExchangeRateData ({ExchangeRateItem.get_sqlite_headers()}) VALUES ({item.unix}, {item.low}, {item.high}, {item.open}, {item.close}, {item.volume}, 'null' )"
        conn.execute(query)    
        conn.commit()
        conn.close()

if __name__ == "__main__":
    coin = "BTC"
    currency = "EUR"
    start =  datetime(2013, 8, 1, 0, 0, 0, 0)
    full_path = ExchangeRateData.getDbFullPath(coin, currency, "")
    ExchangeRateData.fillDb(coin, currency, start, full_path)


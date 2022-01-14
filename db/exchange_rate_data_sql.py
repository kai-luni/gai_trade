
from datetime import datetime, timedelta
import os
import sqlite3

from dto.ObjectsGai import ExchangeRateItem
from api.CoinBaseApi import CoinBaseApi

class ExchangeRateData:
    table_name = "ExchangeRateData"

    def add_entry(day_data : ExchangeRateItem, full_path : str):
        """Add an ExchangeRateItem to db, can be empty db as well
            One entry per hour max, no update yet (when there is n entry in an hour it stays)

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
            if item.date.year == day_data.date.year and item.date.month == day_data.date.month and item.date.day == day_data.date.day and item.date.hour == day_data.date.hour:
                #item exists already TODO: update
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

    def fillDb(coin : str, currency : str, from_date : datetime, full_path : str, granularity : int = 86400):
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
            print(f"Date: {start}")
            data_entries = CoinBaseApi.getDataCoinbase(f"{coin}/{currency}", start, end, granularity=granularity)
            if len(data_entries) == 0:
                print(f"No entry for {start}")
                start = start  + timedelta(days=1)
                end = end  + timedelta(days=1)
                continue
            for entry in data_entries:
                if not ExchangeRateData.add_entry(entry, full_path):
                    print(f"{entry.date} exists already in db.")
            start = start  + timedelta(days=1)
            end = end  + timedelta(days=1)
        return True

    def filter_exchange_items(start_date : datetime, end_date : datetime, exchange_items: 'list[ExchangeRateItem]'):
        """filter the given items by datetime

        Args:
            start_date (datetime): minimum datetime
            end_date (datetime): maximum datetime
            exchange_items (list[ExchangeRateItem]): list of exchange items

        Returns:
            list[ExchangeRateItem]: list of filtered exchange items
        """
        
        found_items = []
        search_first_item = True
        for item in exchange_items:
            if search_first_item:
                if item.date.year < start_date.year:
                    continue
                if item.date.month < start_date.month:
                    continue
                if item.date.day < start_date.day:
                    continue
                # if item.date.hour < start_date.hour:
                #     continue
                search_first_item = False
            if item.date > end_date:
                return found_items
            found_items.append(item)
        return found_items

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
            item.low = row[2]
            item.high = row[3]
            item.open = row[4]
            item.close = row[5]
            item.volume = row[6]
            item.twenty_week_average = row[7]
            item.date = datetime.fromtimestamp(item.unix)
            return_list.append(item)
        return return_list

    def getDbFullPath(coin : str, currency : str, path : str, granularity : int = 86400):
        """get path to db

        Args:
            coin (str): par example BTC
            currency (str): par example ETH
            path (str): relative path, put / after folder as well

        Returns:
            str: complete filepath (relative)
        """
        #try to create missing folder
        if not os.path.isdir(path):
            os.mkdir(path)
        filename = f"{coin}_{currency}_{granularity}_echange_db.sqlite"
        return f"{path}{filename}"

    def insert_item(item : ExchangeRateItem, full_path : str):
        """simple insert into db

        Args:
            item (ExchangeRateItem): entry to insert
            full_path (str): file path
        """
        #TODO: NONE issue
        conn = sqlite3.connect(full_path)
        conn.execute(ExchangeRateItem.get_sqlite_create_table_query())
        conn.commit()
        query = f"INSERT INTO ExchangeRateData ({ExchangeRateItem.get_sqlite_headers()}) VALUES ({item.unix}, '{str(item.date)}', {item.low}, {item.high}, {item.open}, {item.close}, {item.volume}, 'null' )"
        conn.execute(query)    
        conn.commit()
        conn.close()

if __name__ == "__main__":
    coin = "DASH"
    currency = "EUR"
    start =  datetime(2018, 1, 1, 0, 0, 0, 0)
    granularity = 86400
    full_path = ExchangeRateData.getDbFullPath(coin, currency, "db/", granularity=granularity)
    ExchangeRateData.fillDb(coin, currency, start, full_path, granularity=granularity)
    # items = ExchangeRateData.get_all_items(full_path)
    # outF = open("test.csv", "w")
    # outF.write("unix,low,high,open,close,volume,date\n")
    # for ex_item in items:
    #     line_to_write = f"{ex_item.unix};{ex_item.low};{ex_item.high};{ex_item.open};{ex_item.close};{ex_item.volume};{ex_item.date}\n"
    #     outF.write(line_to_write)
    # outF.close()


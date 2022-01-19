from datetime import datetime, timedelta
import os
import sqlite3

from dto.ObjectsGai import FearGreedItem
from api.alternative_fear_greed_api import AlternativeFearGreedApi

class FearGreedData:
    def __init__(self, full_path):
        self.full_path = full_path
        self.table_name = FearGreedItem.get_table_name()

    def delete_all_items(self):
        """Delete all entries for this table

        """
        conn = sqlite3.connect(self.full_path)
        # delete all rows from table
        conn.execute(f"DELETE FROM {self.table_name};")
        conn.commit()
        conn.close()

    def fill_db(self, days_back : int):
        """Try to fill every day in db with coin/currency values.

        Args:
            days_back (str): BTC, ETH...

        Returns:
            bool: success
        """
        data_entries = AlternativeFearGreedApi.getData(days_back)
        if len(data_entries) == 0:
            print("No entries found")
        for entry in data_entries:
            if not self.insert_item_one_per_day(entry):
                print(f"{entry.date} exists already in db.")
        return True

    def get_all_items(self):
        """get all entries from db

        Returns:
            list[ExchangeRateItem]: entries from db
        """
        conn = sqlite3.connect(self.full_path)
        cursor = conn.execute(f"SELECT {FearGreedItem.get_sqlite_headers()} from {self.table_name} order by unix asc")
        return_list = []
        for row in cursor:
            item = FearGreedItem()
            item.unix = int(row[0])
            item.index = int(row[2])
            item.index_text = row[3]
            item.date = datetime.fromtimestamp(item.unix)
            return_list.append(item)
        return return_list

    def get_db_full_path(path : str):
        """get path to db

        Args:
            path (str): relative path, put / after folder as well

        Returns:
            str: complete filepath (relative)
        """
        #try to create missing folder
        if not os.path.isdir(path):
            os.mkdir(path)
        filename = f"fear_greed_db.sqlite"
        return f"{path}{filename}"

    def insert_item(self, item : FearGreedItem):
        """simple insert into db

        Args:
            item (ExchangeRateItem): entry to insert
            full_path (str): file path
        """
        #TODO: NONE issue
        conn = sqlite3.connect(self.full_path)
        
        conn.execute(FearGreedItem.get_sqlite_create_table_query())
        conn.commit()
        query = f"INSERT INTO {self.table_name} ({FearGreedItem.get_sqlite_headers()}) VALUES ({item.unix}, '{str(item.date)}', {item.index}, '{item.index_text}' )"
        conn.execute(query)    
        conn.commit()
        conn.close()

    def insert_item_one_per_day(self, day_data : FearGreedItem):
        """Add an entry to db, can be empty db as well
            One entry per day max, no update yet (when there is an entry in an day it stays)

        Args:
            day_data (FearGreedItem): represent one day of excange rate

        Returns:
            bool: success, false if exists already
        """
        conn = sqlite3.connect(self.full_path)
        conn.execute(FearGreedItem.get_sqlite_create_table_query())
        conn.commit()
        conn.close()

        exchange_items = self.get_all_items()
        for i in range(len(exchange_items)):
            item = exchange_items[i]
            #found item to update
            if item.date.year == day_data.date.year and item.date.month == day_data.date.month and item.date.day == day_data.date.day:
                #item exists already TODO: update
                return False
        self.insert_item(day_data)
        return True

if __name__ == "__main__":
    path_db = FearGreedData.get_db_full_path("db/")
    fear_greed_data = FearGreedData(path_db)
    fear_greed_data.fill_db(5000)

import sqlite3

from ObjectsGai import ExchangeRateItem

class ExchangeRateData:
    def add_day_entry(coin : str, currency : str, day_data : ExchangeRateItem, path : str = "./"):
        """Add an ExchangeRateItem to db, can be empty db as well

        Args:
            coin (str): coin name: BTC, ETH ..
            currency (str): USD, EUR ...
            day_data (ExchangeRateItem): represent one day of excange rate
            path (str): path to db, make sure to have "/" at the end

        Returns:
            bool: success
        """
        conn = sqlite3.connect(ExchangeRateData.getDbFullPath(coin, currency, path))
        conn.execute(ExchangeRateItem.get_sqlite_create_table_query())
        conn.commit()
        conn.close()

    def get_all_items(coin, currency, path):
        conn = sqlite3.connect(ExchangeRateData.getDbFullPath(coin, currency, path))
        cursor = conn.execute(f"SELECT {ExchangeRateItem.get_sqlite_headers()} from ExchangeRateData")
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
            return_list.append(item)
        return return_list

    def getDbFullPath(coin, currency, path):
        filename = f"{coin}_{currency}_echange_db.sqlite"
        return f"{path}{filename}"

    def insert_item(item : ExchangeRateItem, coin, currency, path):
        #TODO: NONE issue
        conn = sqlite3.connect(ExchangeRateData.getDbFullPath(coin, currency, path))
        conn.execute(ExchangeRateItem.get_sqlite_create_table_query())
        conn.commit()
        query = f"INSERT INTO ExchangeRateData ({ExchangeRateItem.get_sqlite_headers()}) VALUES ({item.unix}, {item.low}, {item.high}, {item.open}, {item.close}, {item.volume}, 'null' )"
        conn.execute(query)    
        conn.commit()
        conn.close()

from datetime import datetime, timezone
import os
import time
import unittest

from db.exchange_rate_data_sql import ExchangeRateData
from ObjectsGai import ExchangeRateItem


class TestExchangeRateDataSql(unittest.TestCase):
    def test_insert_item(self):
        entry = ExchangeRateItem()
        entry.high = 1.
        entry.low = .5
        path_db = ExchangeRateData.getDbFullPath("BTC", "EUR", "test_db")
        if os.path.isfile(path_db):
            ExchangeRateData.delete_all_items(path_db)
        ExchangeRateData.insert_item(entry, path_db)
        items = ExchangeRateData.get_all_items(path_db)
        self.assertEqual(1, len(items))

    def test_insert_two_day_entries_same_date(self):
        entry_one, entry_two = ExchangeRateItem(), ExchangeRateItem()
        entry_one.high = 1.
        entry_one.low = .5
        entry_one.unix = time.mktime(datetime(2021, 9, 26, tzinfo=timezone.utc).timetuple())
        entry_one.date = datetime(2021, 9, 26)
        entry_two.high = 1.
        entry_two.low = .5
        entry_two.unix = time.mktime(datetime(2021, 9, 26, tzinfo=timezone.utc).timetuple())
        entry_two.date = datetime(2021, 9, 26)

        path_db = ExchangeRateData.getDbFullPath("BTC", "EUR", "test_db")
        if os.path.isfile(path_db):
            ExchangeRateData.delete_all_items(path_db)

        ExchangeRateData.add_day_entry(entry_one, path_db)
        ExchangeRateData.add_day_entry(entry_two, path_db)

        items = ExchangeRateData.get_all_items(path_db)
        self.assertEqual(1, len(items))

    def test_insert_two_day_entries_different_date(self):
        entry_one, entry_two = ExchangeRateItem(), ExchangeRateItem()
        entry_one.high = 1.
        entry_one.low = .5
        entry_one.unix = time.mktime(datetime(2021, 9, 26).timetuple())
        entry_two.high = 1.
        entry_two.low = .5
        entry_two.unix = time.mktime(datetime(2021, 9, 27).timetuple())

        path_db = ExchangeRateData.getDbFullPath("BTC", "EUR", "test_db")
        if os.path.isfile(path_db):
            ExchangeRateData.delete_all_items(path_db)

        ExchangeRateData.add_day_entry(entry_one, path_db)
        ExchangeRateData.add_day_entry(entry_two, path_db)

        items = ExchangeRateData.get_all_items(path_db)
        self.assertEqual(2, len(items))

    def test_get_all_entries_in_order(self):
        entry_one, entry_two, entry_three = ExchangeRateItem(), ExchangeRateItem(), ExchangeRateItem()
        entry_one.high = 1.
        entry_one.low = .5
        entry_one.unix = time.mktime(datetime(2021, 9, 26).timetuple())
        entry_two.high = 1.
        entry_two.low = .7
        entry_two.unix = time.mktime(datetime(2021, 9, 28).timetuple())
        entry_three.high = 1.
        entry_three.low = .6
        entry_three.unix = time.mktime(datetime(2021, 9, 27).timetuple())

        path_db = ExchangeRateData.getDbFullPath("BTC", "EUR", "test_db")
        if os.path.isfile(path_db):
            ExchangeRateData.delete_all_items(path_db)

        ExchangeRateData.add_day_entry(entry_one, path_db)
        ExchangeRateData.add_day_entry(entry_two, path_db)
        ExchangeRateData.add_day_entry(entry_three, path_db)

        items = ExchangeRateData.get_all_items(path_db)
        self.assertGreater(items[1].unix, items[0].unix)
        self.assertGreater(items[2].unix, items[1].unix)

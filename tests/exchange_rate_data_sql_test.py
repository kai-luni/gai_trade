from datetime import datetime, timezone
import os
import time
import unittest

from db.exchange_rate_data_sql import ExchangeRateData
from dto.ObjectsGai import ExchangeRateItem


class TestExchangeRateDataSql(unittest.TestCase):
    def test_insert_item(self):
        entry = ExchangeRateItem()
        entry.high = 1.
        entry.low = .5
        path_db = ExchangeRateData.getDbFullPath("BTC", "EUR", "test_db/")
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

        ExchangeRateData.add_entry(entry_one, path_db)
        ExchangeRateData.add_entry(entry_two, path_db)

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

        ExchangeRateData.add_entry(entry_one, path_db)
        ExchangeRateData.add_entry(entry_two, path_db)

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

        ExchangeRateData.add_entry(entry_one, path_db)
        ExchangeRateData.add_entry(entry_two, path_db)
        ExchangeRateData.add_entry(entry_three, path_db)

        items = ExchangeRateData.get_all_items(path_db)
        self.assertGreater(items[1].unix, items[0].unix)
        self.assertGreater(items[2].unix, items[1].unix)

    def test_get_item_with_correct_date(self):
        """check if inserted time in unix time is correctly given back as datetime
        """
        hour = 15
        minute = 10

        entry_one = ExchangeRateItem()
        entry_one.high = 1.
        entry_one.low = .5
        entry_one.unix = time.mktime(datetime(2021, 9, 26, hour, minute, 0).timetuple())

        path_db = ExchangeRateData.getDbFullPath("BTC", "EUR", "test_db")
        if os.path.isfile(path_db):
            ExchangeRateData.delete_all_items(path_db)
        ExchangeRateData.add_entry(entry_one, path_db)

        items = ExchangeRateData.get_all_items(path_db)

        self.assertEqual(len(items), 1)
        item : ExchangeRateItem = items[0]
        self.assertEqual(item.date.hour, hour)
        self.assertEqual(item.date.minute, minute)


    def test_get_item_with_correct_low_high(self):
        """check if inserted time in unix time is correctly given back as datetime
        """
        low = .5
        high = 1.

        entry_one = ExchangeRateItem()
        entry_one.low = low
        entry_one.high = high
        entry_one.unix = time.mktime(datetime(2021, 9, 26).timetuple())

        path_db = ExchangeRateData.getDbFullPath("BTC", "EUR", "test_db")
        if os.path.isfile(path_db):
            ExchangeRateData.delete_all_items(path_db)
        ExchangeRateData.add_entry(entry_one, path_db)

        items = ExchangeRateData.get_all_items(path_db)

        self.assertEqual(len(items), 1)
        item : ExchangeRateItem = items[0]
        self.assertEqual(item.low, low)
        self.assertEqual(item.high, high)

    def test_filter_items(self):
        entry_one, entry_two = ExchangeRateItem(), ExchangeRateItem()
        entry_one.high = 1.
        entry_one.low = .5
        entry_one.unix = time.mktime(datetime(2021, 9, 26, 23, 59, 59).timetuple())
        entry_two.high = 2.
        entry_two.low = .7
        entry_two.unix = time.mktime(datetime(2021, 9, 28).timetuple())

        path_db = ExchangeRateData.getDbFullPath("BTC", "EUR", "test_db")
        if os.path.isfile(path_db):
            ExchangeRateData.delete_all_items(path_db)
        ExchangeRateData.add_entry(entry_one, path_db)
        ExchangeRateData.add_entry(entry_two, path_db)

        items = ExchangeRateData.get_all_items(path_db)
        filtered_items = ExchangeRateData.filter_exchange_items(datetime(2021, 9, 26), datetime(2021, 9, 26, 23, 59, 59), items)

        self.assertEqual(len(filtered_items), 1)
        self.assertEqual(filtered_items[0].low, .5)





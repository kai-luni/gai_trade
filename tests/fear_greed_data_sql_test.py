from datetime import datetime, timezone
import os
import time
import unittest

from db.fear_greed_data_sql import FearGreedData
from dto.ObjectsGai import FearGreedItem


class TestFearGreedDataSql(unittest.TestCase):
    def test_insert_item(self):
        path_db = FearGreedData.get_db_full_path("test_db/")
        fear_greed_data = FearGreedData(path_db)

        entry = FearGreedItem()
        entry.index = 144
        entry.index_text = "onefourfour"

        if os.path.isfile(path_db):
            fear_greed_data.delete_all_items()
        fear_greed_data.insert_item(entry)
        items = fear_greed_data.get_all_items()
        self.assertEqual(1, len(items))

    def test_insert_two_day_entries_same_date(self):
        path_db = FearGreedData.get_db_full_path("test_db/")
        fear_greed_data = FearGreedData(path_db)

        entry_one, entry_two = FearGreedItem(), FearGreedItem()
        entry_one.index = 11
        entry_one.unix = time.mktime(datetime(2021, 9, 26, tzinfo=timezone.utc).timetuple())
        entry_one.date = datetime(2021, 9, 26)
        entry_two.index = 111
        entry_two.unix = time.mktime(datetime(2021, 9, 26, tzinfo=timezone.utc).timetuple())
        entry_two.date = datetime(2021, 9, 26)

        if os.path.isfile(path_db):
            fear_greed_data.delete_all_items()

        fear_greed_data.insert_item_one_per_day(entry_one)
        fear_greed_data.insert_item_one_per_day(entry_two)

        items = fear_greed_data.get_all_items()
        self.assertEqual(1, len(items))


    def test_get_all_entries_in_order(self):
        path_db = FearGreedData.get_db_full_path("test_db/")
        fear_greed_data = FearGreedData(path_db)

        entry_one, entry_two, entry_three = FearGreedItem(), FearGreedItem(), FearGreedItem()
        entry_one.index = 123
        entry_one.unix = time.mktime(datetime(2021, 9, 26).timetuple())
        entry_two.index = 23
        entry_two.unix = time.mktime(datetime(2021, 9, 28).timetuple())
        entry_three.index = 13
        entry_three.unix = time.mktime(datetime(2021, 9, 27).timetuple())

        if os.path.isfile(path_db):
            fear_greed_data.delete_all_items()

        fear_greed_data.insert_item_one_per_day(entry_one)
        fear_greed_data.insert_item_one_per_day(entry_two)
        fear_greed_data.insert_item_one_per_day(entry_three)

        items = fear_greed_data.get_all_items()
        self.assertGreater(items[1].unix, items[0].unix)
        self.assertGreater(items[2].unix, items[1].unix)


    def test_get_item_with_correct_date(self):
        """check if inserted time in unix time is correctly given back as datetime
        """
        path_db = FearGreedData.get_db_full_path("test_db/")
        fear_greed_data = FearGreedData(path_db)

        hour = 15
        minute = 10

        entry_one = FearGreedItem()
        entry_one.index = 456
        entry_one.unix = time.mktime(datetime(2021, 9, 26, hour, minute, 0).timetuple())

        if os.path.isfile(path_db):
            fear_greed_data.delete_all_items()
        fear_greed_data.insert_item_one_per_day(entry_one)

        items = fear_greed_data.get_all_items()

        self.assertEqual(len(items), 1)
        item : FearGreedItem = items[0]
        self.assertEqual(item.date.hour, hour)
        self.assertEqual(item.date.minute, minute)


    def test_get_item_with_correct_index_text(self):
        """check if inserted time in unix time is correctly given back as datetime
        """
        path_db = FearGreedData.get_db_full_path("test_db/")
        fear_greed_data = FearGreedData(path_db)
        index = 5
        index_text = "text"

        entry_one = FearGreedItem()
        entry_one.index = index
        entry_one.index_text = index_text
        entry_one.unix = time.mktime(datetime(2021, 9, 26).timetuple())

        if os.path.isfile(path_db):
            fear_greed_data.delete_all_items()
        fear_greed_data.insert_item_one_per_day(entry_one)

        items = fear_greed_data.get_all_items()

        self.assertEqual(len(items), 1)
        item : FearGreedItem = items[0]
        self.assertEqual(item.index, index)
        self.assertEqual(item.index_text, index_text)
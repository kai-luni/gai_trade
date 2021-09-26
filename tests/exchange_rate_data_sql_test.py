import unittest

from db.exchange_rate_data_sql import ExchangeRateData
from ObjectsGai import ExchangeRateItem


class TestExchangeRateDataSql(unittest.TestCase):
    def test_insert_item(self):
        entry = ExchangeRateItem()
        entry.high = 1.
        entry.low = .5
        ExchangeRateData.insert_item(entry, "BTC", "EUR", "test_db")
        items = ExchangeRateData.get_all_items("BTC", "EUR", "test_db")
        self.assertEqual(1, len(items))

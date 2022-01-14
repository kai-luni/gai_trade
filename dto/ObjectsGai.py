from datetime import datetime

class TradeBuy:
    currency = ""
    amount = 0.
    buy_date_utc = datetime(1, 1, 1, 1, 1, 1, 1)
    buy_at_euro = 0.
    buy_text = ""
    sell_at_euro = 0.
    sell_text = ""
    sell_date_utc = datetime(1, 1, 1, 1, 1, 1, 1)
    trade_id_sell = ""
    trade_id_buy = ""
    done = False

class TradeSell:
    currency = ""
    amount = 0.
    buy_back_at = 0.
    sell_date_utc = datetime(1, 1, 1, 1, 1, 1, 1)
    sell_at_euro = 0.
    sell_text = ""
    buy_at_euro = 0.
    buy_text = ""
    buy_date_utc = datetime(1, 1, 1, 1, 1, 1, 1)
    done = False

class TradeSellParams:
    def __init__(self, above_twenty=1.02, days_buy=[1,12,24], buy_times_per_month=3, coin_name="BTC", days_between_sales=30, days_look_back=56, percent_change_sell=1.05, sell_above_20_average=True, sell_part=1., spend_part=1., start=datetime(2016, 5, 10, 16, 0, 0, 0), end=datetime(2021, 7, 6, 0, 0, 0, 0)):
        self.above_twenty = above_twenty
        self.buy_times_per_month = buy_times_per_month
        self.coin_name = coin_name
        self.days_buy = days_buy
        self.days_between_sales = days_between_sales
        self.days_look_back = days_look_back
        self.end = end
        self.percent_change_sell = percent_change_sell
        self.sell_above_20_average = sell_above_20_average
        self.sell_part = sell_part
        self.spend_part = spend_part
        self.start = start
    #factor above twenty week average to sell
    above_twenty = 1.05

    buy_at_gfi = 20
    sell_at_gfi = 80

    coin = "BTC"

    days_buy = []

    #how many days at least between two sales
    days_between_sales = 30

    #how many days back to take into consideration for algo
    days_look_back = 1

    #until what day
    end = datetime(1, 1, 1, 1, 1, 1, 1)
    
    #how much percent increase in timeframe to sell
    percent_change_sell = 1.3

    #sell only above 20 week average
    sell_above_20_average = False

    #sell how much from owned crypto
    sell_part = 0.9

    #how much to spend from the money that we own on buy
    spend_part = 0.8

    #from what day
    start = datetime(1, 1, 1, 1, 1, 1, 1)


class ExchangeRateItem:
    unix = 0
    date = datetime(1, 1, 1, 1, 1, 1, 1)
    low = 0.
    high = 0.
    open = 0.
    close = 0.
    volume = 0.
    #twenty week average, might be None
    twenty_week_average = None

    def get_sqlite_create_table_query():
        sql ='''CREATE TABLE IF NOT EXISTS ExchangeRateData(
            unix INT NOT NULL,
            date STRING NOT NULL,
            low FLOAT NOT NULL,
            high FLOAT NOT NULL,
            open FLOAT NOT NULL,
            close FLOAT NOT NULL,
            volume FLOAT NOT NULL,
            twenty_week_average
            )'''
        return sql

    def get_sqlite_headers():
        return "unix,date,low,high,open,close,volume,twenty_week_average"
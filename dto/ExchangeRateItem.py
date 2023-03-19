import datetime


class ExchangeRateItem:
    def __init__(self, unix : float, date : datetime, low: float, high : float, open : float, close : float, volume: float):
        self.unix = unix
        self.date = date
        self.low = low
        self.high = high
        self.open = open
        self.close = close
        self.volume = volume
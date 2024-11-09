from datetime import datetime

from old import ExchangeRateData


if __name__ == "__main__":
    coin = "BTC"
    currency = "EUR"
    start =  datetime(2013, 8, 1, 0, 0, 0, 0)
    full_path = ExchangeRateData.getDbFullPath(coin, currency, "")

    #fill db
    #ExchangeRateData.fillDb(coin, currency, start, full_path)

    #check db for gaps
    day_entries : 'list[ExchangeRateItem]' = ExchangeRateData.get_all_items(full_path)
    last_date = None
    for day_entry in day_entries:
        print(f"{day_entry.date}: {day_entry.close} and {day_entry.volume}")
        if day_entry.close < 1 or day_entry.volume < 1:
            print("not good")


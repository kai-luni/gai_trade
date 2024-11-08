from datetime import datetime, timedelta
import os

from numpy import append

from dto.ObjectsGai import ExchangeRateItem
from old.CoinBaseApi import CoinBaseApi


class ExchangeRateData:
    def addDayToDb(coin : str, currency : str, day_data : ExchangeRateItem, path : str = "./"):
        """Add an ExchangeRateItem to db, can be empty db as well

        Args:
            coin (str): coin name: BTC, ETH ..
            currency (str): USD, EUR ...
            day_data (ExchangeRateItem): represent one day of excange rate
            path (str): path to db, make sure to have "/" at the end

        Returns:
            bool: success
        """
        filename = f"{coin}_{currency}_echange_db.csv"
        exchange_items = ExchangeRateData.get_exchange_items(f"{path}{filename}")
        for i in range(len(exchange_items)):
            item = exchange_items[i]
            #found item to update
            if item.date.year == day_data.date.year and item.date.month == day_data.date.month and item.date.day == day_data.date.day:
                exchange_items[i] = day_data
                break
            #found gap to insert
            if i > 0 and exchange_items[i-1].date < day_data.date and exchange_items[i].date > day_data.date:
                exchange_items.insert(i, day_data)
                break
            #insert at beginning
            if i == 0 and exchange_items[i].date > day_data.date:
                exchange_items.insert(i, day_data)
                break
            #append at end
            if exchange_items[i].date < day_data.date and i == (len(exchange_items) -1):
                exchange_items.append(day_data)
                break
            if exchange_items[i].date > day_data.date:
                print(f"At index {i} it was noticed that the ExchangeRateItem could not be inserted")
                return False
        #append if empty yet
        if(len(exchange_items) == 0):
            exchange_items.append(day_data)

        outF = open(f"{path}{filename}", "w")
        outF.write("unix,low,high,open,close,volume,date\n")
        for ex_item in exchange_items:
            line_to_write = f"{ex_item.unix};{ex_item.low};{ex_item.high};{ex_item.open};{ex_item.close};{ex_item.volume};{ex_item.date};{ex_item.twenty_week_average}\n"
            outF.write(line_to_write)
        outF.close()

        return True

        # lines = []
        # if os.path.isfile(filename):
        #     file1 = open(filename, 'r')
        #     lines = file1.readlines()
        #     for i in range(len(lines)):

        # write_or_append = "a" if os.path.isfile(filename) else "w"

    def fillDb(coin : str, currency : str, from_date : datetime, path : str = "./"):
        """Try to fill every day in db with coin/currency values.

        Args:
            coin (str): BTC, ETH...
            currency (str): EUR, USD...
            from_date (datetime): start from date
            path (str): path to db, make sure to have "/" at the end

        Returns:
            bool: success
        """

        start =  from_date
        end =  datetime(from_date.year, from_date.month, from_date.day, 23, 59, 59, 0)
        while (start - timedelta(days=1)) < datetime.now():
            data = CoinBaseApi.getDataCoinbase(f"{coin}/{currency}", start, end)
            if len(data) == 0:
                print(f"No entry for {start}")
                start = start  + timedelta(days=1)
                end = end  + timedelta(days=1)
                continue
            if len(data) > 1:
                print(f"There was more than one entry for {start}")
                start = start  + timedelta(days=1)
                end = end  + timedelta(days=1)
                continue
            if not ExchangeRateData.addDayToDb(coin, currency, data[0], path=path):
                return False
            start = start  + timedelta(days=1)
            end = end  + timedelta(days=1)
        return True

    def filter_exchange_items(exchange_items: 'list[ExchangeRateItem]', from_date: datetime, to_date: datetime):
        """filter by date

        Args:
            exchange_items (list[ExchangeRateItem]): all items
            from_date (datetime): min time
            to_date (datetime): max time

        Returns:
            list[ExchangeRateItem]: entries that fit filter criteria
        """
        start = datetime(from_date.year, from_date.month, from_date.day)
        end = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59)
        return_list = []
        for item in exchange_items:
            if item.date >= start and item.date <= end:
                return_list.append(item)
        return return_list


    def get_exchange_items(filename : str):
        """read exchange days into list of objects

        Args:
            filename (string): name of db, can also be path

        Returns:
            list[ExchangeRateItem]: [each entry is a exchange rate day crypto/currency]
        """
        if not os.path.isfile(filename):
            return []
        file1 = open(filename, 'r')
        Lines = file1.readlines()
        first_line = True
        exchange_items = []
        last_day = datetime(1, 1, 1, 1, 1, 1, 1)
        # Strips the newline character
        for line in Lines:
            if first_line:
                first_line = False
                continue
            values = line.replace("\n", "").split(";")
            exchange_item = ExchangeRateItem()
            exchange_item.unix = int(values[0])
            exchange_item.low = float(values[1])
            exchange_item.high = float(values[2])
            exchange_item.open = float(values[3])
            exchange_item.close = float(values[4])
            exchange_item.volume = float(values[5])
            if values[7] != "None":
                exchange_item.twenty_week_average = float(values[7])
            #filter out bad data
            if exchange_item.open  == 0. or exchange_item.close  == 0. or exchange_item.high  == 0. or exchange_item.low  == 0.:
                continue
            exchange_item.date = datetime.strptime(values[6], '%Y-%m-%d %H:%M:%S')
            #only one item per day
            if last_day.year ==  exchange_item.date.year and last_day.month ==  exchange_item.date.month and last_day.day ==  exchange_item.date.day:
                continue
            exchange_items.append(exchange_item)
            last_day = exchange_item.date
        return exchange_items

    def GetExchangeRateItemOfDate(exchange_items: 'list[ExchangeRateItem]', now_date: datetime):
        for item in exchange_items:
            if now_date.year == item.date.year and now_date.month == item.date.month and now_date.day == item.date.day:
                return item
        return None

    def GetTwentyWeekAverage(exchange_items: 'list[ExchangeRateItem]', now_date: datetime):
        """calculate twenty week average for given item

        Args:
            exchange_items (list[ExchangeRateItem]): list of exchange items, preferably containing the last twenty weeks or most of it
            now_date (datetime): date of item we want to calculate the 20 week average for

        Returns:
            float: twenty week average
        """
        return_list : 'list[ExchangeRateItem]' = []
        start_date = now_date + timedelta(days=-140)
        while start_date <= now_date:
            item_day = ExchangeRateData.GetExchangeRateItemOfDate(exchange_items, start_date)
            if not item_day:
                start_date = start_date + timedelta(days=1)
                continue
            return_list.append(item_day.open)
            start_date = start_date + timedelta(days=1)
        if len(return_list) < 90:
            return None
        one = sum(return_list)
        two = len(return_list)
        return one / two

if __name__ == "__main__":
    coin = "BTC"
    currency = "EUR"
    start =  datetime(2013, 8, 1, 0, 0, 0, 0)

    ExchangeRateData.fillDb(coin, currency, start, path="./db/")

    # items = ExchangeRateData.get_exchange_items("./db/LTC_EUR_echange_db.csv")

    # for item in items:
    #     twenty_week_average = ExchangeRateData.GetTwentyWeekAverage(items, item.date)
    #     item.twenty_week_average = twenty_week_average
    #     ExchangeRateData.addDayToDb(coin, currency, item, path="./db/")




from datetime import datetime, time, timedelta

from ObjectsGai import ExchangeRateItem

class BitcoinDeApi:
    def GetFileName(coin : str, from_date : str, to_date : str):
        """Get filename of csv with exchange items

        Args:
            coin (str): coinname...ETH, BTC
            from_date (str): from date of file
            to_date (str): to date of file

        Returns:
            str: filename
        """
        return f'{coin}_usd_{from_date}_{to_date}.csv'

    def get_exchange_items(coin : str, from_date : str, to_date : str):
        """read exchange days into list of objects

        Args:
            coin (string): for now eth or btc
            from (string): from date for csv name file
            to (string): to date for csv name file

        Returns:
            list[ExchangeRateItem]: [each entry is a currency day crypto/usd]
        """
        filename = BitcoinDeApi.GetFileName(coin,from_date, to_date)
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

    def GetDateOfLastEntry(filename: str):
        """Get date of last entry of csv with exchange items

        Args:
            filename (str): filename of of csv with exchange items

        Returns:
            datetime: date of last entry
        """
        file1 = open(filename, 'r')
        lines = file1.readlines()
        last_line = lines[0]
        values = last_line.replace("\n", "").split(";")
        return datetime.strptime(values[6], '%Y-%m-%d %H:%M:%S')


    def GetExchangeRateItemOfDate(exchange_items: 'list[ExchangeRateItem]', now_date: datetime):
        for item in exchange_items:
            if now_date.year == item.date.year and now_date.month == item.date.month and now_date.day == item.date.day:
                return item
        return None

    def GetTwentyWeekAverage(exchange_items: 'list[ExchangeRateItem]', now_date: datetime):
        return_list : 'list[ExchangeRateItem]' = []
        start_date = now_date + timedelta(days=-140)
        while start_date <= now_date:
            item_day = BitcoinDeApi.GetExchangeRateItemOfDate(exchange_items, start_date)
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



        

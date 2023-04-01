from datetime import datetime, timedelta
import sys
import os
import time

#add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from api.CoinBaseRepo import CoinBaseRepo
from api.FearGreedRepo import FearGreedRepo 




#if main
if __name__ == "__main__":
    start = datetime(2023, 1, 1)
    end = datetime(2023, 2, 1)
    while start < datetime.now():
        CoinBaseRepo.fetch_daily_data('BTC/EUR', start, end)
        start = start + timedelta(days=30)
        end = end + timedelta(days=30)
        #sleep 1 second
        time.sleep(1)
        print(start)

    dictlist = CoinBaseRepo.read_csv_to_dict('api/BTC_EUR.csv')
    exchange_items = CoinBaseRepo.get_exchange_rate_items(datetime(2021, 8, 1), datetime(2021, 8, 10), dictlist)
    print(f"Number of exchange items: {len(exchange_items)}")

    FearGreedRepo.get_data(10000, 'api/fear_greed_data.csv')
    dictlist = FearGreedRepo.read_csv_file(datetime(2021, 8, 1), datetime(2021, 8, 10), 'api/fear_greed_data.csv')
    print(f"Number of fear greed items: {len(dictlist)}")
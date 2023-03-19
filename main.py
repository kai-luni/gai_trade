from datetime import datetime
import sys
import os

#add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from api.CoinBaseRepo import CoinBaseRepo
from api.FearGreedRepo import FearGreedRepo 




#if main
if __name__ == "__main__":
    dictlist = CoinBaseRepo.read_csv_to_dict('api/BTC_EUR.csv')
    exchange_items = CoinBaseRepo.get_exchange_rate_items(datetime(2021, 8, 1), datetime(2021, 8, 10), dictlist)
    print(f"Number of exchange items: {len(exchange_items)}")

    FearGreedRepo.get_data(10000, 'api/fear_greed_data.csv')
    dictlist = FearGreedRepo.read_csv_file(datetime(2021, 8, 1), datetime(2021, 8, 10), 'api/fear_greed_data.csv')
    print(f"Number of fear greed items: {len(dictlist)}")
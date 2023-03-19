from datetime import datetime
import sys
import os

#add the parent directory to the path
test = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from api.CoinBaseRepo import CoinBaseRepo as cbRepo




#if main
if __name__ == "__main__":
    dictlist = cbRepo.read_csv_to_dict('api/BTC_EUR.csv')
    exchange_items = cbRepo.get_exchange_rate_items(datetime(2021, 8, 1), datetime(2021, 8, 10), dictlist)
    print(f"Number of exchange items: {len(exchange_items)}")
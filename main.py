from datetime import datetime, timedelta
import sys
import os
import time

#add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from api.CoinBaseRepo import CoinBaseRepo
from api.FearGreedRepo import FearGreedRepo 

from algos.PearsonCorrelationCoefficient import PCC




#if main
if __name__ == "__main__":
    # start = datetime(2023, 1, 1)
    # end = datetime(2023, 2, 1)
    # while start < datetime.now():
    #     CoinBaseRepo.fetch_daily_data('BTC/EUR', start, end)
    #     start = start + timedelta(days=30)
    #     end = end + timedelta(days=30)
    #     #sleep 1 second
    #     time.sleep(1)
    #     print(start)

    start_filter = datetime(2018, 2, 1)
    end_filter = datetime(2023, 3, 20)
    dictlist = CoinBaseRepo.read_csv_to_dict('api/BTC_EUR.csv')
    exchange_items = CoinBaseRepo.get_exchange_rate_items(start_filter, end_filter, dictlist)
    print(f"Number of exchange items: {len(exchange_items)}")

    FearGreedRepo.get_data(10000, 'api/fear_greed_data.csv')
    greed_items = FearGreedRepo.read_csv_file(start_filter, end_filter, 'api/fear_greed_data.csv')
    print(f"Number of fear greed items: {len(dictlist)}")

    dates = [item.date for item in exchange_items]
    opens = [item.open for item in exchange_items]
    greed_indexes = [item["index"] for item in greed_items]

    PCC.calculate_pcc(dates, opens, greed_indexes)
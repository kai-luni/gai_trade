from datetime import datetime, timedelta

from algo.algo_sell_fear import AlgoSellFear
from db.fear_greed_data_sql import FearGreedData
from db.exchange_rate_data_sql import ExchangeRateData
from dto.ObjectsGai import TradeSellParams

class GaiTradeBot:
    currency = "EUR"
    coin = "LTC"

    print("Update Fear Greed Database")
    path_db = FearGreedData.get_db_full_path("db/")
    fear_greed_data = FearGreedData(path_db)
    fear_greed_data.fill_db(50)

    print("Update Currency Database")
    path_db_ex = ExchangeRateData.getDbFullPath(coin, currency, "db/")
    ExchangeRateData.fillDb(coin, currency, datetime(2022,1,1) + timedelta(days=-10), path_db_ex)

    print("Trade")
    params = TradeSellParams()
    params.start = datetime(2015, 10, 1, 0, 0, 0, 0)
    params.end = datetime.today()
    params.days_look_back = 105
    params.buy_at_gfi = 8
    params.percent_change_sell = 4.2
    params.coin = coin
    params.currency = "EUR"
    AlgoSellFear.trade(params, debug = True)




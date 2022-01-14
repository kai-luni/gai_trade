from datetime import datetime, timedelta

from data.fear_greed_data_reader import FearGreedDataReader
from db.exchange_rate_data_sql import ExchangeRateData
from dto.ObjectsGai import ExchangeRateItem, TradeSellParams

def buy(money : float, coins : float, currency : float):
    if money <= 0:
        return money, coins
    return 0, coins + (money / currency)

def sell(money : float, coins : float, currency : float):
    if coins <= 0:
        return money, coins
    return money + (coins * currency), 0

def trade(params : TradeSellParams, debug=False):
    full_path = ExchangeRateData.getDbFullPath(params.coin, "EUR", "./db/")
    all_entries : 'list[ExchangeRateItem]' = ExchangeRateData.get_all_items(full_path)
    filtered_entries : 'list[ExchangeRateItem]' = ExchangeRateData.filter_exchange_items(params.start, params.end, all_entries)
    fear_greed_entries = FearGreedDataReader()

    coins = 1
    money = 0

    # for entry in filtered_entries[params.days_look_back, len(filtered_entries)]:
    #     print(f"{entry.unix} {entry.date}")
    for i in range(params.days_look_back, len(filtered_entries)):
        today = filtered_entries[i].date
        currency_entry : ExchangeRateItem = filtered_entries[i]
        greed_fear_entry = fear_greed_entries.get_entry_for_day(filtered_entries[i].date.day, filtered_entries[i].date.month, filtered_entries[i].date.year)
        if greed_fear_entry == None:
            raise Exception(f"Missing Greed Fear Entry for {today}")

        ### sell
        today_exchange_rate = currency_entry.close
        first_entry_look_back_exchange = filtered_entries[i-params.days_look_back].close
        factor_change_value = today_exchange_rate / first_entry_look_back_exchange
        if coins > 0 and factor_change_value > params.percent_change_sell and greed_fear_entry[1] >= 90:
            money, coins = sell(money, coins, currency_entry.close)
            if debug:
                print(f"sell on {today}, we now have {money} EUR")

        ### buy
        if greed_fear_entry[1] < params.buy_at_gfi and money > 0:
            money, coins = buy(money, coins, currency_entry.close)
            if debug:
                print(f"buy on {filtered_entries[i].date}, we now have {coins} coins.")
        
    final_coins = coins + (money/filtered_entries[-1].close)
    if debug:
        print(f"Final Coins: {final_coins}, {params.days_look_back} {params.buy_at_gfi} {params.percent_change_sell}")
    return final_coins



if __name__ == "__main__":
    params = TradeSellParams()
    params.start = datetime(2018, 2, 1, 0, 0, 0, 0)
    params.end = datetime(2022, 1, 12, 0, 0, 0, 0)
    params.days_look_back = 175
    params.buy_at_gfi = 30
    params.percent_change_sell = 3.8
    params.coin = "LTC"
    trade(params, debug = True)
    # for i in range(50, 200, 5):
    #     for k in range(101, 270, 5):
    #         params.days_look_back = i
    #         params.percent_change_sell = float(k) / 100.

    #         params.coin = "BTC"
    #         coins_btc = trade(params)
    #         params.coin = "ETH"
    #         coins_eth = trade(params)
    #         params.coin = "LTC"
    #         coins_ltc = trade(params)
    #         if ((coins_btc + coins_eth + coins_ltc)/3) > 1.5 and coins_btc > 1. and coins_eth > 1. and coins_ltc > 1.:
    #             print(f"{coins_btc} {coins_eth} {coins_ltc} lookback: {i}, percent_change_sell: {params.percent_change_sell}")
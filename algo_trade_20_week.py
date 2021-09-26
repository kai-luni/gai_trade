from datetime import datetime, timedelta
import os

# import numpy as np
# import matplotlib.pyplot as plt

from test_api import BitcoinDeApi
from ObjectsGai import TradeBuy, TradeSell, TradeSellParams

def SellSomeCoins(owned_coins, currency, date, sell_part):
    coins_to_sell = (owned_coins * sell_part)

    trade = TradeSell()
    trade.currency = "EUR"
    trade.amount = coins_to_sell
    trade.sell_date_utc = date
    trade.sell_at_euro = currency
    trade.sell_text = f"sell {coins_to_sell} for {coins_to_sell*currency}"
    
    return trade

def trade(all_exchange_items, params: TradeSellParams, always_spend=False,  calculate_per_year=False):
    original_start = params.start
    trade_list_buy = []
    trade_list_sell = []
    money = 0.
    coins = 10.
    while params.start <= params.end:
        exchange_items = BitcoinDeApi.FilterExchangeItems(params.start, params.start + timedelta(days=1), all_exchange_items)
        if not exchange_items or len(exchange_items) == 0:
            params.start = params.start + timedelta(days=1)
            continue

        # >>> sell
        #trade time
        last_trade_date = datetime(1, 1, 1, 1, 1, 1, 1) if len(trade_list_sell) == 0 else trade_list_sell[-1].sell_date_utc
        sell_time_ok = last_trade_date + timedelta(days=params.days_between_sales) < params.start
        if (coins > .1 and sell_time_ok) or (always_spend and coins > .1) or money > 100:
            #20 week average
            twenty_week_average = BitcoinDeApi.GetTwentyWeekAverage(all_exchange_items, params.start)
            above_twenty_week_average = twenty_week_average and exchange_items[0].open > (twenty_week_average * params.above_twenty)
            below_twenty_week_average = twenty_week_average and exchange_items[0].open < (twenty_week_average * params.below_twenty)

            #sell
            if above_twenty_week_average and coins > .1:
                trade = SellSomeCoins(coins, exchange_items[0].open, exchange_items[0].date, params.sell_part)

                trade_list_sell.append(trade)
                money += trade.amount*trade.sell_at_euro
                coins -= trade.amount

            #buy
            elif params.start.day == 1 and money > 100:
                money_to_spend = (money * params.spend_part)
                money -= money_to_spend
                coins += money_to_spend / exchange_items[0].open

                trade = TradeBuy()
                trade.currency = "EUR"
                trade.amount = money_to_spend / exchange_items[0].open
                trade.buy_date_utc = exchange_items[0].date
                trade.buy_at_euro = exchange_items[0].open
                trade.buy_text = f"buy {trade.amount} for {trade.amount*exchange_items[0].open}"
                trade_list_buy.append(trade)

        new_start = params.start + timedelta(days=1)
        if new_start.month != params.start.month:
            money += 0 
        params.start = new_start

    if calculate_per_year:
        time_calc = original_start
        while time_calc.year <= params.final_end.year:
            sell_year = 0
            for sell_item in trade_list_sell:
                if sell_item.sell_date_utc.year == time_calc.year:
                    sell_year += sell_item.amount*sell_item.sell_at_euro
            buy_year = 0
            for buy_item in trade_list_buy:
                if buy_item.buy_date_utc.year == time_calc.year:
                    buy_year += buy_item.amount*buy_item.buy_at_euro
            print(f"In year {time_calc.year} sold {sell_year} bought {buy_year} gain {sell_year-buy_year}")
            time_calc = time_calc + timedelta(days=365)
    return coins + (money / exchange_items[0].open)


best_win_eth = 0
best_win_btc = 0
best_win_three = 0
all_exchange_items_btc = BitcoinDeApi.get_exchange_items("btc", "20151009", "20210701")
all_exchange_items_eth = BitcoinDeApi.get_exchange_items("eth", "20160510", "20210706")

filename = datetime.now().strftime('%Y-%m-%d_%H%M%S') + ".txt"
write_or_append = "a" if os.path.isfile(filename) else "w"
outF = open(filename, write_or_append)

for spend_part in [1., 0.92, .85, .5, .3]:
    for days_between_sales in [5, 30, 80]:
        for sell_part in[1., .9, .85, .5, .45, .3]:
            for above_twenty in [1.03, 1.05, 1.07, 1.1, 1.2]:
                for below_twenty in [.99, .96, .92, .9, .8]:
                    params_eth = TradeSellParams(above_twenty, below_twenty, None, days_between_sales, None, None, sell_part, spend_part, datetime(2016, 5, 10, 16, 0, 0, 0), datetime(2021, 7, 6, 0, 0, 0, 0))
                    total_money_eth = trade(all_exchange_items_eth ,params_eth)
                    params_btc = TradeSellParams(above_twenty, below_twenty, None, days_between_sales, None, None, sell_part, spend_part, datetime(2016, 5, 10, 16, 0, 0, 0), datetime(2021, 7, 1, 0, 0, 0, 0))
                    total_money_btc = trade(all_exchange_items_btc ,params_btc)
                    #total_money_three = trade(all_exchange_items, days_look_back, spend_part, percent_change_sell, days_between_sales, sell_part, buy_times_per_month, datetime(2020, 4, 19, 16, 0, 0, 0), datetime(2021, 7, 1, 0, 0, 0, 0))
                    string_add = ""
                    #if (total_money_one  > best_win_eth and total_money_two > best_win_btc) or (total_money_one  > best_win_eth and total_money_three > best_win_three) or (total_money_two  > best_win_btc and total_money_three > best_win_three):
                    if (total_money_eth  > best_win_eth and total_money_btc > best_win_btc or total_money_eth  > best_win_eth and total_money_btc > best_win_btc*0.8 or total_money_eth  > best_win_eth*0.7 and total_money_btc > best_win_btc):
                        best_win_eth = total_money_eth
                        best_win_btc = total_money_btc
                        output_string = f">>> coins eth {total_money_eth} and btc {total_money_btc} above_twenty {above_twenty} days_between_sales {days_between_sales} sell_part {sell_part} spend_part {spend_part} below_twenty {below_twenty}"
                        print(output_string)
                        outF.write(output_string + "\n")
                            
outF.close()
#>>> coins eth 34.56816466754604 and btc 17.640990538790447 days_look_back 56 spend_part 1.0 percent_change_sell 1.05 days_between_sales 30 buy_times_per_month 3 sell_part 1.0
# params_loc = TradeSellParams(3, 30, 56, 1.05, 1., 1., datetime(2016, 5, 10, 16, 0, 0, 0), datetime(2021, 7, 6, 0, 0, 0, 0))
# total_money_one = trade(all_exchange_items_eth ,params_loc, sell_above_20_average=True)
# #total_money_one = trade(all_exchange_items, 7, 1., 1.3, 5, 1., 3, datetime(2017, 12, 1, 16, 0, 0, 0), datetime(2021, 5, 26, 0, 0, 0, 0))
#print(total_money_one)



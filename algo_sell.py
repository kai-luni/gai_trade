from datetime import datetime, timedelta

# import numpy as np
# import matplotlib.pyplot as plt

from ExchangeRateData import ExchangeRateData
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

def trade(all_exchange_items, params: TradeSellParams, always_spend=False, sell_above_20_average=False, buy_below_20_average=False, calculate_per_year=False):
    original_start = params.start
    end = params.start
    trade_list_buy = []
    trade_list_sell = []
    money = 0.
    coins = 10.
    while end < params.end:
        
        end = params.start + timedelta(days=params.days_look_back)
        exchange_items = BitcoinDeApi.FilterExchangeItems(params.start, end, all_exchange_items)
        total_money = money + (coins*exchange_items[-1].open)

        value_start = exchange_items[0].open
        value_end = exchange_items[-1].open
        factor_change_value = value_end / value_start

        # >>> buy
        buying = False
        # if end.day in [1, 12, 24]:
        #     buying = True
        # twenty_week_average = exchange_items[-1].twenty_week_average
        # below_twenty_week_average = twenty_week_average and exchange_items[-1].open < twenty_week_average

        # if params.below_twenty_buy == True:
        #     if exchange_items[-1].twenty_week_average != None:
        #         twenty_week_average = exchange_items[-1].twenty_week_average
        #         buying = exchange_items[-1].open < twenty_week_average
        if end.day in params.days_buy:
            buying = True



        if buying == True and money > 100:
            money_to_spend = (money * params.spend_part)
            money -= money_to_spend
            coins += money_to_spend / exchange_items[-1].open

            trade = TradeBuy()
            trade.currency = "EUR"
            trade.amount = money_to_spend / exchange_items[-1].open
            trade.buy_date_utc = exchange_items[-1].date
            trade.buy_at_euro = exchange_items[-1].open
            trade.buy_text = f"buy {trade.amount} for {trade.amount*exchange_items[-1].open}"
            trade_list_buy.append(trade)


        # >>> sell
        #trade time
        last_trade_date = datetime(1, 1, 1, 1, 1, 1, 1) if len(trade_list_sell) == 0 else trade_list_sell[-1].sell_date_utc
        sell_time_ok = last_trade_date + timedelta(days=params.days_between_sales) < end

        if (factor_change_value > params.percent_change_sell and coins > .1 and sell_time_ok) or (always_spend and coins > .1):
            #20 week average, next line is quite ugly. should be changed.
            above_twenty_week_average = False if sell_above_20_average == True else True
            if sell_above_20_average == True and exchange_items[-1].twenty_week_average != None:
                twenty_week_average = exchange_items[-1].twenty_week_average
                above_twenty_week_average = twenty_week_average and (exchange_items[-1].open > (twenty_week_average * params.above_twenty))

            if above_twenty_week_average:
                #print(f"exchange_items[-1].open {exchange_items[-1].open} twenty_week_average * params.above_twenty {twenty_week_average * params.above_twenty} exchange_items[-1].date {exchange_items[-1].date}")
                trade = SellSomeCoins(coins, exchange_items[-1].open, exchange_items[-1].date, params.sell_part)
                trade_list_sell.append(trade)
                money += trade.amount*trade.sell_at_euro
                coins -= trade.amount

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
    return coins + (money / exchange_items[-1].open)


best_win_eth = 0
best_win_btc = 0
best_win_three = 0
all_exchange_items_btc = BitcoinDeApi.get_exchange_items("btc", "20151009", "20210701")
all_exchange_items_eth = BitcoinDeApi.get_exchange_items("eth", "20160510", "20210706")

all_exchange_items_ltc = ExchangeRateData.get_exchange_items("./db/LTC_EUR_echange_db.csv")

# filename = datetime.now().strftime('%Y-%m-%d_%H%M%S') + ".txt"
# write_or_append = "a" if os.path.isfile(filename) else "w"
# outF = open(filename, write_or_append)
# for days_look_back in [2, 3, 4, 7, 9, 20, 54, 56, 58, 60, 62, 100]:
#     print(f">>>>>> At day {days_look_back}")
#     for percent_change_sell in [1., 1.03, 1.05, 1.15, 1.19, 1.25, 2]:
#         for days_between_sales in [1, 5, 10, 30, 80]:
#             for sell_part in[1., .9, .85, .5, .45, .3]:
#                 for above_twenty in [1.02, 1.04, 1.07, 1.1]:
#                     for twenty_av_above in [True, False]:
#                         for sell_below_20 in [True]:
#                             for spend_part in [1., .98, .95, .8, .5]:
#                                 params_eth = TradeSellParams(above_twenty, sell_below_20, 3, days_between_sales, days_look_back, percent_change_sell, sell_part, spend_part, datetime(2016, 5, 10, 16, 0, 0, 0), datetime(2021, 7, 6, 0, 0, 0, 0))
#                                 total_money_eth = trade(all_exchange_items_eth ,params_eth, sell_above_20_average=twenty_av_above)
#                                 params_btc = TradeSellParams(above_twenty, sell_below_20, 3, days_between_sales, days_look_back, percent_change_sell, sell_part, spend_part, datetime(2016, 5, 10, 16, 0, 0, 0), datetime(2021, 7, 1, 0, 0, 0, 0))
#                                 total_money_btc = trade(all_exchange_items_btc ,params_btc, sell_above_20_average=twenty_av_above)
#                                 string_add = ""
#                                 #if total_money_eth > 10. and total_money_btc > 10. and (total_money_eth  > best_win_eth and total_money_btc > best_win_btc or total_money_eth  > best_win_eth and total_money_btc > best_win_btc*0.8 or total_money_eth  > best_win_eth*0.7 and total_money_btc > best_win_btc):
#                                 if (total_money_eth  > best_win_eth and total_money_btc > best_win_btc or total_money_eth  > best_win_eth and total_money_btc > best_win_btc*0.9 or total_money_eth  > best_win_eth*0.9 and total_money_btc > best_win_btc):
#                                     best_win_eth = total_money_eth
#                                     best_win_btc = total_money_btc
#                                     output_string = f">>> coins eth {total_money_eth} and btc {total_money_btc} above_twenty {above_twenty} buy_times_per_month {3} days_between_sales {days_between_sales} days_look_back {days_look_back} sell_below_20 {sell_below_20} spend_part {spend_part} percent_change_sell {percent_change_sell} sell_part {sell_part} twenty_av_above {twenty_av_above}"
#                                     print(output_string)
#                                     outF.write(output_string + "\n")
                            
# outF.close()



#>>> coins eth 32.15023144328083 and btc 19.964081656858333 above_twenty 1.02 buy_times_per_month 3 days_between_sales 30 days_look_back 56 sell_below_20 False spend_part 1.0 percent_change_sell 1.05 sell_part 1.0 twenty_av_above True
# for i in range(30):
#     for j in range(30):
#         for k in range(30):
#             days = [i, j, k]
#             params_loc_eth = TradeSellParams(1.02, [1,12,24], None, 30, 56, 1.05, 1., 1., datetime(2016, 5, 10, 16, 0, 0, 0), datetime(2021, 7, 6, 0, 0, 0, 0))
#             params_loc_btc = TradeSellParams(1.02, [1,12,24], None, 30, 56, 1.05, 1., 1., datetime(2015, 10, 10, 16, 0, 0, 0), datetime(2021, 7, 1, 0, 0, 0, 0))
#             total_money_eth = trade(all_exchange_items_eth ,params_loc_eth, sell_above_20_average=True)
#             total_money_btc = trade(all_exchange_items_btc ,params_loc_btc, sell_above_20_average=True)
#             if total_money_btc > 22 and total_money_eth > 32:
#                 print(f"above_twenty {days} total_money_eth {total_money_eth} total_money_btc {total_money_btc}")
# total_money_one = trade(all_exchange_items_eth, 7, 1., 1.3, 5, 1., 3, datetime(2017, 12, 1, 16, 0, 0, 0), datetime(2021, 5, 26, 0, 0, 0, 0))
# print(total_money_one)

params_loc_eth = TradeSellParams(1.02, [1,12,24], None, 30, 56, 1.05, 1., 1., datetime(2016, 5, 10, 16, 0, 0, 0), datetime(2021, 7, 6, 0, 0, 0, 0))
params_loc_btc = TradeSellParams(1.02, [1,12,24], None, 30, 56, 1.05, 1., 1., datetime(2015, 10, 10, 16, 0, 0, 0), datetime(2021, 7, 1, 0, 0, 0, 0))
params_loc_ltc = TradeSellParams(1.02, [1,12,24], None, 30, 56, 1.05, 1., 1., datetime(2017, 5, 30, 16, 0, 0, 0), datetime(2021, 8, 5, 0, 0, 0, 0))
total_money_eth = trade(all_exchange_items_eth ,params_loc_eth, sell_above_20_average=True)
total_money_btc = trade(all_exchange_items_btc ,params_loc_btc, sell_above_20_average=True)
total_money_ltc = trade(all_exchange_items_ltc ,params_loc_ltc, sell_above_20_average=True)
print(f"eth: {total_money_eth}")
print(f"btc: {total_money_btc}")
print(f"ltc: {total_money_ltc}")


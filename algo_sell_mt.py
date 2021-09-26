import copy
from datetime import datetime, timedelta
from itertools import product
import multiprocessing as mp
import time
import os

# import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import linregress

from ExchangeRateData import ExchangeRateData
from test_api import BitcoinDeApi
from ObjectsGai import TradeBuy, TradeSell, TradeSellParams, ExchangeRateItem

def CreateTradeSell(owned_coins : float, currency : float, date : datetime, sell_part : float):
    """Create a Trade Items with amount that can be sold calculated

    Args:
        owned_coins (float): amount of coins owned
        currency (float): current currency
        date (datetime): current date
        sell_part (float): percentage to sell (0.5 would be 50 percent)

    Returns:
        TradeSell: Trade Object
    """
    coins_to_sell = (owned_coins * sell_part)

    trade = TradeSell()
    trade.currency = "EUR"
    trade.amount = coins_to_sell
    trade.sell_date_utc = date
    trade.sell_at_euro = currency
    trade.sell_text = f"sell {coins_to_sell} for {coins_to_sell*currency}"
    
    return trade

def buy(current_day : datetime, params: TradeSellParams, current_exchange_item: ExchangeRateItem, money : float, coins : float):
    """execute buy if conditions are right

    Args:
        current_day (datetime): current day
        params (TradeSellParams): parameters for buying and selling
        current_exchange_item (ExchangeRateItem): exchange rate item of day
        money (float): current money owned
        coins (float): current coins owned

    Returns:
        TradeBuy, float, float: trade made, money coins
    """
    buying = False
    if current_day.day in params.days_buy:
        buying = True

    if buying == True and money > 100:
        money_to_spend = (money * params.spend_part)
        money -= money_to_spend
        coins += money_to_spend / current_exchange_item.open

        trade = TradeBuy()
        trade.currency = "EUR"
        trade.amount = money_to_spend / current_exchange_item.open
        trade.buy_date_utc = current_exchange_item.date
        trade.buy_at_euro = current_exchange_item.open
        trade.buy_text = f"buy {trade.amount} for {trade.amount*current_exchange_item.open}"
        return trade, money, coins
    return None, money, coins

def sell(current_day : datetime, trade_list_sell : 'list[TradeSell]', params: TradeSellParams, current_exchange_item: ExchangeRateItem, money : float, coins : float, factor_change_value : float, always_spend=False):
    """[summary]

    Args:
        current_day (datetime): current day
        trade_list_sell (list[TradeSell]): all past sell trades
        params (TradeSellParams): parameters for buying and selling
        current_exchange_item (ExchangeRateItem): exchange rate item of day
        money (float): current money owned
        coins (float): current coins owned
        factor_change_value (float): basically the slope of past days exchange rate
        always_spend (bool, optional): If True always buy when have money. Defaults to False.

    Returns:
        TradeBuy, float, float: trade made, money coins
    """
    #trade time
    last_trade_date = datetime(1, 1, 1, 1, 1, 1, 1) if len(trade_list_sell) == 0 else trade_list_sell[-1].sell_date_utc
    sell_time_ok = last_trade_date + timedelta(days=params.days_between_sales) < current_day

    if (factor_change_value > params.percent_change_sell and coins > .1 and sell_time_ok) or (always_spend and coins > .1):
        #20 week average, next line is quite ugly. should be changed.
        above_twenty_week_average = False if params.sell_above_20_average == True else True
        if params.sell_above_20_average == True and current_exchange_item.twenty_week_average != None:
            twenty_week_average = current_exchange_item.twenty_week_average
            above_twenty_week_average = twenty_week_average and (current_exchange_item.open > (twenty_week_average * params.above_twenty))

        if above_twenty_week_average:
            #print(f"current_exchange_item.open {current_exchange_item.open} twenty_week_average * params.above_twenty {twenty_week_average * params.above_twenty} current_exchange_item.date {current_exchange_item.date}")
            trade = CreateTradeSell(coins, current_exchange_item.open, current_exchange_item.date, params.sell_part)
            money += trade.amount*trade.sell_at_euro
            coins -= trade.amount
            return trade, money, coins

    return None, money, coins

def trade(all_exchange_items: 'list[ExchangeRateItem]', params: TradeSellParams, always_spend=False, calculate_per_year=False):
    original_start = params.start
    end = params.start
    trade_list_buy = []
    trade_list_sell = []
    money = 0.
    coins = 10.
    while end < params.end:
        
        end = params.start + timedelta(days=params.days_look_back)
        exchange_items = BitcoinDeApi.FilterExchangeItems(params.start, end, all_exchange_items)

        value_start = exchange_items[0].open
        value_end = exchange_items[-1].open
        factor_change_value = value_end / value_start

        # >>> buy
        trade_buy, money, coins = buy(end, params, exchange_items[-1], money, coins)
        if trade_buy != None:
            trade_list_buy.append(trade_buy)


        # >>> sell
        x_values = [x for x in range(len(exchange_items))]
        y_values = [y.open for y in exchange_items]
        slope = linregress(x_values, y_values).slope * len(exchange_items)
        factor_change_value = (slope/y_values[0])
        trade_sell, money, coins = sell(end, trade_list_sell, params, exchange_items[-1], money, coins, factor_change_value)
        if trade_sell != None:
            trade_list_sell.append(trade_sell)


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



def calculateMultiThread(exchange_items_eth, exchange_items_btc, exchange_items_ltc, param_list):
    # with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    #     futures = [executor.submit(trade, exchange_items, param) for param in param_list]
    #     return [f.result() for f in futures]
    pool = mp.Pool(mp.cpu_count())
    results = []
    for param in param_list:
        if param.coin_name == "ETH":
            results.append(pool.apply_async(trade, (exchange_items_eth, param)))
        if param.coin_name == "BTC":
            results.append(pool.apply_async(trade, (exchange_items_btc, param)))
        if param.coin_name == "LTC":
            results.append(pool.apply_async(trade, (exchange_items_ltc, param)))
    pool.close()
    pool.join()
    return [result.get() for result in results]


if __name__ == '__main__':
    trade_sell_params = []

    best_win_eth = 0
    best_win_btc = 0
    best_win_ltc = 0
    all_exchange_items_btc = BitcoinDeApi.get_exchange_items("btc", "20151009", "20210701")
    all_exchange_items_eth = BitcoinDeApi.get_exchange_items("eth", "20160510", "20210706")
    all_exchange_items_ltc = ExchangeRateData.get_exchange_items("./db/LTC_EUR_echange_db.csv")

    # >>>>>>>>>>>>> time measure
    # start_time = time.time()

    filename = datetime.now().strftime('%Y-%m-%d_%H%M%S') + ".txt"
    write_or_append = "a" if os.path.isfile(filename) else "w"
    outF = open(filename, write_or_append)
    for days_look_back in [20, 30, 56, 58, 60, 62, 90, 100, 120]:
        trade_params = TradeSellParams()
        trade_params.days_look_back = days_look_back
        print(f">>>>>> At day {days_look_back}")
        for percent_change_sell in [1., 1.03, 1.05, 1.15, 1.19, 1.25, 2]:
            trade_params.percent_change_sell = percent_change_sell
            for days_between_sales in [1, 5, 10, 30, 80, 100]:
                trade_params.days_between_sales = days_between_sales
                for sell_part in[1., .9, .85, .5, .45, .3]:
                    trade_params.sell_part = sell_part
                    for above_twenty in [1., 1.02, 1.04, 1.07, 1.1]:
                        trade_params.above_twenty = above_twenty
                        for sell_above_20_average in [True, False]:
                            trade_params.sell_above_20_average = sell_above_20_average
                            for spend_part in [1., .98, .95, .8, .5, .2]:
                                trade_params.spend_part = spend_part

                                trade_params_eth = copy.deepcopy(trade_params)
                                trade_params_eth.coin_name = "ETH"
                                trade_params_eth.start = datetime(2016, 5, 10, 16, 0, 0, 0)
                                trade_params_eth.end = datetime(2021, 7, 6, 0, 0, 0, 0)
                                trade_sell_params.append(trade_params_eth)

                                trade_params_btc = copy.deepcopy(trade_params)
                                trade_params_btc.coin_name = "BTC"
                                trade_params_btc.start = datetime(2015, 10, 10, 16, 0, 0, 0)
                                trade_params_btc.end = datetime(2021, 7, 1, 0, 0, 0, 0)
                                trade_sell_params.append(trade_params_btc)

                                trade_params_ltc = copy.deepcopy(trade_params)
                                trade_params_ltc.coin_name = "LTC"
                                trade_params_ltc.start = datetime(2017, 5, 30, 16, 0, 0, 0)
                                trade_params_ltc.end = datetime(2021, 8, 5, 0, 0, 0, 0)
                                trade_sell_params.append(trade_params_ltc)
                                #trade_sell_params.append(TradeSellParams(above_twenty, [1, 12, 24], 3, days_between_sales, days_look_back, percent_change_sell, sell_above_20_average, sell_part, spend_part, datetime(2016, 5, 10, 16, 0, 0, 0), datetime(2021, 7, 1, 0, 0, 0, 0)))
                                if len(trade_sell_params) > 1000:
                                    total_btc_list = calculateMultiThread(all_exchange_items_eth, all_exchange_items_btc, all_exchange_items_ltc, trade_sell_params)    
                                    for i in range(len(total_btc_list)):
                                        if trade_sell_params[i].coin_name != "LTC":
                                            continue
                                        total_money_eth = total_btc_list[i-2]
                                        total_money_btc = total_btc_list[i-1]
                                        total_money_ltc = total_btc_list[i]
                                        #total_money_btc = trade(all_exchange_items_btc ,params_btc, sell_above_20_average=sell_above_20_average)
                                        if (total_money_eth > (best_win_eth*.8) and total_money_btc > (best_win_btc*.8) and total_money_ltc > (best_win_ltc*.8)):
                                            if total_money_eth > best_win_eth:
                                                best_win_eth = total_money_eth
                                            if total_money_btc > best_win_btc:
                                                best_win_btc = total_money_btc
                                            if total_money_ltc > best_win_ltc:
                                                best_win_ltc = total_money_ltc

                                            output_string_eth = f">>> coins eth {total_money_eth} above_twenty {trade_sell_params[i].above_twenty} buy_times_per_month {trade_sell_params[i].buy_times_per_month} days_between_sales {trade_sell_params[i].days_between_sales} days_look_back {trade_sell_params[i].days_look_back} spend_part {trade_sell_params[i].spend_part} percent_change_sell {trade_sell_params[i].percent_change_sell} sell_part {trade_sell_params[i].sell_part} twenty_av_above {trade_sell_params[i].sell_above_20_average}"
                                            output_string_btc = f">>> coins btc {total_money_btc} above_twenty {trade_sell_params[i].above_twenty} buy_times_per_month {trade_sell_params[i].buy_times_per_month} days_between_sales {trade_sell_params[i].days_between_sales} days_look_back {trade_sell_params[i].days_look_back} spend_part {trade_sell_params[i].spend_part} percent_change_sell {trade_sell_params[i].percent_change_sell} sell_part {trade_sell_params[i].sell_part} twenty_av_above {trade_sell_params[i].sell_above_20_average}"
                                            output_string_ltc = f">>> coins ltc {total_money_ltc} above_twenty {trade_sell_params[i].above_twenty} buy_times_per_month {trade_sell_params[i].buy_times_per_month} days_between_sales {trade_sell_params[i].days_between_sales} days_look_back {trade_sell_params[i].days_look_back} spend_part {trade_sell_params[i].spend_part} percent_change_sell {trade_sell_params[i].percent_change_sell} sell_part {trade_sell_params[i].sell_part} twenty_av_above {trade_sell_params[i].sell_above_20_average}"
                                            print("------------")
                                            print(output_string_eth)
                                            print(output_string_btc)
                                            print(output_string_ltc)
                                            print("------------")
                                            outF.write(output_string_eth + "\n")
                                            outF.write(output_string_btc + "\n")
                                            outF.write(output_string_ltc + "\n")
                                    trade_sell_params = []
                                    # >>>>>>>>>>>>> time measure
                                    #print("--- %s seconds ---" % (time.time() - start_time))
                                
    outF.close()


# params_loc_eth = TradeSellParams(above_twenty=1.02, days_buy=[1,12,24], buy_times_per_month=3, days_between_sales=30, days_look_back=56, percent_change_sell=1.05, sell_above_20_average=True, sell_part=1., spend_part=1., start=datetime(2016, 5, 10, 16, 0, 0, 0), end=datetime(2021, 7, 6, 0, 0, 0, 0))
# params_loc_btc = TradeSellParams(above_twenty=1.02, days_buy=[1,12,24], buy_times_per_month=3, days_between_sales=30, days_look_back=56, percent_change_sell=1.05, sell_above_20_average=True, sell_part=1., spend_part=1., start=datetime(2015, 10, 10, 16, 0, 0, 0), end=datetime(2021, 7, 1, 0, 0, 0, 0))
# params_loc_ltc = TradeSellParams(above_twenty=1.02, days_buy=[1,12,24], buy_times_per_month=3, days_between_sales=30, days_look_back=56, percent_change_sell=1.05, sell_above_20_average=True, sell_part=1., spend_part=1., start=datetime(2017, 5, 30, 16, 0, 0, 0), end=datetime(2021, 8, 5, 0, 0, 0, 0))
# total_money_eth = trade(all_exchange_items_eth ,params_loc_eth)
# total_money_btc = trade(all_exchange_items_btc ,params_loc_btc)
# total_money_ltc = trade(all_exchange_items_ltc ,params_loc_ltc)
# print(f"eth: {total_money_eth}")
# print(f"btc: {total_money_btc}")
# print(f"ltc: {total_money_ltc}")





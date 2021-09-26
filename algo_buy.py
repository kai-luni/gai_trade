from datetime import datetime, timedelta
import os

# import numpy as np
# import matplotlib.pyplot as plt

from test_api import BitcoinDeApi
from ObjectsGai import TradeBuy, TradeSell

def BuySomeCoins(owned_money, currency, bought_date, sell_at_factor, sell_part):
    coins_to_buy = (owned_money / sell_part) / currency

    trade = TradeBuy()
    trade.currency = "EUR"
    trade.amount = coins_to_buy
    trade.buy_at_date = bought_date
    trade.buy_at_euro = currency
    trade.sell_at_euro = currency * sell_at_factor
    trade.buy_text = f"buy {coins_to_buy} for {coins_to_buy*currency}"

    return trade

def SellSomeCoins(owned_coins, currency, date, buy_at_factor):
    coins_to_sell = (owned_coins / 4)

    trade = TradeSell()
    trade.currency = "EUR"
    trade.amount = coins_to_sell
    trade.sold_at_date = date
    trade.sell_at_euro = currency
    trade.buy_at_euro = currency * buy_at_factor
    trade.sell_text = f"sell {coins_to_sell} for {coins_to_sell*currency}"
    
    return trade

def trade(all_exchange_items, days_look_back, sell_at_factor, percent_loss_buy, time_between_trades, sell_part, amount_coins_multiplyer, sell_above_20_average, start, final_end, always_spend=False):
    #start =  datetime(2017, 6, 2, 16, 0, 0, 0)
    
    #start = datetime(2019, 1, 20, 16, 0, 0, 0)
    end = start
    #final_end = datetime(2020, 5, 14, 0, 0, 0, 0)
    #final_end = datetime(2021, 5, 14, 0, 0, 0, 0)
    counter_slope = 1
    trade_list = []
    money = 600.
    coins = 0.
    total_money = 0.
    while end < final_end:
        
        end = start + timedelta(days=days_look_back)
        exchange_items = BitcoinDeApi.FilterExchangeItems(start, end, all_exchange_items)
        total_money = money + (coins*exchange_items[-1].open)

        value_start = exchange_items[0].open
        value_end = exchange_items[-1].open
        factor_change_value = value_end / value_start

        #>>> sell
        for trade_item in trade_list:
            if trade_item.done:
                continue
            if exchange_items[-1].open > trade_item.sell_at_euro:
                # if (trade_item.amount * exchange_items[-1].high) > (money * 0.4):
                #     continue
                # new_amount = trade_item.amount / sell_at_factor
                # sell_amount = new_amount * exchange_items[-1].open
                # if sell_amount > money:
                #     continue
                amount = trade_item.amount * amount_coins_multiplyer
                if amount > coins:
                    continue

                trade_item.done = True
                trade_item.sell_text = f"sold {amount} for {amount * exchange_items[-1].open}"
                trade_item.sold_at_date = exchange_items[-1].date
                #amount = (money*.4) / exchange_items[-1].high
                money += amount * exchange_items[-1].open
                coins -= amount
                #print(f"{exchange_items[-1].date}: SELL {trade_item.amount} coins for {trade_item.amount * exchange_items[-1].high}EUR. money: {money} coins: {coins}")
                # if money > 300:
                #     #print(f"moneyhave: {money}, buy {(money/3) / exchange_items[-1].open} coins for {money/3} at {exchange_items[-1].date}")
                #     money -= (money/3)
                #     coins += (money/3) / exchange_items[-1].open


        # x_narray = np.array(x_list)
        # y_narray = np.array(y_list)
        # m, b = np.polyfit(x_narray, y_narray, 1)
        #percent = (((y_list[0] + m) / y_list[0]) * 100) - 100
        # if end.year == 2020 and end.month == 3 and end.day==18:
        #     print(percent)
        #     plt.plot(x_narray, y_narray, 'o')
        #     plt.plot(x_narray, m*x_narray + b)
        #     plt.show()

        # >>> buy
        #trade time
        last_trade_date = datetime(1, 1, 1, 1, 1, 1, 1) if len(trade_list) == 0 else trade_list[-1].buy_at_date
        trade_time_ok = last_trade_date + timedelta(days=time_between_trades) < end

        #20 week average
        above_twenty_week_average = True
        if sell_above_20_average == True:
            twenty_week_average = BitcoinDeApi.GetTwentyWeekAverage(all_exchange_items, start + timedelta(days=-1))
            above_twenty_week_average = twenty_week_average and exchange_items[-1].open < twenty_week_average
        if (factor_change_value < percent_loss_buy and money > 100 and trade_time_ok and above_twenty_week_average) or (always_spend and money > 100):
            trade = BuySomeCoins(money, exchange_items[-1].open, exchange_items[-1].date, sell_at_factor, sell_part)

            trade_list.append(trade)
            money -= trade.amount*trade.buy_at_euro
            coins += trade.amount

        new_start = start + timedelta(days=1)
        if new_start.month != start.month:
            money += 50 
        start = new_start
    # for trade in trade_list:
    #     if trade.done:
    #         continue
    #     print(f"buy: {trade.buy_at_euro} amount: {trade.amount} at: {trade.buy_at_date}")
    return total_money

# for trade in trade_list:
#     if trade.done:
#         continue
#     print(f"sell: {trade.sell_at_euro} at: {trade.bought_date_utc}")

    #print(f"Date {start}, Slope {m}")
# plt.plot(x_narray, y_narray, 'o')
# plt.plot(x_narray, m*x_narray + b)
# plt.show()

#for sell_part in[1.2, 1.5, 1.8, 2, 3, 4, 6]:


best_win_one = 0
best_win_two = 0
best_win_three = 0
all_exchange_items = BitcoinDeApi.get_exchange_items()
#total_money_one_no_buy = trade(all_exchange_items, 1, 1000, 0.8, 8, 1, 1, False, datetime(2015, 10, 15, 16, 0, 0, 0), datetime(2021, 5, 26, 0, 0, 0, 0))
# total_money_two_no_buy = trade(all_exchange_items, 1, 1000, 0.8, 8, 1, 1, datetime(2016, 9, 12, 16, 0, 0, 0), datetime(2020, 5, 19, 16, 0, 0, 0))
# total_money_three_no_buy = trade(all_exchange_items, 1, 1000, 0.8, 8, 1, 1, False, datetime(2015, 10, 15, 16, 0, 0, 0), datetime(2021, 5, 14, 0, 0, 0, 0))
# print(total_money_three_no_buy)
# total_money_three_no_ = trade(all_exchange_items, 7, 1000, 0.88, 5, 1, 2., False, datetime(2015, 10, 15, 16, 0, 0, 0), datetime(2021, 5, 14, 0, 0, 0, 0))
# print(total_money_three_no_)

filename = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ".txt"
write_or_append = "a" if os.path.isfile(filename) else "w"
outF = open(filename, write_or_append)
for days_look_back in [1, 3, 7, 14, 21, 40]:
    for sell_at_factor in [1000]:
        for percent_loss_buy in [0.99 ,0.95, 0.9, 0.88, .85, .8, .7, .6]:
            for time_between_trades in [1, 5, 40, 80]:
                for sell_part in[1]:
                    for amount_coins_multiplyer in [1.]:
                        for sell_only_above_20_average in [False]:
                            total_money_one = trade(all_exchange_items, days_look_back, sell_at_factor, percent_loss_buy, time_between_trades, sell_part, amount_coins_multiplyer, sell_only_above_20_average, datetime(2015, 10, 15, 16, 0, 0, 0), datetime(2021, 5, 14, 0, 0, 0, 0))
                            total_money_two = trade(all_exchange_items, days_look_back, sell_at_factor, percent_loss_buy, time_between_trades, sell_part, amount_coins_multiplyer, sell_only_above_20_average, datetime(2017, 12, 1, 16, 0, 0, 0), datetime(2021, 5, 14, 0, 0, 0, 0))
                            total_money_three = trade(all_exchange_items, days_look_back, sell_at_factor, percent_loss_buy, time_between_trades, sell_part, amount_coins_multiplyer, sell_only_above_20_average, datetime(2020, 4, 19, 16, 0, 0, 0), datetime(2021, 5, 14, 0, 0, 0, 0))
                            string_add = ""
                            if (total_money_one  > best_win_one and total_money_two > best_win_two) or (total_money_one  > best_win_one and total_money_three > best_win_three) or (total_money_two  > best_win_two and total_money_three > best_win_three):
                                best_win_one = total_money_one
                                best_win_two = total_money_two
                                best_win_three = total_money_three
                                print(f">>>>>> {best_win_one} {best_win_two} {best_win_three}")
                                string_add = ">>>"
                            output_string = f"{string_add} TMoney {total_money_one} two {total_money_two} three {total_money_three} dayslookback {days_look_back} sellatfactor {sell_at_factor} percentlossbuy {percent_loss_buy} timebetweentrades {time_between_trades} sellpart {sell_part} amountcoinsmultiplyer {amount_coins_multiplyer} over20wa {sell_only_above_20_average}"
                            #print(output_string)
                            if len(string_add) > 0:
                                outF.write(output_string + "\n")
outF.close()

# date_one_from = datetime(2016, 1, 15, 16, 0, 0, 0)
# date_one_to = datetime(2020, 3, 14, 0, 0, 0, 0)
# date_two_from = datetime(2017, 2, 1, 16, 0, 0, 0)
# date_two_to = datetime(2020, 5, 14, 0, 0, 0, 0)

# total_money_one = trade(all_exchange_items, 7, 1.9, 0.88, 5, 1, 1.2, False, date_one_from, date_one_to)
# total_money_two = trade(all_exchange_items, 7, 1.9, 0.88, 5, 1, 1.2, False, date_two_from, date_two_to)
# total_money_three = trade(all_exchange_items, 7, 1.9, 0.88, 5, 1, 1.2, False, datetime(2020, 4, 19, 16, 0, 0, 0), datetime(2021, 5, 14, 0, 0, 0, 0))

# total_money_one_dca = trade(all_exchange_items, 7, 1000, 0.88, 5, 1, 2., False, date_one_from, date_one_to, always_spend=True)
# total_money_two_dca = trade(all_exchange_items, 7, 1000, 0.88, 5, 1, 2., False, date_two_from, date_two_to, always_spend=True)
# total_money_three_dca = trade(all_exchange_items, 7, 1000, 0.88, 5, 1, 2., False, datetime(2020, 4, 19, 16, 0, 0, 0), datetime(2021, 5, 14, 0, 0, 0, 0), always_spend=True)

# print(f"{total_money_one} {total_money_two} {total_money_three} vs {total_money_one_dca} {total_money_two_dca} {total_money_three_dca}")
# total_money_one = trade(all_exchange_items, 14, 1.4, 0.8, 80, 1, 1.05, True, datetime(2017, 6, 2, 16, 0, 0, 0), datetime(2021, 5, 14, 0, 0, 0, 0))
# print(f"{total_money_one}")
#print(f"{string_add} TMoney {total_money} dayslookback {days_look_back} buyatfactor {buy_at_factor} percentwinsell {percent_loss_buy} timebetweentrades {time_between_trades}")


# total_money_one = trade(all_exchange_items, 3, 1.4, 0.9, 3, 2, 1.5, datetime(2015, 10, 15, 16, 0, 0, 0), datetime(2021, 5, 26, 0, 0, 0, 0))
# total_money_two = trade(all_exchange_items, 3, 1.4, 0.9, 3, 2, 1.5, datetime(2016, 9, 12, 16, 0, 0, 0), datetime(2020, 5, 19, 16, 0, 0, 0))
# total_money_three = trade(all_exchange_items, 3, 1.4, 0.9, 3, 2, 1.5, datetime(2018, 12, 19, 16, 0, 0, 0), datetime(2021, 4, 13, 0, 0, 0, 0))
# print(f">>>>>> {total_money_one} {total_money_two} {total_money_three} vs {total_money_one_no_buy} {total_money_two_no_buy} {total_money_three_no_buy}")





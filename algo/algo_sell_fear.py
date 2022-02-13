from datetime import datetime, timedelta

from api.alternative_fear_greed_api import AlternativeFearGreedApi
from db.exchange_rate_data_sql import ExchangeRateData
from db.fear_greed_data_sql import FearGreedData
from data.fear_greed_data_reader import FearGreedDataReader
from dto.ObjectsGai import ExchangeRateItem, TradeSellParams, FearGreedItem

#TODO: finish this sqlite reader
_fear_greed_entries = AlternativeFearGreedApi(5000)

class AlgoSellFear:
    def buy(money : float, coins : float, currency : float):
        """calculate buy operation

        Args:
            money (float): money owned
            coins (float): coins owned
            currency (float): exchange rate

        Returns:
            float, float: money owned (always 0), coins owned after buying 
        """
        if money <= 0:
            return money, coins
        return 0, coins + (money / currency)

    def sell(money : float, coins : float, currency : float):
        """calculate buy operation

        Args:
            money (float): money owned
            coins (float): coins owned
            currency (float): exchange rate 

        Returns:
            float, float: money owned, coins owned (always 0) after selling
        """
        if coins <= 0:
            return money, coins
        return money + (coins * currency), 0

    def trade(params : TradeSellParams, debug=False):
        """simulate trading operations for a timerange

        Args:
            params (TradeSellParams): parameters, check out the object for details
            debug (bool, optional): give out some debug information. Defaults to False.

        Returns:
            float: final coins
        """
        full_path = ExchangeRateData.getDbFullPath(params.coin, params.currency, "./db/")
        all_entries : 'list[ExchangeRateItem]' = ExchangeRateData.get_all_items(full_path)
        filtered_entries : 'list[ExchangeRateItem]' = ExchangeRateData.filter_exchange_items(params.start, params.end, all_entries)
        

        coins = 1
        money = 0
        dates_sell = []
        dates_buy = []
        last_sell_rate = 0.

        # for entry in filtered_entries[params.days_look_back, len(filtered_entries)]:
        #     print(f"{entry.unix} {entry.date}")
        for i in range(params.days_look_back, len(filtered_entries)):
            today = filtered_entries[i].date
            currency_entry : ExchangeRateItem = filtered_entries[i]
            greed_fear_entry = _fear_greed_entries.get_entry_for_day(filtered_entries[i].date.day, filtered_entries[i].date.month, filtered_entries[i].date.year)
            if greed_fear_entry == None:
                greed_fear_entry = FearGreedItem()

            ### sell
            today_exchange_rate = currency_entry.close
            first_entry_look_back_exchange = filtered_entries[i-params.days_look_back].close
            factor_change_value = today_exchange_rate / first_entry_look_back_exchange
            greed_fear_index_sell = greed_fear_entry.index if  greed_fear_entry.index > -1 else 100
            if coins > 0 and factor_change_value > params.percent_change_sell and greed_fear_index_sell > 25:
                money, coins = AlgoSellFear.sell(money, coins, currency_entry.close)
                dates_sell.append(today)
                last_sell_rate = today_exchange_rate
                if debug:
                    print(f"sell on {today}, we now have {money} EUR")

            ### buy
            if greed_fear_entry.index == -1 and today.day != 1:
                continue
            if greed_fear_entry.index < params.buy_at_gfi and money > 0 and today_exchange_rate < last_sell_rate:
                money, coins = AlgoSellFear.buy(money, coins, currency_entry.close)
                dates_buy.append(today)
                if debug:
                    print(f"buy on {filtered_entries[i].date}, we now have {coins} coins.")
            
        final_coins = coins + (money/filtered_entries[-1].close)
        if debug:
            print(f"Final Coins: {final_coins}, {params.days_look_back} {params.buy_at_gfi} {params.percent_change_sell}")
            x = [i.date for i in filtered_entries]
            y = [i.close for i in filtered_entries]
            AlgoSellFear.create_diagram(x, y, dates_buy, dates_sell, params.coin)
        return final_coins

    def create_diagram(x : 'list[datetime]', y : 'list[float]', buy_dates : 'list[datetime]', sell_dates : 'list[datetime]', coin_name : str):
        """write a diagram of a coin with buy dates (green) and sell dates (red) for visually checking the sanity of the trading algo

        Args:
            x (list[datetime]): datetimes, like each day one datetime   
            y (list[float]): same length as x, with exchange rate of that day  
            buy_dates (list[datetime]): dates the algo bought
            sell_dates (list[datetime]): dates the algo sold
            coin_name (str): name of coin for filename
        """
        import matplotlib.pyplot as plt
        # plot
        plt.plot(x,y)
        for buy_date in buy_dates:
            plt.axvline(x=buy_date, color="green")
        for sell_date in sell_dates:
            plt.axvline(x=sell_date, color="red")
        # beautify the x-labels
        plt.gcf().autofmt_xdate()

        plt.savefig(f"buy_sell_output_{coin_name}.png")



if __name__ == "__main__":
    params = TradeSellParams()
    params.start = datetime(2015, 10, 1, 0, 0, 0, 0)
    params.end = datetime(2022, 1, 12, 0, 0, 0, 0)
    params.days_look_back = 165
    params.buy_at_gfi = 11
    params.percent_change_sell = 4.64

    # params.coin = "BTC"
    # params.currency = "EUR"
    # AlgoSellFear.trade(params, debug = True)

    # exit(0)

    for dlb in range(55, 250, 10):
        print(f"dlb {dlb}")
        for pcs in range(105, 770, 15):
            for bf in range(7, 15, 1):
                params.days_look_back = dlb
                params.percent_change_sell = float(pcs) / 100.
                params.buy_at_gfi = bf

                params.coin = "BTC"
                coins_btc = AlgoSellFear.trade(params)
                if coins_btc < 2.:
                    continue

                params.coin = "ETH"
                coins_eth = AlgoSellFear.trade(params)
                if coins_eth < 2.:
                    continue

                params.coin = "LTC"
                coins_ltc = AlgoSellFear.trade(params)
                if coins_ltc < 2.:
                    continue

                params.coin = "XLM"
                coins_xlm = AlgoSellFear.trade(params)
                if coins_xlm < 1.:
                    continue
                
                print(f"{coins_btc} {coins_eth} {coins_ltc} {coins_xlm} lookback: {params.days_look_back}, percent_change_sell: {params.percent_change_sell} buy_at_gfi: {params.buy_at_gfi}")
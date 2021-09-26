from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from ExchangeRateData import ExchangeRateData
from ObjectsGai import ExchangeRateItem
from machine_learning.data_calc import DataCalc

class GaiPlot:
    def print_currency_plot(from_date: datetime, to_date: datetime):
        exchange_entries = ExchangeRateData.get_exchange_items("db/LTC_EUR_echange_db.csv")
        filtered_items = ExchangeRateData.filter_exchange_items(exchange_entries, from_date, to_date)
        daily_date = [day.date for day in filtered_items]
        daily_exchange = [((day.high-day.low)/2) for day in filtered_items]
        fig, ax = plt.subplots()
        fig.autofmt_xdate()
        ax.plot(daily_date,daily_exchange)
        plt.show()

    def print_prediction_plot_compare(dates : datetime, exchange_item_prediction : "list[ExchangeRateItem]"):
        exchange_entries = ExchangeRateData.get_exchange_items("db/BTC_EUR_echange_db.csv")
        filtered_items = ExchangeRateData.filter_exchange_items(exchange_entries, datetime(dates[0].year, dates[0].month, dates[0].day), datetime(dates[-1].year, dates[-1].month, dates[-1].day, 23, 59, 56))
        daily_date = [day.date for day in filtered_items]
        daily_exchange = [day.low+((day.high-day.low)/2) for day in filtered_items]
        # for day in filtered_items:
        #     print(f"day {day.date} er {day.low+((day.high-day.low)/2)} low {day.low} high {day.high}")
        fig, ax = plt.subplots()
        fig.autofmt_xdate()
        ax.plot(dates, exchange_item_prediction, label="Prediction")        
        ax.plot(daily_date, daily_exchange, label="Reality")
        ax.legend(loc='best')
        ax.grid()
        plt.show()

    def print_diagram(x_values, y_values_list, labels):
        fig, ax = plt.subplots()
        ax.set_title("Deviance")
        ax.set_xlabel("Date")
        ax.set_ylabel("Percent")
        for i in range(len(y_values_list)):
            y_values = y_values_list[i]
            ax.plot(x_values, y_values, label=labels[i])
        ax.grid()
        ax.legend(loc='best')
        #print proper date        
        plt.gcf().autofmt_xdate()
        #plt.savefig("test.png")
        plt.show()
        plt.close(fig)

    def print_forecast_deviation(dates : np.array, prices :np.array, forecasts : np.array):
        """print deviation in percent from price and forecast

        Args:
            dates (np.array): dates
            prices (np.array): real prices
            forecasts (np.array): forecast prices
        """
        if len(dates) != len(forecasts):
            raise AssertionError("Lists need to have same length")
        if len(dates) != len(prices):
            raise AssertionError("Lists need to have same length")
        deviations = []
        for i in range(len(prices)):
            deviations.append(DataCalc.diff_percentage(price=prices[i], price_forecast=forecasts[i]))
        #print(f"deviations: {np.mean(np.array(deviations))}")

        GaiPlot.print_diagram([dates], np.array([deviations]))

    def print_forecast_deviation_relative_last_day(dates : np.array, forecasts : np.array):
        if len(dates) != len(forecasts):
            raise AssertionError("Lists need to have same length")

        exchange_entries = ExchangeRateData.get_exchange_items("db/BTC_EUR_echange_db.csv")
        filtered_items = ExchangeRateData.filter_exchange_items(exchange_entries, datetime(dates[0].year, dates[0].month, dates[0].day), datetime(dates[-1].year, dates[-1].month, dates[-1].day, 23, 59, 56))
        #daily_date = [day.date for day in filtered_items]
        prices = [day.low+((day.high-day.low)/2) for day in filtered_items]
        if len(dates) != len(prices):
            raise AssertionError("Lists need to have same length")
        expected_deviation = []
        prediction_deviation = []
        for i in range(len(prices)):
            if i == 0:
                continue
            expected_deviation.append(DataCalc.diff_percentage(price=prices[i-1], price_forecast=prices[i]))
            prediction_deviation.append(DataCalc.diff_percentage(price=prices[i-1], price_forecast=forecasts[i]))
            print(f"date {dates[i]} from {prices[i-1]} real {prices[i]} pred {forecasts[i]} %real {expected_deviation[-1]} %pred {prediction_deviation[-1]}")
        GaiPlot.print_diagram(dates[1:len(dates)], [expected_deviation, prediction_deviation], ["Expected Deviation", "Prediction Deviation"])





if __name__ == "__main__":
    from_date = datetime(2020, 1, 1)
    to_date = datetime(2021, 12, 31)
    GaiPlot.print_currency_plot(from_date, to_date)
from datetime import datetime
from ObjectsGai import ExchangeRateItem
from test_api import BitcoinDeApi

file1 = open('gemini_BTCUSD_day.csv', 'r')
lines = file1.readlines()

counter = 0
exchange_items = []
for line in lines:
    if counter < 2:
        counter += 1
        continue
    values = line.replace("\n", "").split(",")
    exchange_item = ExchangeRateItem()
    exchange_item.unix = int(values[0])
    exchange_item.low = float(values[5])
    exchange_item.high = float(values[4])
    exchange_item.open = float(values[3])
    exchange_item.close = float(values[6])
    exchange_item.volume = float(values[7])
    exchange_item.date = datetime.strptime(values[1], '%Y-%m-%d %H:%M:%S')
    exchange_items.append(exchange_item)
    
exchange_items_ordered = []
for ex_item in reversed(exchange_items):
    ex_item.twenty_week_average = BitcoinDeApi.GetTwentyWeekAverage(exchange_items_ordered, ex_item.date)
    exchange_items_ordered.append(ex_item)

outF = open("btc_usd_20151009_20210701.csv", "w")
outF.write("unix,low,high,open,close,volume,date\n")
for ex_item in exchange_items_ordered:
    line_to_write = f"{ex_item.unix};{ex_item.low};{ex_item.high};{ex_item.open};{ex_item.close};{ex_item.volume};{ex_item.date};{ex_item.twenty_week_average}\n"
    outF.write(line_to_write)
outF.close()


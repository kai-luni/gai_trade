import csv
from pycoingecko import CoinGeckoAPI
import concurrent.futures
import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pycoingecko import CoinGeckoAPI

cg = CoinGeckoAPI()

def get_monthly_volume(coin_id, date_from, date_to):
    date_from_dt = datetime(date_from.year, date_from.month, date_from.day)
    date_to_dt = datetime(date_to.year, date_to.month, date_to.day)
    historical_data = cg.get_coin_market_chart_range_by_id(coin_id, vs_currency='usd', from_timestamp=date_from_dt.timestamp(), to_timestamp=date_to_dt.timestamp())
    volume_data = historical_data['total_volumes']

    monthly_volume = 0
    for data in volume_data:
        timestamp, volume = data
        monthly_volume += volume

    return monthly_volume / len(volume_data) * 30

def get_top_cryptos_by_market_cap(count=200):
    top_cryptos = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=count, page=1)
    filtered_cryptos = [crypto for crypto in top_cryptos if crypto['id'] not in ['tether', 'usd-coin', 'staked-ether', 'dai']]
    return filtered_cryptos


def fetch_historical_data(coin, date):
    try:
        coin_id = coin['id']
        historical_snapshot = cg.get_coin_history_by_id(coin_id, date.strftime('%d-%m-%Y'))
        market_data = historical_snapshot['market_data']
        if market_data:
            return {
                'id': coin_id,
                'name': historical_snapshot['name'],
                'symbol': historical_snapshot['symbol'],
                'market_cap': coin['market_cap']
            }
    except Exception as e:
        pass
    return None

def get_historical_top_cryptos(date, count=100):
    historical_data = []
    coins_list = get_top_cryptos_by_market_cap(count)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_historical_data, coin, date) for coin in coins_list]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                historical_data.append(result)

    sorted_data = sorted(historical_data, key=lambda x: x['market_cap'], reverse=True)[:10]
    return sorted_data

def get_data():
    # Get the current month and year
    today = datetime.date.today()
    month = today.month
    year = today.year

    with open('top_cryptos_by_ratio.txt', 'w') as f:
        f.write("Top 2 cryptocurrencies by market cap to volume ratio for each month in the last 36 months:\n")

        for i in range(36):
            first_day_of_month = datetime.date(year, month, 1)
            prev_month_last_day = first_day_of_month - datetime.timedelta(days=1)
            prev_month_first_day = datetime.date(prev_month_last_day.year, prev_month_last_day.month, 1)

            top_cryptos = get_historical_top_cryptos(prev_month_first_day)

            ratios = []
            for crypto in top_cryptos:
                coin_id = crypto['id']
                name = crypto['name']
                symbol = crypto['symbol'].upper()
                market_cap = crypto['market_cap']

                monthly_volume = get_monthly_volume(coin_id, prev_month_first_day, prev_month_last_day)
                ratio = monthly_volume / market_cap
                ratios.append((name, symbol, ratio, market_cap))

            top_2 = sorted(ratios, key=lambda x: x[2], reverse=True)[:2]

            print(f"{prev_month_first_day.strftime('%B %Y')}:\n")
            f.write(f"{prev_month_first_day.strftime('%B %Y')}:\n")
            for coin in top_2:
                name, symbol, ratio, _ = coin
                print(f"  {name} ({symbol}): Market Cap to Volume Ratio = {ratio:.2f}\n")
                f.write(f"  {name} ({symbol}): Market Cap to Volume Ratio = {ratio:.2f}\n")

            # Update month and year values
            month -= 1
            if month == 0:
                month = 12
                year -= 1

    print("Results saved to top_cryptos_by_ratio.txt")



def get_current_price(crypto, date=None):
    if date:
        # Convert date to timestamp
        timestamp = int(date.timestamp())
        # Fetch the historical price of the given crypto from CoinGecko API
        price_data = cg.get_coin_market_chart_range_by_id(id=crypto.lower(), vs_currency='eur', from_timestamp=timestamp, to_timestamp=timestamp+1000)
        
        if not price_data['prices']:
            # Handle the situation when the prices list is empty, e.g., return None, skip the iteration, or use an alternative price
            return None

        return price_data['prices'][0][1]
    else:
        # Fetch the current price of the given crypto from CoinGecko API
        price_data = cg.get_price(ids=crypto.lower(), vs_currencies='eur')
        return price_data[crypto.lower()]['eur']


def main():
    coins_list = cg.get_coins_list()
    total_value = 0.
    total_spent = 0.

    crypto_dict = {}
    for coin in coins_list:
        symbol = coin['symbol']
        coin_id = coin['id']
        if symbol not in crypto_dict:
            crypto_dict[symbol] = coin_id

    with open("top_vol_ratio.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            date = datetime(int(row["Year"]), int(row["Month"]), 1)
            date_with_added_month = date + relativedelta(months=1)

            print(f"Now at {date_with_added_month}")

            crypto1 = row["Crypto1"]
            crypto2 = row["Crypto2"]

            # Buy 100 Euro worth of each cryptocurrency
            value1 = get_current_price(crypto_dict[crypto1.lower()], date_with_added_month)
            value2 = get_current_price(crypto_dict[crypto2.lower()], date_with_added_month)
            amount1 = 100 / value1
            amount2 = 100 / value2

            # Add the value of the bought cryptocurrencies at the current price
            total_value += amount1 * get_current_price(crypto_dict[crypto1.lower()])
            total_value += amount2 * get_current_price(crypto_dict[crypto2.lower()])

            # Add the spent amount (100 Euro for each cryptocurrency)
            total_spent += 100 + 100

    # Calculate the net value
    net_value = total_value - total_spent

    print(f"Total spent: {total_spent} Euro")
    print(f"Total value today: {total_value} Euro")
    print(f"Net value: {net_value} Euro")

main()
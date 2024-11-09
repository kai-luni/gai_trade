from ast import Dict
from datetime import datetime, timedelta

from dto.ExchangeRateItem import ExchangeRateItem
from dto.RiskItem import RiskEntry
from repos import AlphaSquaredRepo, CoinBaseRepo



class TestAlphaSquaredStrategy:
    def run():
        """
        The main function to load and process AlphaSquared and BTC/EUR data for the specified date range.
        """
        # Define the date range
        # from_date = "2020-01-01"
        # to_date = "2024-11-10"

        try:
            start_filter = datetime(2018, 12, 31)
            end_filter = datetime(2024, 1, 7)

            print(f"Loading AlphaSquared data from {start_filter} to {end_filter}...")
            alphasq_data = AlphaSquaredRepo.AlphaSquaredRepo.get_risk_entries(start_filter, end_filter, "repos/AlphaSquaredRiskData_BTC.csv")
            print(f"Loaded {len(alphasq_data)} AlphaSquared entries.")

            print(f"Loading BTC/EUR data from {start_filter} to {end_filter}...")
            dictlist_btc = CoinBaseRepo.CoinBaseRepo.read_csv_to_dict('repos/BTC_EUR.csv')
            sample_exchange_rates = CoinBaseRepo.CoinBaseRepo.get_exchange_rate_items(start_filter, end_filter, dictlist_btc)
            print(f"Loaded {len(sample_exchange_rates)} BTC/EUR entries.")            

            # Process or analyze the data as needed
            print("Start Simulation...")
            TestAlphaSquaredStrategy.simulate_trading(alphasq_data, sample_exchange_rates, start_filter, end_filter)
            TestAlphaSquaredStrategy.simple_monthly_investment(sample_exchange_rates, start_filter, end_filter)

        except Exception as e:
            print(f"Error: {e}")

    def simulate_trading(alphasq_data: 'list[RiskEntry]', exchange_rate_data: 'list[ExchangeRateItem]', start_date: datetime, end_date: datetime):
        """
        Simulates trading based on the provided AlphaSquared risk data and BTC/EUR exchange rates.
        
        Parameters:
        - alphasq_data: list of RiskEntry
        - exchange_rate_data: list of ExchangeRateItem
        - start_date: datetime
        - end_date: datetime
        
        Returns:
        - A list of daily portfolio values.
        """
        
        # Initialize depot and holdings
        depot = 0.0  # in Euros
        holdings = 0.0  # in BTC
        
        # Prepare data structures for quick lookup
        # Create a dict mapping dates to risk values
        risk_dict = {entry.date: entry.risk for entry in alphasq_data}
        # Create a dict mapping dates to exchange rates (close price)
        exchange_rate_dict = {item.date.date(): item.close for item in exchange_rate_data}
        
        # To handle missing data, we can create sorted lists of dates
        risk_dates = sorted(risk_dict.keys())
        exchange_rate_dates = sorted(exchange_rate_dict.keys())
        
        # For recording portfolio value over time
        portfolio_values = []
        
        # Start simulation
        current_date = start_date.date()
        end_date = end_date.date()
        
        # We can store the last known risk and exchange rate
        last_risk = None
        last_exchange_rate = None
        
        # Loop over each day
        while current_date <= end_date:
            # Update last known risk
            if current_date in risk_dict:
                last_risk = risk_dict[current_date]
            elif last_risk is None:
                # Find the previous available risk value
                prior_dates = [date for date in risk_dates if date <= current_date]
                if prior_dates:
                    last_risk = risk_dict[prior_dates[-1]]
                else:
                    last_risk = None  # no prior risk data
            
            # Update last known exchange rate
            if current_date in exchange_rate_dict:
                last_exchange_rate = exchange_rate_dict[current_date]
            elif last_exchange_rate is None:
                # Find the previous available exchange rate
                prior_dates = [date for date in exchange_rate_dates if date <= current_date]
                if prior_dates:
                    last_exchange_rate = exchange_rate_dict[prior_dates[-1]]
                else:
                    last_exchange_rate = None  # no prior exchange rate data
            
            # If it's the first of the month, add 200 Euro to depot
            if current_date.day == 1:
                depot += 200.0
                # After adding 200 Euro, check the risk and possibly buy bitcoin
                if last_risk is not None and last_exchange_rate is not None:
                    # Determine the percentage to invest based on risk
                    invest_percent = 0.0
                    risk = last_risk
                    if 60 <= risk < 70:
                        invest_percent = 0.05
                    elif 50 <= risk < 60:
                        invest_percent = 0.10
                    elif 40 <= risk < 50:
                        invest_percent = 0.20
                    elif 30 <= risk < 40:
                        invest_percent = 0.30
                    elif 20 <= risk < 30:
                        invest_percent = 0.35
                    elif 10 <= risk < 20:
                        invest_percent = 0.40
                    elif 0 <= risk < 10:
                        invest_percent = 0.50
                    # Calculate the amount to invest
                    amount_to_invest = depot * invest_percent
                    if amount_to_invest > 0:
                        # Buy bitcoin
                        btc_bought = amount_to_invest / last_exchange_rate
                        holdings += btc_bought
                        depot -= amount_to_invest
                        # print(f">>>{current_date}: Bought {btc_bought:.6f} BTC for {amount_to_invest:.2f} EUR at risk {risk}%")
                        # TestAlphaSquaredStrategy.display_portfolio(portfolio_values)
            
            # If it's Monday, check the risk and possibly sell holdings
            if current_date.weekday() == 0:  # Monday is 0
                if last_risk is not None and last_exchange_rate is not None:
                    risk = last_risk
                    if risk > 90:
                        # Sell 70% of holdings
                        btc_to_sell = holdings * 0.70
                        amount_received = btc_to_sell * last_exchange_rate
                        holdings -= btc_to_sell
                        depot += amount_received
                        print(f">>>{current_date}: Sold {btc_to_sell:.6f} BTC for {amount_received:.2f} EUR at risk {risk}%")
                        TestAlphaSquaredStrategy.display_portfolio(portfolio_values)
                    elif risk > 80:
                        # Sell 30% of holdings
                        btc_to_sell = holdings * 0.30
                        amount_received = btc_to_sell * last_exchange_rate
                        holdings -= btc_to_sell
                        depot += amount_received
                        print(f">>>{current_date}: Sold {btc_to_sell:.6f} BTC for {amount_received:.2f} EUR at risk {risk}%")
                        TestAlphaSquaredStrategy.display_portfolio(portfolio_values)
            
            # Calculate portfolio value
            if last_exchange_rate is not None:
                holdings_value = holdings * last_exchange_rate
            else:
                holdings_value = holdings * 0  # No exchange rate data
            total_portfolio_value = depot + holdings_value
            portfolio_values.append({
                'date': current_date,
                'depot': depot,
                'exchange_rate': last_exchange_rate,
                'holdings': holdings,
                'holdings_value': holdings_value,
                'total_value': total_portfolio_value,
            })
            
            # Move to next day
            current_date += timedelta(days=1)
        
        print(">>>")
        TestAlphaSquaredStrategy.display_portfolio(portfolio_values)
        return portfolio_values
    
    def simple_monthly_investment(exchange_rate_data: 'list[ExchangeRateItem]', start_date: datetime, end_date: datetime):
        """
        Buys BTC for 200 Euro on the first of each month from start_date to end_date.
        
        Parameters:
        - exchange_rate_data: list of ExchangeRateItem
        - start_date: datetime
        - end_date: datetime
        
        Returns:
        - Total value of holdings at end_date.
        """
        holdings = 0.0  # in BTC

        # Prepare exchange rate dict
        exchange_rate_dict = {item.date.date(): item.close for item in exchange_rate_data}
        exchange_rate_dates = sorted(exchange_rate_dict.keys())

        current_date = start_date.date()
        end_date = end_date.date()

        last_exchange_rate = None

        while current_date <= end_date:
            # Update last known exchange rate
            if current_date in exchange_rate_dict:
                last_exchange_rate = exchange_rate_dict[current_date]
            elif last_exchange_rate is None:
                # Find the previous available exchange rate
                prior_dates = [date for date in exchange_rate_dates if date <= current_date]
                if prior_dates:
                    last_exchange_rate = exchange_rate_dict[prior_dates[-1]]
                else:
                    last_exchange_rate = None  # No prior exchange rate data

            # If it's the first of the month, buy BTC for 200 Euro
            if current_date.day == 1 and last_exchange_rate is not None:
                btc_bought = 200.0 / last_exchange_rate
                holdings += btc_bought
                # Optional: print(f"{current_date}: Bought {btc_bought:.6f} BTC at rate {last_exchange_rate:.2f} EUR")

            # Move to next day
            current_date += timedelta(days=1)

        # At the end, compute total value
        # Get exchange rate at end date
        final_exchange_rate = None
        lookup_date = end_date
        while final_exchange_rate is None and lookup_date >= start_date.date():
            if lookup_date in exchange_rate_dict:
                final_exchange_rate = exchange_rate_dict[lookup_date]
            else:
                lookup_date -= timedelta(days=1)

        if final_exchange_rate is None:
            print("No exchange rate data available for the end date.")
            return None

        total_value = holdings * final_exchange_rate
        print(f"!>>> Only Buy Strategy Total BTC holdings: {holdings:.6f} BTC")
        print(f"Total value on {end_date}: {total_value:.2f} EUR <<<!")

        return total_value
    
    def display_portfolio(portfolio_values: 'list[Dict]'):
        if portfolio_values:
            holdings = portfolio_values[-1]['holdings']
            depot = portfolio_values[-1]['depot']
            total_money = depot + holdings * portfolio_values[-1]['exchange_rate']
        else:
            holdings = 'N/A'
            depot = 'N/A'
            total_money = 'N/A'
        
        print(f"BTC: {holdings} Depot: {depot} Total Money: {total_money} Currency: {portfolio_values[-1]['exchange_rate']} <<<")



# Main execution
if __name__ == "__main__":
    print("run")
    TestAlphaSquaredStrategy.run()

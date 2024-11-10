from ast import Dict
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random

from dto.ExchangeRateItem import ExchangeRateItem
from dto.RiskItem import RiskEntry
from repos import AlphaSquaredRepo, CoinBaseRepo



class TestAlphaSquaredStrategy:
    def run():
        """
        The main function to load and process AlphaSquared and BTC/EUR data for the specified date range.
        """

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
            investment_rules = {
                (0, 10): 0.80,     # Buy 50% of depot
                (10, 20): 0.99,    # Buy 40% of depot
                (20, 30): 0.65,
                (30, 40): 0.76,
                (40, 50): 0.98,
                (50, 60): 0.02,
                (60, 70): 0.35,
                (80, 90): -0.92,   # Sell 30% of holdings
                (90, 100): -0.34   # Sell 70% of holdings
            }
            TestAlphaSquaredStrategy.simulate_trading(alphasq_data, sample_exchange_rates, start_filter, end_filter, investment_rules, print_logs=True)
            TestAlphaSquaredStrategy.simple_monthly_investment(sample_exchange_rates, start_filter, end_filter)

            # print("Start Optimization")
            # TestAlphaSquaredStrategy.optimize_investment_strategy(alphasq_data, sample_exchange_rates, start_filter, end_filter, num_iterations=100000)

        except Exception as e:
            print(f"Error: {e}")

    def run_monthly_comparisons():
        """
        Executes simulate_trading and simple_monthly_investment over rolling 5-year periods,
        starting from 23.04.2015, incrementing the start date by 1 month each time.
        Calculates the factor by which simulate_trading outperforms simple_monthly_investment
        in terms of total portfolio value, and computes the mean of these factors.
        """
        factors = []
        start_filter = datetime(2015, 4, 23)
        end_of_data = datetime(2024, 11, 7)  # Assuming this is the end of available data

        while start_filter + relativedelta(years=5) <= end_of_data:
            end_filter = start_filter + relativedelta(years=5)

            print(f"\nPeriod: {start_filter.date()} to {end_filter.date()}")

            try:

                # Load AlphaSquared data
                alphasq_data = AlphaSquaredRepo.AlphaSquaredRepo.get_risk_entries(
                    start_filter, end_filter, "repos/AlphaSquaredRiskData_BTC.csv")
                # Load BTC/EUR exchange data
                dictlist_btc = CoinBaseRepo.CoinBaseRepo.read_csv_to_dict('repos/BTC_EUR.csv')
                sample_exchange_rates = CoinBaseRepo.CoinBaseRepo.get_exchange_rate_items(
                    start_filter, end_filter, dictlist_btc, throwExceptionMissingDay=False)

                # Define investment rules
                investment_rules = {
                    (0, 10): 1,
                    (10, 20): 1,
                    (20, 30): 1,
                    (30, 40): 1,
                    (40, 50): 1,
                    (50, 60): 0.4,
                    (60, 70): 0.4,
                    (80, 90): -0.3,
                    (90, 100): -0.5
                }

                # Execute simulate_trading
                portfolio_values_sim = TestAlphaSquaredStrategy.simulate_trading(
                    alphasq_data,
                    sample_exchange_rates,
                    start_filter,
                    end_filter,
                    investment_rules,
                    print_logs=False
                )
                final_portfolio_sim = portfolio_values_sim[-1]
                total_value_sim = final_portfolio_sim['total_value']

                # Execute simple_monthly_investment
                total_value_simple = TestAlphaSquaredStrategy.simple_monthly_investment(
                    sample_exchange_rates,
                    start_filter,
                    end_filter
                )

                # Calculate the factor
                if total_value_simple > 0:
                    factor = total_value_sim / total_value_simple
                    factors.append(factor)
                    print(f"{start_filter} to {end_filter}: Factor (simulate_trading / simple_monthly_investment): {factor:.2f}")
                else:
                    print("Simple monthly investment total value is zero or negative.")

            except Exception as e:
                print(f"Error during simulation: {e}")

            # Increment start_filter by 1 month
            start_filter += relativedelta(months=1)

        # Calculate the mean of the factors
        if factors:
            mean_factor = sum(factors) / len(factors)
            print(f"\nMean factor over all periods: {mean_factor:.2f}")
        else:
            print("No factors calculated.")

    def optimize_investment_strategy(
        alphasq_data: 'list[RiskEntry]',
        exchange_rate_data: 'list[ExchangeRateItem]',
        start_date: datetime,
        end_date: datetime,
        num_iterations: int = 100000
    ):
        """
        Executes simulate_trading in a loop with random investment_rules.
        If the total_value of the last entry in the return list is higher than the record,
        prints the investment_rules and total_value.

        Parameters:
        - alphasq_data: list of RiskEntry
        - exchange_rate_data: list of ExchangeRateItem
        - start_date: datetime
        - end_date: datetime
        - num_iterations: int, number of iterations to run
        """
        record_total_value = 0.0
        best_investment_rules = None

        # Define the risk ranges, including the new selling ranges
        risk_ranges = [
            (0, 10),
            (10, 20),
            (20, 30),
            (30, 40),
            (40, 50),
            (50, 60),
            (60, 70),
            (80, 90),   # Selling range
            (90, 100)   # Selling range
        ]

        for i in range(num_iterations):
            # Generate random investment percentages for each risk range
            investment_rules = {}
            for risk_range in risk_ranges:
                # For selling ranges, generate negative percentages
                if risk_range[0] >= 80:
                    invest_percent = -random.uniform(0, 1)  # Negative percentage for selling
                else:
                    invest_percent = random.uniform(0, 1)   # Positive percentage for buying
                investment_rules[risk_range] = invest_percent

            # Run the simulation with the generated investment_rules
            portfolio_values = TestAlphaSquaredStrategy.simulate_trading(
                alphasq_data,
                exchange_rate_data,
                start_date,
                end_date,
                investment_rules
            )

            # Get the total_value of the last day
            final_portfolio = portfolio_values[-1]
            total_value = final_portfolio['total_value']

            # Check if this is a new record
            if total_value > record_total_value:
                record_total_value = total_value
                best_investment_rules = investment_rules
                print(f"New record on iteration {i+1}:")
                print("Investment Rules:")
                for risk_range, percent in investment_rules.items():
                    action = "Invest" if percent >= 0 else "Sell"
                    print(f"  Risk {risk_range}: {action} {abs(percent)*100:.2f}%")
                print(f"Total Portfolio Value: {total_value:.2f} EUR")
                print("----")

        # Optionally, print the best investment rules at the end
        print("\nBest Investment Rules Found:")
        for risk_range, percent in best_investment_rules.items():
            action = "Invest" if percent >= 0 else "Sell"
            print(f"Risk {risk_range}: {action} {abs(percent)*100:.2f}%")
        print(f"Highest Total Portfolio Value: {record_total_value:.2f} EUR")


    def simulate_trading(
        alphasq_data: 'list[RiskEntry]',
        exchange_rate_data: 'list[ExchangeRateItem]',
        start_date: datetime,
        end_date: datetime,
        investment_rules: dict,
        print_logs: bool = False
    ):
        """
        Simulates trading based on the provided AlphaSquared risk data and BTC/EUR exchange rates.

        Parameters:
        - alphasq_data: list of RiskEntry
        - exchange_rate_data: list of ExchangeRateItem
        - start_date: datetime
        - end_date: datetime
        - investment_rules: dict
            A dictionary mapping risk thresholds to investment percentages.
            Positive percentages indicate buying; negative percentages indicate selling.
            Example:
                {
                    (0, 10): 0.50,     # Buy 50% of depot
                    (10, 20): 0.40,    # Buy 40% of depot
                    (20, 30): 0.35,
                    (30, 40): 0.30,
                    (40, 50): 0.20,
                    (50, 60): 0.10,
                    (60, 70): 0.05,
                    (80, 90): -0.30,   # Sell 30% of holdings
                    (90, 100): -0.70   # Sell 70% of holdings
                }
        - print_logs: bool
            If True, prints transaction logs.

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
                raise Exception(f"Risk: This should not happen {current_date}")

            # Update last known exchange rate
            if current_date in exchange_rate_dict:
                last_exchange_rate = exchange_rate_dict[current_date]
            elif last_exchange_rate is None:
                raise Exception(f"Currency: This should not happen {current_date}")

            # Determine the investment or sell percentage based on risk
            invest_percent = 0.0
            if last_risk is not None:
                risk = last_risk

                # Iterate over the investment rules to find the matching risk range
                # Sort the risk ranges by lower_bound to ensure correct order
                sorted_rules = sorted(investment_rules.items(), key=lambda x: x[0][0])
                for (lower_bound, upper_bound), percent in sorted_rules:
                    if lower_bound <= risk < upper_bound:
                        invest_percent = percent
                        break

            # If it's the first of the month, add 200 Euro to depot
            if current_date.day == 1:
                depot += 200.0

            # Buying logic (only on the first of the month)
            if current_date.day == 1 and invest_percent > 0 and depot > 0 and last_exchange_rate is not None:
                amount_to_invest = depot * invest_percent
                if amount_to_invest > 0:
                    btc_bought = amount_to_invest / last_exchange_rate
                    holdings += btc_bought
                    depot -= amount_to_invest
                    if print_logs:
                        print(f"{current_date}: Bought {btc_bought:.6f} BTC for {amount_to_invest:.2f} EUR at risk {risk}%")

            # Selling logic (only on Mondays)
            elif current_date.weekday() == 0 and invest_percent < 0 and holdings > 0 and last_exchange_rate is not None:
                sell_percent = -invest_percent  # Convert to positive value
                btc_to_sell = holdings * sell_percent
                amount_received = btc_to_sell * last_exchange_rate
                holdings -= btc_to_sell
                depot += amount_received
                if print_logs:
                    print(f"{current_date}: Sold {btc_to_sell:.6f} BTC for {amount_received:.2f} EUR at risk {risk}%")

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

        if print_logs:
            print(">>> Final Portfolio:")
            final_portfolio = portfolio_values[-1]
            print(f"Date: {final_portfolio['date']}")
            print(f"Depot: {final_portfolio['depot']:.2f} EUR")
            print(f"Holdings: {final_portfolio['holdings']:.6f} BTC")
            print(f"Total Value: {final_portfolio['total_value']:.2f} EUR")

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
    #TestAlphaSquaredStrategy.run()
    TestAlphaSquaredStrategy.run_monthly_comparisons()

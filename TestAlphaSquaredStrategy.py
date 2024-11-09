from datetime import datetime

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
            start_filter = datetime(2016, 4, 1)
            end_filter = datetime(2023, 8, 24)

            print(f"Loading AlphaSquared data from {start_filter} to {end_filter}...")
            alphasq_data = AlphaSquaredRepo.AlphaSquaredRepo.get_risk_entries(start_filter, end_filter, "repos/AlphaSquaredRiskData_BTC.csv")
            print(f"Loaded {len(alphasq_data)} AlphaSquared entries.")

            print(f"Loading BTC/EUR data from {start_filter} to {end_filter}...")
            dictlist_btc = CoinBaseRepo.CoinBaseRepo.read_csv_to_dict('repos/BTC_EUR.csv')
            sample_exchange_rates = CoinBaseRepo.CoinBaseRepo.get_exchange_rate_items(start_filter, end_filter, dictlist_btc)
            print(f"Loaded {len(sample_exchange_rates)} BTC/EUR entries.")

            # Process or analyze the data as needed
            print("Processing data...")
            # Add your processing logic here

        except Exception as e:
            print(f"Error: {e}")


# Main execution
if __name__ == "__main__":
    print("run")
    TestAlphaSquaredStrategy.run()

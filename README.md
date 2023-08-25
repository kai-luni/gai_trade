

# gai_trade
A collection of Crypto trading algorithms in the search of a reasonable trading strategy.

## Testing the "Buy Below Moving Average" Strategy

In the quest for profitable Bitcoin trading strategies, one simple approach is often proposed: buy when the current price is below its moving average. The idea is that if the price is below its average, it's a good buy opportunity since the price might revert to the mean.

To test this hypothesis, we've set up an experiment in the file `test_buy_below_moving_average.py`.

### Experiment Details

1. **Data Source**: The experiment leverages historical Bitcoin exchange rate data fetched from `CoinBaseRepo`, spanning from April 1, 2016, to August 24, 2023.

2. **Investment Strategy**:
    - Every first day of the month, we simulate an investment of 100 Euros.
    - We calculate the moving average of the closing prices of Bitcoin using varying window lengths.
    - If the closing price of Bitcoin for a particular day is below its moving average and we have available balance, we buy Bitcoin with the entire balance.
    - This process is repeated for every day in our dataset.

3. **Window Lengths**: The experiment was run for several window lengths to see how the duration of the moving average affects the outcome. The window lengths tested were: 1, 2, 5, 10, 20, 40, and 80 days.

4. **Output Metrics**:
    - Total Bitcoins purchased.
    - Remaining balance in Euros.
    - Total amount of Euros spent.
    - The value of total Bitcoins purchased at the last known exchange rate.

### Observations

By running the script, we can observe the outcome of our strategy for different moving average window lengths. The main metric of interest is the value of total Bitcoins purchased at the last known exchange rate. This gives an idea of how well our strategy performed compared to simply holding the invested Euros.

### Conclusion

Based on the results of our experiment, it appears that simply buying below the moving average does not consistently yield better results than other strategies or holding. The efficacy of the strategy varies based on the chosen window length for the moving average and the overall market conditions.

It's essential to keep in mind that historical performance is not indicative of future results. While this experiment provides insights into the past behavior of the "buy below moving average" strategy, it doesn't guarantee future profitability.

Furthermore, the "buy below moving average" strategy is a simplistic approach. In real-world scenarios, traders and algorithms use a combination of indicators, risk management techniques, and market insights to make informed decisions.

---

Note: This README provides an overview of the experiment and its outcomes. For detailed results and to run the experiment yourself, please refer to the `test_buy_below_moving_average.py` file.









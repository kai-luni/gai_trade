import os
from typing import Callable
import warnings
import argparse
import sys
import time
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from openai import AzureOpenAI

from repos import CoinBaseRepo
from dto.ExchangeRateItem import ExchangeRateItem

console = Console()


# Strategy: Trend Following - Further refined thresholds, decision logic, and responsiveness to market conditions
# This strategy identifies strong trends using moving averages, volatility, volume, and RSI, while filtering signals effectively.
# Improvements include dynamic thresholds, better edge case handling, and combining signals more intelligently.

def daily_decision(exchange_rates: list[ExchangeRateItem]) -> int:
    # Validate input length
    if len(exchange_rates) < 30:
        raise ValueError(f"Insufficient data: {len(exchange_rates)} days provided, at least 30 required.")
    
    # Extract closing prices, volumes, highs, and lows
    closing_prices = [er.close for er in exchange_rates]
    volumes = [er.volume for er in exchange_rates]
    highs = [er.high for er in exchange_rates]
    lows = [er.low for er in exchange_rates]
    
    # Calculate moving averages
    ma_3 = sum(closing_prices[-3:]) / 3
    ma_30 = sum(closing_prices[-30:]) / 30
    
    # Calculate volatility for the last 3 days
    volatility_3_days = sum([(highs[-i] - lows[-i]) / closing_prices[-i] for i in range(1, 4)]) / 3
    
    # Calculate today's volume compared to the 30-day average
    avg_volume_30 = sum(volumes[-30:]) / 30
    volume_ratio = volumes[-1] / avg_volume_30 if avg_volume_30 != 0 else 0
    
    # Calculate RSI (Relative Strength Index) using the 14-day period
    gains = [max(0, closing_prices[i] - closing_prices[i - 1]) for i in range(-14, 0)]
    losses = [max(0, closing_prices[i - 1] - closing_prices[i]) for i in range(-14, 0)]
    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14
    rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
    rsi = 100 - (100 / (1 + rs))
    
    # Further refined thresholds and decision logic
    # Buy signal: Strong upward trend, confirmed by volatility, volume, and RSI in a favorable range
    if ma_3 > ma_30 * 1.01 and volatility_3_days > 0.025 and volume_ratio > 1.3 and 40 < rsi < 65:
        return 1
    # Sell signal: Strong downward trend, confirmed by volatility, volume, and RSI in a favorable range
    elif ma_3 < ma_30 * 0.99 and volatility_3_days > 0.025 and volume_ratio > 1.3 and rsi > 55:
        return 2
    # Hold signal: No clear trend or conflicting indicators
    else:
        return 0
    

def simulate_trading(decision_func : Callable[[list[ExchangeRateItem]], int], exchange_rates: list[ExchangeRateItem], debug: bool = False) -> tuple[float, list]:
    """
    Simulates a BTC trading strategy from day 30 onward using historical data.

    Args:
        decision_func: Function that analyzes historical data and returns:
                       1 = BUY (use all cash), 2 = SELL (all BTC), 0 = HOLD
        exchange_rates: Historical BTC price data (requires >=30 entries minimum)
        debug: Show trading actions in console when enabled

    Returns:
        Tuple containing:
        - float: Final portfolio value (cash + BTC at last day's close price)
        - list: Daily portfolio values (in EUR) for all trading days

    Behavior:
        - Starts trading at index 30 (first 30 entries are history for initial analysis)
        - Full position sizes: Either 100% cash or 100% BTC (no partial positions)
        - Errors trigger HOLD action and continue simulation
    """
    portfolio = {
        'cash': 1000.00,  # Initial capital in EUR
        'btc': 0.0,       # BTC holdings
        'history': []     # Daily portfolio value in EUR
    }
    
    for i in range(30, len(exchange_rates)):  # Start from day 30 (min required data)
        try:
            # Get data up to current day
            current_data = exchange_rates[:i+1]
            today = current_data[-1]
            
            # Get decision for today
            decision = decision_func(current_data)
            
            # Execute trade based on decision
            if decision == 1 and portfolio['cash'] > 0:
                console.print(f"[green]BUY signal on {today.date} at price {today.close:.2f}[/green]")
                portfolio['btc'] = portfolio['cash'] / today.close
                portfolio['cash'] = 0.0
            elif decision == 2 and portfolio['btc'] > 0:
                console.print(f"[red]SELL signal on {today.date} at price {today.close:.2f}[/red]")
                portfolio['cash'] = portfolio['btc'] * today.close
                portfolio['btc'] = 0.0
            # else:
            #     if debug:
            #         console.print(f"[yellow]HOLD on {today.date}[/yellow]")

            # Debug output for today's decision and portfolio status
            if debug and (decision == 1 or decision == 2):
                console.print(f"[blue]Day {today.date}:[/blue] Decision = {decision}, Price = {today.close:.2f}, Cash = €{portfolio['cash']:.2f}, BTC Cash = {(portfolio['btc'] * today.close):.2f}")
            
            # Record portfolio value (cash + BTC value)
            current_value = portfolio['cash'] + (portfolio['btc'] * today.close)
            portfolio['history'].append(current_value)
            
        except Exception as e:
            if debug:
                console.print(f"[red]Error on {today.date}: {str(e)}[/red]")
            # Treat errors as hold action
            current_value = portfolio['cash'] + (portfolio['btc'] * today.close)
            portfolio['history'].append(current_value)
    
    return portfolio['history'][-1], portfolio['history']

def calculate_buy_and_hold(exchange_rates: list[ExchangeRateItem]) -> float:
    """
    Calculate the final value if 1000 EUR was invested on the first day and never touched.
    
    Args:
        exchange_rates: Historical BTC price data
        
    Returns:
        float: Final value of the buy-and-hold strategy
    """
    if len(exchange_rates) < 31:  # Need at least 31 days (30 days + 1)
        raise ValueError("Insufficient data for buy-and-hold comparison")
    
    # Buy BTC on day 30 (same starting point as the trading simulation)
    initial_price = exchange_rates[30].close
    btc_amount = 1000.00 / initial_price
    
    # Calculate final value based on the last day's price
    final_price = exchange_rates[-1].close
    final_value = btc_amount * final_price
    
    return final_value

def main():
    """Load and validate BTC/EUR historical data from CSV"""
    csv_path = 'repos/BTC_EUR.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    console.print(f"\n📂 Loading historical data from {csv_path}", style="bold blue")

    # Load raw CSV data
    raw_data = CoinBaseRepo.CoinBaseRepo.read_csv_to_dict(csv_path)

    # Convert to ExchangeRateItem objects
    all_items = CoinBaseRepo.CoinBaseRepo.get_exchange_rate_items(
        datetime(2018, 1, 1),  # Broad initial filter
        datetime(2024, 9, 1), 
        raw_data
    )

    # Run the trading simulation
    console.print("\n[bold]Running trading strategy simulation...[/bold]")
    final_value, value_history = simulate_trading(daily_decision, all_items, debug=True)
    
    # Calculate buy-and-hold strategy
    console.print("\n[bold]Calculating buy-and-hold strategy...[/bold]")
    buy_hold_value = calculate_buy_and_hold(all_items)
    
    # Calculate performance metrics
    trading_return_pct = ((final_value / 1000.0) - 1) * 100
    buy_hold_return_pct = ((buy_hold_value / 1000.0) - 1) * 100
    performance_diff = trading_return_pct - buy_hold_return_pct
    
    # Display comparison
    table = Table(title="Investment Strategy Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Initial Investment", style="green")
    table.add_column("Final Value", style="green")
    table.add_column("Return %", style="yellow")
    
    table.add_row(
        "Trading Algorithm", 
        "€1,000.00", 
        f"€{final_value:,.2f}", 
        f"{trading_return_pct:+,.2f}%"
    )
    table.add_row(
        "Buy and Hold", 
        "€1,000.00", 
        f"€{buy_hold_value:,.2f}", 
        f"{buy_hold_return_pct:+,.2f}%"
    )
    
    console.print("\n")
    console.print(table)
    
    # Display which strategy performed better
    performance_message = ""
    if final_value > buy_hold_value:
        performance_message = f"[bold green]Trading algorithm outperformed buy-and-hold by {performance_diff:+,.2f}%[/bold green]"
    elif final_value < buy_hold_value:
        performance_message = f"[bold red]Trading algorithm underperformed buy-and-hold by {abs(performance_diff):,.2f}%[/bold red]"
    else:
        performance_message = "[bold yellow]Trading algorithm performed exactly the same as buy-and-hold[/bold yellow]"
    
    console.print("\n" + performance_message)

if __name__ == "__main__":
    main()
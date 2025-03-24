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

from dto.NetflowDto import NetflowDto
from dto.ObjectsGai import FearGreedItem
from repos import CoinBaseRepo
from dto.ExchangeRateItem import ExchangeRateItem
from repos.FearGreedRepo import FearGreedRepo
from repos.NetflowRepo import NetflowRepo

console = Console()


# Strategy: Trend Following with Multi-Confirmation - Enhanced thresholds and momentum confirmation for better responsiveness
def daily_decision(exchange_rates: list[ExchangeRateItem]) -> int:
    if len(exchange_rates) < 30:
        raise ValueError(f"Insufficient data: {len(exchange_rates)} days provided, need 30")
    
    closing_prices = [er.close for er in exchange_rates]
    ma_3 = sum(closing_prices[-3:])/3
    ma_30 = sum(closing_prices[-30:])/30
    today = exchange_rates[-1]
    yesterday = exchange_rates[-2]
    
    # Enhanced volatility calculation (3-day average)
    volatility = [((er.high - er.low)/er.close) if er.close != 0 else 0.0 for er in exchange_rates[-3:]]
    avg_volatility = sum(volatility)/3
    
    # Dynamic volume analysis with momentum check
    volume_avg = sum(er.volume for er in exchange_rates[-30:])/30
    volume_ratio = today.volume/volume_avg if volume_avg != 0 else 0
    volume_increasing = today.volume > yesterday.volume
    
    # Corrected RSI calculation with 14-day period
    rsi_closes = closing_prices[-15:]  # Last 15 days including today
    deltas = [rsi_closes[i] - rsi_closes[i-1] for i in range(1, len(rsi_closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_gain = sum(gains)/14 if len(gains) else 0
    avg_loss = sum(losses)/14 if len(losses) else 1
    
    rs = avg_gain/(avg_loss or 1)  # Handle zero division
    rsi = 100 - (100/(1 + rs)) if avg_loss != 0 else 100
    
    # Momentum confirmation metrics
    prev_ma_3 = sum(closing_prices[-4:-1])/3  # Previous 3-day MA
    ma_3_trend = ma_3 > prev_ma_3
    
    # Trend confirmation logic with multiple filters
    price_above = today.close > ma_3 and today.close > ma_30
    price_above_prev = yesterday.close > ma_3 and yesterday.close > ma_30
    
    buy_signal = (
        ma_3 > ma_30 and
        price_above and
        price_above_prev and
        rsi < 60 and
        volume_ratio >= 1.1 and
        volume_increasing and
        avg_volatility > 0.015 and
        ma_3_trend
    )
    
    # Sell conditions with inverse logic
    price_below = today.close < ma_3 and today.close < ma_30
    price_below_prev = yesterday.close < ma_3 and yesterday.close < ma_30
    ma_3_trend_down = ma_3 < prev_ma_3
    
    sell_signal = (
        ma_3 < ma_30 and
        price_below and
        price_below_prev and
        rsi > 40 and
        volume_ratio >= 1.1 and
        avg_volatility > 0.015 and
        ma_3_trend_down
    )
    
    return 1 if buy_signal else 2 if sell_signal else 0
    

def simulate_trading(decision_func: Callable[[list[ExchangeRateItem], list[FearGreedItem], list[NetflowDto]], int], 
                    exchange_rates: list[ExchangeRateItem], 
                    fear_greed_data: list[FearGreedItem] = None,
                    netflow_data: list[NetflowDto] = None,
                    debug: bool = False) -> tuple[float, list, list]:
    """
    Simulates a BTC trading strategy from day 30 onward using historical data.

    Args:
        decision_func: Function that analyzes historical data and returns:
                       1 = BUY (use all cash), 2 = SELL (all BTC), 0 = HOLD
        exchange_rates: Historical BTC price data (requires >=30 entries minimum)
        fear_greed_data: Historical Fear & Greed Index data (optional)
        netflow_data: Historical Bitcoin exchange flow data (optional)
        debug: Show trading actions in console when enabled

    Returns:
        Tuple containing:
        - float: Final portfolio value (cash + BTC at last day's close price)
        - list: Daily portfolio values (in EUR) for all trading days
        - list: Errors encountered during simulation
    """
    portfolio = {
        'cash': 0.00,  # Initial capital in EUR
        'btc': 0.1,    # BTC holdings
        'history': []  # Daily portfolio value in EUR
    }
    
    # Collect simulation errors
    errors = []
    
    # Prepare aligned fear & greed data if available
    aligned_fear_greed = None
    if fear_greed_data:
        # Create a date-indexed dictionary for faster lookup
        fear_greed_dict = {item.date.date(): item for item in fear_greed_data}
        
        # We'll build aligned data as we go through the simulation
        aligned_fear_greed = []
    
    # Prepare aligned netflow data if available
    aligned_netflow = None
    if netflow_data:
        # Create a date-indexed dictionary for faster lookup
        netflow_dict = {item.date_time.date(): item for item in netflow_data}
        
        # We'll build aligned data as we go through the simulation
        aligned_netflow = []
    
    for i in range(30, len(exchange_rates)):  # Start from day 30 (min required data)
        try:
            # Get data up to current day
            current_data = exchange_rates[:i+1]
            today = current_data[-1]
            
            # Add 100 euros on the first day of each month
            if today.date.day == 1:
                portfolio['btc'] += (100.00 / today.close)
                if debug:
                    console.print(f"[cyan]First day of month ({today.date}): Added €100.00 to portfolio[/cyan]")
            
            # Prepare aligned fear & greed data for the current time window if available
            current_fear_greed = None
            if fear_greed_data:
                aligned_fear_greed = []
                for er in current_data:
                    # Look up the fear & greed item for this date
                    fg_item = fear_greed_dict.get(er.date.date())
                    if fg_item:
                        aligned_fear_greed.append(fg_item)
                
                # If we have aligned data, pass it to the decision function
                if aligned_fear_greed:
                    current_fear_greed = aligned_fear_greed
            
            # Prepare aligned netflow data for the current time window if available
            current_netflow = None
            if netflow_data:
                aligned_netflow = []
                for er in current_data:
                    # Look up the netflow item for this date
                    nf_item = netflow_dict.get(er.date.date())
                    if nf_item:
                        aligned_netflow.append(nf_item)
                
                # If we have aligned data, pass it to the decision function
                if aligned_netflow:
                    current_netflow = aligned_netflow
            
            # Get decision for today
            decision = decision_func(current_data)
            
            # Execute trade based on decision
            if decision == 1 and portfolio['cash'] > 0:
                console.print(f"[green]BUY signal on {today.date} at price {today.close:.2f}[/green]")
                portfolio['btc'] = portfolio['btc'] + (portfolio['cash'] / today.close)
                portfolio['cash'] = 0.0
            elif decision == 2 and portfolio['btc'] > 0:
                console.print(f"[red]SELL signal on {today.date} at price {today.close:.2f}[/red]")
                portfolio['cash'] = portfolio['cash'] + (portfolio['btc'] * today.close)
                portfolio['btc'] = 0.0
            
            # Debug output for today's decision and portfolio status
            if debug and (decision == 1 or decision == 2):
                fg_info = ""
                if current_fear_greed and len(current_fear_greed) > 0:
                    try:
                        today_fg = next((fg for fg in current_fear_greed if fg.date.date() == today.date.date()), None)
                        if today_fg:
                            fg_info = f", F&G Index: {today_fg.index} ({today_fg.index_text})"
                    except:
                        pass
                
                nf_info = ""
                if current_netflow and len(current_netflow) > 0:
                    try:
                        today_nf = next((nf for nf in current_netflow if nf.date_time.date() == today.date.date()), None)
                        if today_nf and today_nf.aggregated_exchanges is not None:
                            flow_dir = "IN" if today_nf.aggregated_exchanges > 0 else "OUT"
                            nf_info = f", Netflow: {abs(today_nf.aggregated_exchanges):.2f} BTC {flow_dir}"
                    except:
                        pass
                
                console.print(f"[blue]Day {today.date}:[/blue] Decision = {decision}, Price = {today.close:.2f}, Cash = €{portfolio['cash']:.2f}, BTC Cash = {(portfolio['btc'] * today.close):.2f}{fg_info}{nf_info}")
            
            # Record portfolio value (cash + BTC value)
            current_value = portfolio['cash'] + (portfolio['btc'] * today.close)
            portfolio['history'].append(current_value)
            
        except Exception as e:
            error_msg = f"Error on {today.date}: {str(e)}"
            if debug:
                console.print(f"[red]{error_msg}[/red]")
            
            # Add error to collection with date context
            errors.append(error_msg)
            
            current_value = portfolio['cash'] + (portfolio['btc'] * today.close)
            portfolio['history'].append(current_value)

    
    return portfolio['history'][-1], portfolio['history'], errors

def buy_and_hold_decision(exchange_rates, fear_greed_data, netflow_data):
    """
    Decision function that always returns 0 (HOLD) to simulate a buy-and-hold strategy.
    This will ensure BTC is kept but still allows the monthly deposits to occur.
    """
    return 0

def calculate_buy_and_hold(exchange_rates, fear_greed_data=None, netflow_data=None):
    """
    Calculate the final value if we follow a buy-and-hold strategy,
    including monthly deposits matching the simulation parameters.
    
    Args:
        exchange_rates: Historical BTC price data
        fear_greed_data: Historical Fear & Greed Index data (optional)
        netflow_data: Historical Bitcoin exchange flow data (optional)
        
    Returns:
        float: Final value of the buy-and-hold strategy
    """
    # Use the simulate_trading function with our buy-and-hold decision function
    final_value, history, errors = simulate_trading(
        buy_and_hold_decision, 
        exchange_rates, 
        fear_greed_data, 
        netflow_data,
        debug=False  # Set to True if you want to see the monthly deposits
    )
    
    return final_value

def main():
    """Load and validate BTC/EUR historical data from CSV"""
    csv_path = 'repos/BTC_EUR.csv'
    console.print(f"\n📂 Loading historical data from {csv_path}", style="bold blue")
    
    # Define a common start and end date for all data sources
    start_date = datetime(2018, 2, 4)
    end_date = datetime(2025, 1, 1)  # Using the latest common date from your data
    
    console.print(f"\n[bold]Data range: {start_date.date()} to {end_date.date()}[/bold]")
    
    # Load raw CSV data
    raw_data = CoinBaseRepo.CoinBaseRepo.read_csv_to_dict(csv_path)
    
    # Convert to ExchangeRateItem objects
    all_items = CoinBaseRepo.CoinBaseRepo.get_exchange_rate_items(
        start_date,
        end_date,
        raw_data
    )
    
    # Load Fear & Greed data
    fear_greed_items = FearGreedRepo.read_csv_file(start_date, end_date)
    
    # Load Netflow data
    netflow_repo = NetflowRepo("repos/ITB_btc_netflows.csv")
    netflow_data = netflow_repo.get_range(start_date.date(), end_date.date())
    
    # Validate data lengths
    validate_data_length(all_items, fear_greed_items, netflow_data)
    
    # Run the trading simulation
    console.print("\n[bold]Running trading strategy simulation...[/bold]")
    final_value, _, _ = simulate_trading(daily_decision, all_items, fear_greed_items, netflow_data, debug=True)
    
    # Calculate buy-and-hold strategy
    console.print("\n[bold]Calculating buy-and-hold strategy...[/bold]")
    buy_hold_value = calculate_buy_and_hold(all_items, fear_greed_items, netflow_data)
    
    # Display comparison
    table = Table(title="Investment Strategy Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Final Value", style="green")
    table.add_row(
        "Trading Algorithm",
        f"€{final_value:,.2f}"
    )
    table.add_row(
        "Buy and Hold",
        f"€{buy_hold_value:,.2f}"
    )
    console.print("\n")
    console.print(table)
    
    # Display which strategy performed better
    performance_diff_absolute = final_value - buy_hold_value
    performance_diff_percent = (performance_diff_absolute / buy_hold_value) * 100
    performance_message = ""
    if final_value > buy_hold_value:
        performance_message = f"[bold green]Trading algorithm outperformed buy-and-hold by €{performance_diff_absolute:,.2f} ({performance_diff_percent:+,.2f}%)[/bold green]"
    elif final_value < buy_hold_value:
        performance_message = f"[bold red]Trading algorithm underperformed buy-and-hold by €{abs(performance_diff_absolute):,.2f} ({performance_diff_percent:+,.2f}%)[/bold red]"
    else:
        performance_message = "[bold yellow]Trading algorithm performed exactly the same as buy-and-hold[/bold yellow]"
    console.print("\n" + performance_message)


def validate_data_length(exchange_items, fear_greed_items, netflow_items):
    """
    Validate that all three data sources have the same length.
    Throws an exception if there's a mismatch.
    
    Args:
        exchange_items: List of exchange rate items
        fear_greed_items: List of fear and greed index items
        netflow_items: List of netflow data items
    
    Raises:
        ValueError: If the data sources have different lengths
    """
    exchange_length = len(exchange_items)
    fear_greed_length = len(fear_greed_items)
    netflow_length = len(netflow_items)
    
    console.print(f"\n[bold]Data validation:[/bold]")
    console.print(f"Exchange rate data points: {exchange_length}")
    console.print(f"Fear & Greed data points: {fear_greed_length}")
    console.print(f"Netflow data points: {netflow_length}")
    
    if not (exchange_length == fear_greed_length == netflow_length):
        error_message = (
            f"\n[bold red]ERROR: Data length mismatch![/bold red]\n"
            f"Exchange rate: {exchange_length} days\n"
            f"Fear & Greed: {fear_greed_length} days\n"
            f"Netflow: {netflow_length} days"
        )
        console.print(error_message)
        raise ValueError("Data sources have different lengths. Please ensure all data sources cover the same date range with no missing values.")
    else:
        console.print(f"[bold green]✓ All data sources have the same length: {exchange_length} days[/bold green]")

if __name__ == "__main__":
    main()
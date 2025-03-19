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
from repos import CoinBaseRepo, FearGreedRepo
from dto.ExchangeRateItem import ExchangeRateItem

console = Console()


# Strategy: Enhanced Breakout with SMA & Sentiment Filters - 18-day breakout levels with 1.3x volume, 5D>15D volatility (min 1.5%), RSI 65/35, 20-day SMA trend alignment, and Fear/Greed sentiment filter
def daily_decision(exchange_rates: list[ExchangeRateItem], fear_greed_data: list[FearGreedItem]) -> int:
    if len(exchange_rates) < 30:
        raise ValueError(f"Need at least 30 days, got {len(exchange_rates)}")
    
    # Data extraction
    closing_prices = [er.close for er in exchange_rates]
    highs = [er.high for er in exchange_rates]
    lows = [er.low for er in exchange_rates]
    volumes = [er.volume for er in exchange_rates]
    today = exchange_rates[-1]
    
    # Trend and breakout levels
    resistance = max(highs[-19:-1]) if len(highs) >= 19 else 0  # 18-day resistance
    support = min(lows[-19:-1]) if len(lows) >= 19 else 0       # 18-day support
    sma_20 = sum(closing_prices[-20:])/20 if len(closing_prices) >= 20 else 0
    
    # Volume analysis (5-day average)
    volume_5_avg = sum(volumes[-6:-1])/5 if len(volumes) >=6 else 0
    volume_trigger = today.volume > 1.3 * volume_5_avg
    
    # Volatility analysis
    vol_period = 7  # Compare 5-day vs 15-day volatility
    vol_short = sum((er.high - er.low)/er.close for er in exchange_rates[-5:])/5 if len(exchange_rates)>=5 else 0
    vol_long = sum((er.high - er.low)/er.close for er in exchange_rates[-20:-5])/15 if len(exchange_rates)>=20 else 0
    volatility_rising = (vol_short > 1.25 * vol_long) and (vol_short > 0.015)
    
    # Momentum analysis (RSI 14)
    rsi = 50
    if len(closing_prices) >= 15:
        changes = [closing_prices[i] - closing_prices[i-1] for i in range(-14, 0)]
        gains = [max(c, 0) for c in changes]
        losses = [max(-c, 0) for c in changes]
        avg_gain = sum(gains)/14
        avg_loss = sum(losses)/14
        if avg_loss == 0:
            rsi = 100 if avg_gain != 0 else 50
        else:
            rs = avg_gain/(avg_loss + 1e-10)
            rsi = 100 - (100/(1 + rs))
    
    # Sentiment analysis
    fg_ok_buy = True
    fg_ok_sell = True
    if fear_greed_data:
        latest_fg = fear_greed_data[-1].index_text
        fg_ok_buy = latest_fg not in ['Extreme Fear']
        fg_ok_sell = latest_fg not in ['Extreme Greed']
    
    # Decision logic with multiple confirmations
    buy_signal = (today.close > resistance * 1.007 and  # 0.7% buffer
                 volume_trigger and
                 volatility_rising and
                 rsi < 65 and
                 today.close > sma_20 and
                 fg_ok_buy)
    
    sell_signal = (today.close < support * 0.993 and  # 0.7% buffer
                  volume_trigger and
                  volatility_rising and
                  rsi > 35 and
                  today.close < sma_20 and
                  fg_ok_sell)
    
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
            decision = decision_func(current_data, aligned_fear_greed)
            
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

    # Load raw CSV data
    raw_data = CoinBaseRepo.CoinBaseRepo.read_csv_to_dict(csv_path)

    # Convert to ExchangeRateItem objects
    all_items = CoinBaseRepo.CoinBaseRepo.get_exchange_rate_items(
        datetime(2018, 1, 1),  # Broad initial filter
        datetime(2025, 3, 5), 
        raw_data
    )
    
    fear_greed_items = FearGreedRepo.FearGreedRepo.read_csv_file(datetime(2018, 1, 1), datetime(2024, 5, 3))

    # Run the trading simulation
    console.print("\n[bold]Running trading strategy simulation...[/bold]")
    final_value, _, _ = simulate_trading(daily_decision, all_items, fear_greed_items, debug=True)
    
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
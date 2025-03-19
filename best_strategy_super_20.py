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
from repos.NetflowRepo import NetflowRepo

console = Console()


# Strategy: Weighted Sentiment Momentum v2 - Adjusted thresholds, enhanced trend confirmation, stricter volume checks, RSI momentum, and stronger 1st-day weighting. Requires 4 points for buy (-6 sell) with 4-point boost on allocations.  
def daily_decision(exchange_rates: list[ExchangeRateItem], fear_greed_data: list[FearGreedItem], netflow_data: list[NetflowDto]) -> int:
    if len(exchange_rates) < 30:
        raise ValueError(f"Need at least 30 days of exchange rates, got {len(exchange_rates)}")
    
    # Data extraction
    closing_prices = [er.close for er in exchange_rates]
    high_prices = [er.high for er in exchange_rates]
    low_prices = [er.low for er in exchange_rates]
    volumes = [er.volume for er in exchange_rates]
    fear_greed_indices = [fg.index for fg in fear_greed_data]
    netflows = [nf.aggregated_exchanges for nf in netflow_data if nf.aggregated_exchanges is not None]
    
    # Technical indicators
    def calculate_rsi(prices, period=14):
        if len(prices) < period+1: return []
        deltas = [prices[i]-prices[i-1] for i in range(1, len(prices))]
        gains = [max(d,0) for d in deltas]
        losses = [max(-d,0) for d in deltas]
        avg_gain = sum(gains[:period])/period
        avg_loss = sum(losses[:period])/period if sum(losses[:period])>0 else 1
        rs = avg_gain/avg_loss
        rsi = [100 - 100/(1+rs)]
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain*(period-1) + gains[i])/period
            avg_loss = (avg_loss*(period-1) + losses[i])/period if (avg_loss*(period-1)+losses[i])>0 else 1
            rs = avg_gain/avg_loss
            rsi.append(100 - 100/(1+rs))
        return rsi
    
    rsi_values = calculate_rsi(closing_prices)
    current_rsi = rsi_values[-1] if rsi_values else 50
    rsi_trend = rsi_values[-1] > rsi_values[-2] if len(rsi_values)>=2 else False
    
    # Momentum calculations
    roc_3 = ((closing_prices[-1]/closing_prices[-4])-1)*100 if len(closing_prices)>=4 and closing_prices[-4]!=0 else 0
    roc_7 = ((closing_prices[-1]/closing_prices[-8])-1)*100 if len(closing_prices)>=8 and closing_prices[-8]!=0 else 0
    
    # Volatility metrics
    volatility = [(high_prices[i]-low_prices[i])/closing_prices[i] if closing_prices[i]!=0 else 0 for i in range(len(closing_prices))]
    vol_3_avg = sum(volatility[-3:])/3 if len(volatility)>=3 else 0
    vol_10_avg = sum(volatility[-10:])/10 if len(volatility)>=10 else 0
    
    # Netflow analysis
    sum_last_3 = sum(netflows[-3:]) if len(netflows)>=3 else 0
    sum_prev_3 = sum(netflows[-6:-3]) if len(netflows)>=6 else 0
    
    # Fear & Greed trend (3-day upward trend)
    fg_trend = (len(fear_greed_indices)>=3 and 
                fear_greed_indices[-1] > fear_greed_indices[-2] and 
                fear_greed_indices[-2] > fear_greed_indices[-3])
    current_fg = fear_greed_data[-1]
    
    # Volume confirmation
    volume_10_avg = sum(volumes[-10:])/10 if len(volumes)>=10 else 0
    current_volume = volumes[-1] if volumes else 0
    price_trend_3d = closing_prices[-1] > closing_prices[-2] > closing_prices[-3] if len(closing_prices)>=3 else False
    
    # Scoring system
    score = 0
    
    # Momentum (max +3)
    if roc_3 > 1.5 and roc_7 > 3: score += 3
    elif roc_3 < -1.5 and roc_7 < -3: score -= 3
    
    # Volatility contraction (max ±2)
    if vol_3_avg < 0.35 * vol_10_avg and vol_10_avg != 0: score += 2
    elif vol_3_avg > 1.8 * vol_10_avg and vol_10_avg != 0: score -= 2
    
    # Netflow acceleration (max ±2)
    if sum_last_3 < (0.6 * sum_prev_3) and sum_prev_3 != 0: score += 2
    elif sum_last_3 > (1.4 * sum_prev_3) and sum_prev_3 != 0: score -= 2
    
    # F&G + trend (max +3)
    if current_fg.index < 25 and fg_trend and current_fg.index_text == 'Extreme Fear': score += 3
    elif current_fg.index > 75 and not fg_trend and current_fg.index_text == 'Extreme Greed': score -= 2
    
    # Volume confirmation (max ±2)
    if current_volume > 2 * volume_10_avg:
        if price_trend_3d: score += 2
        else: score -= 1
    
    # RSI conditions with trend (max ±2)
    if current_rsi < 30 and rsi_trend: score += 2
    elif current_rsi > 70 and not rsi_trend: score -= 2
    
    # Monthly allocation boost (stronger weighting)
    if exchange_rates[-1].date.day == 1:
        score += 4
    
    # Decision logic with stricter sell requirements
    if score >= 4:
        return 1
    elif score <= -6:
        return 2
    else:
        return 0
    

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
            decision = decision_func(current_data, current_fear_greed, current_netflow)
            
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

    # Load Netflow data
    netflow_repo = NetflowRepo("repos/ITB_btc_netflows.csv")  # Adjust path as needed
    netflow_data = netflow_repo.get_range(datetime(2018, 1, 1).date(), datetime(2024, 9, 1).date())

    # Run the trading simulation
    console.print("\n[bold]Running trading strategy simulation...[/bold]")
    final_value, _, _ = simulate_trading(daily_decision, all_items, fear_greed_items, netflow_data, debug=True)
    
    # Calculate buy-and-hold strategy using the improved method
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

if __name__ == "__main__":
    main()
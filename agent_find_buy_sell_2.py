# Load environment variables from .env
from typing import Callable, List
from dotenv import load_dotenv
import os

from dto.ExchangeRateItem import ExchangeRateItem
load_dotenv()

# Check Azure OpenAI credentials
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
    raise ValueError("Missing Azure OpenAI credentials in .env file")

# -----------------------------
# 1. Imports & Configuration
# -----------------------------
import warnings
import argparse
import sys
import time
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from openai import AzureOpenAI
from repos import CoinBaseRepo

warnings.simplefilter("ignore", DeprecationWarning)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-12-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
model_name = "gpt-4o"
console = Console()

# -----------------------------
# 2. Core Domain Objects
# -----------------------------

# -----------------------------
# 3. Data Preparation
# -----------------------------
def load_historical_data() -> list[ExchangeRateItem]:
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
    
    # Filter and sort data
    filtered_items = [
        item for item in all_items
        if datetime(2018, 12, 31) <= item.date <= datetime(2024, 1, 7)
    ]
    filtered_items.sort(key=lambda x: x.date)
    
    # Data integrity checks
    if len(filtered_items) < 365:
        raise ValueError("Insufficient historical data (less than 1 year)")
    
    console.print(f"✅ Successfully loaded {len(filtered_items)} trading days")
    console.print(f"⏳ Date range: {filtered_items[0].date:%Y-%m-%d} → {filtered_items[-1].date:%Y-%m-%d}")
    
    return filtered_items

# -----------------------------
# 4. LLM Strategy Generation
# -----------------------------

def generate_trading_strategy(feedback: str = "", attempt_num: int = 1, max_attempts: int = 3, debug: bool = False) -> str:
    """Generate daily trading decision function with emphasis on refinement, occasional exploration"""
    
    # Import random for strategy mode selection
    import random
    
    # Logging file setup
    log_file = 'log.txt'
    with open(log_file, 'a') as log:
        log.write("\n" + "=" * 40 + "\n")
        log.write(f"Execution Date: {datetime.now()}\n")
    
    # Base prompt with diversification guidance
    base_prompt = (
        "Write a SINGLE Python function named 'daily_decision' using this EXACT signature:\n"
        "def daily_decision(exchange_rates: list[ExchangeRateItem]) -> int:\n\n"
        "=== INPUT SPECIFICATIONS ===\n"
        "1. ExchangeRateItem PREDEFINED CLASS (DO NOT REDECLARE) with fields:\n"
        " - date: datetime (access via .date)\n"
        " - close: float (closing price EUR)\n"
        " - high: float (daily high EUR)\n"
        " - low: float (daily low EUR)\n"
        " - open: float (opening price EUR)\n"
        " - volume: float (BTC traded)\n"
        "2. Input list is chronological - exchange_rates[-1] is today\n\n"
        "=== OUTPUT REQUIREMENTS ===\n"
        "Return ONLY 0/1/2 integer. NO OTHER OUTPUT:\n"
        "- 0: Hold | 1: Buy today | 2: Sell today\n\n"
        "=== STRATEGY RULES ===\n"
        "1. START with '# Strategy: [Strategy Type] - ' comment explaining your approach\n"
        "2. Required technical factors:\n"
        " a) 3-day vs 30-day moving averages of closing_prices\n"
        " b) Volatility: (high - low)/close for last 3 days\n"
        " c) Volume analysis: Compare today's volume to 30-day avg\n"
        " d) RSI calculation using 14-day period\n"
        "3. Edge cases:\n"
        " a) Raise ValueError if len(exchange_rates) < 30\n"
        " b) Default to 0 if no clear signal\n\n"
        "=== CODING CONSTRAINTS ===\n"
        "1. FUNCTION NAME MUST BE 'daily_decision' EXACTLY\n"
        "2. DO NOT IMPORT MODULES or REDEFINE CLASSES\n"
        "3. Use List Comprehensions for data extraction:\n"
        " closing_prices = [er.close for er in exchange_rates]\n"
        "4. Complete all variable assignments\n"
        "5. Handle division zero errors in calculations\n"
        "6. Use f-strings for error messages\n\n"
        "Return ONLY the complete function code starting with # Strategy comment. "
        "NO CLASS DEFINITIONS. NO TEST CODE. NO MARKDOWN."
    )
    
    # Strategy suggestions for exploration
    strategy_suggestions = [
        "Mean Reversion - Buy on dips below a moving average and sell on rallies above it",
        "Trend Following - Follow the established trends with confirmation from other indicators",
        "Breakout Strategy - Buy on upside breakouts and sell on downside breakouts",
        "Volatility Strategy - Buy during low volatility periods and sell during high volatility",
        "Volume Strategy - Focus primarily on volume as the key decision driver",
        "RSI Oscillator - Use RSI extremes as the primary decision maker with confirmation",
        "Moving Average Crossover - Focus on multiple moving average crossovers",
        "Price Pattern - Use price patterns like higher highs/lower lows to make decisions",
        "Momentum Strategy - Buy when momentum increases and sell when it decreases"
    ]
    
    # Determine mode: Every 4th attempt or first attempt is exploration, otherwise refinement
    explore_mode = (attempt_num % 4 == 0 or attempt_num == 1)
    
    # Create strategy guidance based on attempt number and chosen mode
    if attempt_num == 1:
        strategy_message = "\nThis is your first attempt. Create a balanced trading strategy."
        mode_description = "EXPLORATION (first attempt)"
    elif explore_mode:
        # For exploration attempts, suggest a specific type of strategy to try
        strategy_index = (attempt_num // 4) % len(strategy_suggestions)
        suggested_strategy = strategy_suggestions[strategy_index]
        
        strategy_message = (
            f"\nThis is attempt #{attempt_num}. This is an EXPLORATION attempt."
            f"\nYour task is to create a COMPLETELY DIFFERENT strategy than before."
            f"\nSpecifically, consider creating a '{suggested_strategy}' type of strategy."
            f"\nChanging thresholds alone is NOT sufficient - the underlying decision logic should be fundamentally different."
            f"\nUse the required technical factors in novel ways with different combinations and logic."
        )
        mode_description = "EXPLORATION"
    else:
        # Refinement mode is the default
        strategy_message = (
            f"\nThis is attempt #{attempt_num}. This is a REFINEMENT attempt."
            f"\nFocus on IMPROVING the best performing strategy from previous attempts. "
            f"\nRefine the thresholds, decision logic, or signal combinations to make it more effective."
            f"\nYou should keep the core strategy approach similar, but make it more effective by:"
            f"\n1. Fine-tuning thresholds to be more responsive to market conditions"
            f"\n2. Adjusting the decision logic to capture more profitable trades"
            f"\n3. Potentially combining signals in more sophisticated ways"
            f"\n4. Adding conditions to filter out false signals"
        )
        mode_description = "REFINEMENT"
    
    messages = [{
        "role": "system",
        "content": base_prompt + strategy_message
    }]
    
    # Process feedback from previous attempts
    if feedback and attempt_num > 1:
        attempts = [a.strip() for a in feedback.split("----------------------------") if a.strip()]
        
        # Extract previous approaches and results
        prior_approaches = []
        performance_records = {}
        best_strategy = {"description": "", "performance": 0, "code": ""}
        most_recent_strategy = {"description": "", "performance": 0, "code": ""}
        
        for i, attempt in enumerate(attempts):
            strategy_desc = ""
            performance = 0
            
            # Extract strategy description
            for line in attempt.split('\n'):
                if '# Strategy:' in line:
                    strategy_desc = line.strip()
                    break
            
            # Extract performance
            for line in attempt.split('\n'):
                if "Final: €" in line:
                    try:
                        performance = float(line.split("Final: €")[1].split()[0])
                        performance_records[strategy_desc] = performance
                        
                        # Track best strategy
                        if performance > best_strategy["performance"]:
                            best_strategy["description"] = strategy_desc
                            best_strategy["performance"] = performance
                            best_strategy["code"] = attempt
                            
                        # Track most recent strategy (always the last one)
                        if i == len(attempts) - 1:
                            most_recent_strategy["description"] = strategy_desc
                            most_recent_strategy["performance"] = performance
                            most_recent_strategy["code"] = attempt
                    except:
                        pass
                    break
            
            if strategy_desc:
                prior_approaches.append(strategy_desc)
        
        # Different feedback approach based on mode
        if not explore_mode:
            # For refinement mode, prioritize improving the best strategy
            target_strategy = best_strategy if best_strategy["performance"] > 0 else most_recent_strategy
            
            if target_strategy["description"]:
                # Add important context about the performance
                performance_context = ""
                if target_strategy["performance"] == 1000.00 or target_strategy["performance"] == 1000:
                    performance_context = (
                        "\nIMPORTANT: This strategy performed at exactly €1000.00, which means it never executed any trades "
                        "or had a net zero impact (we always start with €1000.00). The strategy's decision logic is likely "
                        "too conservative or the thresholds are too strict, causing it to hold throughout the entire period. "
                        "Your improvements should make the strategy more active in finding trading opportunities."
                    )
                else:
                    return_pct = ((target_strategy["performance"] / 1000.0) - 1) * 100
                    performance_context = f"\nThis represents a {return_pct:+.2f}% return on the initial €1000.00 investment."
                
                messages.append({
                    "role": "user", 
                    "content": (
                        f"I want you to IMPROVE this strategy which performed at €{target_strategy['performance']:.2f}:\n"
                        f"{target_strategy['description']}\n"
                        f"{performance_context}\n\n"
                        f"Make it more effective by:\n"
                        f"1. Fine-tuning the thresholds\n"
                        f"2. Adjusting the decision logic\n"
                        f"3. Better handling of edge cases\n"
                        f"4. Making it more responsive to market conditions\n\n"
                        f"The core approach should remain similar, but make it more effective."
                    )
                })
                
                # Extract and provide the exact code from the target strategy
                code_lines = []
                in_code_section = False
                for line in target_strategy["code"].split("\n"):
                    if line.startswith("Code:"):
                        in_code_section = True
                        continue
                    if in_code_section and not line.startswith("---"):
                        code_lines.append(line)
                
                if code_lines:
                    messages.append({
                        "role": "user",
                        "content": f"Here is the exact code to improve:\n\n{''.join(code_lines)}\n\nMake targeted improvements to this code while keeping the core strategy approach."
                    })
        else:
            # Exploration mode - avoid previous approaches
            if prior_approaches:
                approach_summary = "\n".join(prior_approaches)
                messages.append({
                    "role": "user", 
                    "content": f"I've tried these approaches previously:\n{approach_summary}\n\nPlease create a trading strategy that uses a COMPLETELY DIFFERENT approach than these."
                })
            
            # Add most recent attempt but focus on what didn't work
            performance_line = ""
            for line in most_recent_strategy["code"].split('\n'):
                if "Final:" in line and "Return:" in line:
                    performance_line = line
                    break
            
            # Add context about starting capital
            if "Final: €1000.00" in performance_line or "Final: €1000" in performance_line:
                performance_note = "\nNote: We start with €1000.00, so this strategy didn't execute any trades or had no net impact."
            else:
                performance_note = "\nNote: We start with €1000.00 initial capital."
                
            messages.append({
                "role": "user", 
                "content": (
                    f"My most recent attempt had this performance:\n{performance_line}{performance_note}\n\n"
                    f"Analysis of what didn't work well:\n"
                    f"1. The strategy likely wasn't making enough trades or was making trades at suboptimal times\n"
                    f"2. The decision thresholds may have been too strict or not aligned with market patterns\n" 
                    f"3. The indicators need to be combined in a more effective way\n\n"
                    f"Please develop a trading strategy with a different decision algorithm."
                )
            })
    
    # Adjust temperature based on attempt number and mode
    if not explore_mode:
        # Lower temperature for refinement (more focused)
        temperature = 0.3  
    else:
        # Higher temperature for exploration (more creative)
        temperature = min(0.7, 0.4 + (attempt_num * 0.05))
    
    # Final user instruction
    if not explore_mode and attempt_num > 1:
        messages.append({
            "role": "user", 
            "content": f"Generate an improved version of the strategy for attempt #{attempt_num}. Focus on making it more effective while keeping the core approach."
        })
    else:
        messages.append({
            "role": "user", 
            "content": f"Generate a trading strategy for attempt #{attempt_num}." + 
                       (" Make sure it's fundamentally different from previous attempts." if explore_mode else "")
        })
    
    # Debug output for prompt history
    if debug:
        from rich.text import Text
        from rich.syntax import Syntax
        from rich.box import ROUNDED

        debug_output = Text("LLM Prompt Chain:\n", style="bold cyan")
        debug_output.append(f"\nMode: {mode_description} (Temperature: {temperature:.2f})\n", style="bold magenta")
        
        for idx, msg in enumerate(messages, 1):
            debug_output.append(f"\n{idx}. {msg['role'].upper()}:\n", style="bold yellow")
            content = msg['content'].replace("\n", "\n ")
            debug_output.append(f" {content}\n", style="bright_white")
        console.print(Panel(
            debug_output,
            title=f"[bold]DEBUG - ATTEMPT {attempt_num}/{max_attempts} ({mode_description})[/]",
            border_style="bright_blue",
            padding=(1, 2),
            box=ROUNDED,
            width=min(console.width, 140)
        ))
    
    # Log all prompt text to log file
    with open(log_file, 'a') as log:
        log.write(f"\n>>>>> ATTEMPT {attempt_num}/{max_attempts} ({mode_description}, Temperature: {temperature:.2f}):\n")
        for msg in messages:
            log.write(f"{msg['role'].upper()}: {msg['content']}\n")

    # Get and clean response
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=800
    )
    
    # Extract the response
    code = response.choices[0].message.content
    cleaned_code = code.replace("```python", "").replace("```", "").strip()
    
    # Log the response to log file
    with open(log_file, 'a') as log:
        log.write("\n>>>>> Response:\n")
        log.write(f"{code}\n")
    
    if debug:
        console.print("\n[bold magenta]RAW LLM RESPONSE:[/bold magenta]")
        console.print(Panel(code, title="Original Response", style="yellow"))
        console.print("\n[bold green]PROCESSED CODE:[/bold green]")
        console.print(Panel(cleaned_code, title="Final Output", style="green"))
    
    return cleaned_code

# -----------------------------
# 5. Strategy Validation & Simulation
# -----------------------------
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


def validate_strategy(code: str, historical_data: list[ExchangeRateItem], debug: bool = False) -> tuple[bool, str]:
    """Validate and backtest strategy; returns simulation outcome"""
    #historical_data = load_historical_data()
    namespace = {"ExchangeRateItem": ExchangeRateItem}
    
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"Compilation failed: {str(e)}"
    
    decision_func = namespace.get("daily_decision")
    if not decision_func:
        return False, "Missing daily_decision function"

    # (Optional) Run test cases here if defined...
    # For now, we proceed to full simulation
    console.print("\n[bold]💰 Running Trading Simulation[/bold]")
    try:
        final_value, value_history = simulate_trading(decision_func, historical_data, debug=debug)
        returns_pct = ((final_value / 1000) - 1) * 100
        peak_value = max(value_history)
        drawdown = (peak_value - min(value_history)) / peak_value * 100
        
        result_summary = (
            f"Start: €1000.00 → Final: €{final_value:.2f}\n"
            f"Return: {returns_pct:+.2f}%\n"
            f"Peak Value: €{peak_value:.2f}\n"
            f"Max Drawdown: {drawdown:.1f}%"
        )
        console.print(Panel.fit(result_summary, title="Simulation Results", style="green" if final_value >= 1000 else "red"))
        return True, f"{result_summary}"
        
    except Exception as e:
        return False, f"Simulation failed: {str(e)}"

# -----------------------------
# 6. Main Application
# -----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="AI Trading Strategy Generator")
    parser.add_argument("--attempts", type=int, default=1, 
                       help="Maximum number of generation attempts")
    parser.add_argument("--threshold", type=float, default=20000,
                       help="Target final portfolio value (in EUR)")
    parser.add_argument("--test", action="store_true",
                       help="Run validation tests only")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    return parser.parse_args()


def main():
    args = parse_arguments()  
    # test the test function  
    if args.test:
        test_app()
        return
    console.print(Panel.fit(
        f"[bold]📈 AI Trading Bot[/bold]\n"
        f"Target: €{args.threshold:.2f}\n"
        f"Max Attempts: {args.attempts}",
        style="blue"
    ))

    best_value = 0.0
    best_code = ""
    feedback_history = []

    for attempt in range(1, args.attempts + 1):
        console.rule(f"[bold]Attempt {attempt}/{args.attempts}")
        
        try:
            # Determine mode based on attempt number
            explore_mode = (attempt % 4 == 0 or attempt == 1)
            mode = "EXPLORATION" if explore_mode else "REFINEMENT"
            mode_style = "bright_yellow" if explore_mode else "bright_magenta"
            
            console.print(f"[bold {mode_style}]Mode: {mode}[/bold {mode_style}]")
            
            # Generate and validate strategy with attempt number
            code = generate_trading_strategy(
                feedback="\n".join(feedback_history), 
                attempt_num=attempt,
                max_attempts=args.attempts,
                debug=args.debug
            )
            
            # Extract strategy description for display
            strategy_desc = "Unknown"
            for line in code.split('\n'):
                if '# Strategy:' in line:
                    strategy_desc = line.replace('# Strategy:', '').strip()
                    break
                    
            console.print(f"[bold cyan]Approach: [/bold cyan][yellow]{strategy_desc}[/yellow]")
            
            # Show modification info for refinement attempts
            if not explore_mode and attempt > 1:
                console.print(f"[bold magenta]Refining previous strategy...[/bold magenta]")
            
            # Validate and simulate
            valid, message = validate_strategy(code, load_historical_data(), debug=True)
            
            # Extract simulation result value from message if possible
            current_value = float(message.split("€")[-1].split()[0]) if "€" in message else 0
            
            # Store feedback with strategy information
            feedback_entry = (
                f"Attempt {attempt}: {message}\n"
                f"# Strategy: {strategy_desc}\n"
                f"Code:\n{code}\n"
                f"----------------------------"
            )
            feedback_history.append(feedback_entry)
            
            if valid and current_value > best_value:
                best_value = current_value
                best_code = code
                console.print(Panel.fit(
                    f"[green]New Best Strategy: €{best_value:.2f}[/green]\n"
                    f"Approach: {strategy_desc}\n"
                    f"Mode: {mode}",
                    style="bold green"
                ))

            if best_value >= args.threshold:
                break

        except Exception as e:
            console.print(Panel(f"[red]Error: {str(e)}[/red]", title="Runtime Error"))
    
    # Final results
    if best_value > 0:
        console.print(Panel.fit(
            f"[bold]🏆 Best Result: €{best_value:.2f}[/bold]\n"
            f"Saving strategy to 'best_strategy.py'",
            style="green"
        ))
        with open("best_strategy.py", "w") as f:
            f.write(best_code)
    else:
        console.print(Panel.fit(
            "[red]❌ Failed to generate valid strategy[/red]",
            style="bold red"
        ))
        sys.exit(1)


def test_app():
    # Generate 30 days of September dummy data (2023-09-01 to 2023-09-30)
    exchange_rates = []
    start_date = datetime(2023, 9, 1)
    
    for day in range(30):
        current_date = start_date + timedelta(days=day)
        exchange_rates.append(ExchangeRateItem(
            unix=time.mktime(current_date.timetuple()),
            date=current_date,
            low=25000.0,
            high=25000.0,
            open=25000.0,
            close=25000.0,
            volume=10000.0
        ))
    
    # Add our test days (October 1-3 with significant price moves)
    test_prices = [
        (datetime(2023, 10, 1), 1000.0),
        (datetime(2023, 10, 2), 2000.0),
        (datetime(2023, 10, 3), 3000.0),
    ]
    
    for date, price in test_prices:
        exchange_rates.append(ExchangeRateItem(
            unix=time.mktime(date.timetuple()),
            date=date,
            low=price,
            high=price,
            open=price,
            close=price,
            volume=10000.0
        ))

    # Define test decision function
    def decision_func(data: List[ExchangeRateItem]) -> int:
        current_day = data[-1].date
        
        if current_day == datetime(2023, 10, 1):
            return 2  # Attempt to sell on Oct 1 (no BTC held yet)
        elif current_day == datetime(2023, 10, 2):
            return 1  # Buy on Oct 2 at 2000
        elif current_day == datetime(2023, 10, 3):
            return 0  # Hold on Oct 3
        return 0  # Default: hold

    # Run the simulation with debug output
    final_value, history = simulate_trading(
        decision_func=decision_func,
        exchange_rates=exchange_rates,
        debug=True
    )
    
    print(f"\nTest Results:")
    print(f"Expected final value: €1500.00")
    print(f"Actual final value:   €{final_value:.2f}")
    print(f"Value history: {[round(v, 2) for v in history]}")

if __name__ == "__main__":
    main()
    #test_app()

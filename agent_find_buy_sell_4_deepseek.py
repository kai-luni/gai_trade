# Load environment variables from .env
from typing import Callable, List
from dotenv import load_dotenv
import os
import random
import time

from dto.ExchangeRateItem import ExchangeRateItem
from dto.ObjectsGai import FearGreedItem
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
from openai import AzureOpenAI, OpenAI
from repos import CoinBaseRepo, FearGreedRepo

warnings.simplefilter("ignore", DeprecationWarning)

# Initialize Azure OpenAI client
# client = AzureOpenAI(
#     api_key=AZURE_OPENAI_API_KEY,
#     api_version="2023-12-01-preview",
#     azure_endpoint=AZURE_OPENAI_ENDPOINT
# )
# model_name = "gpt-4o"
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)
model_name = "deepseek-ai/DeepSeek-R1-fast"
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

def generate_trading_strategy(money_if_never_sell, feedback: str = "", attempt_num: int = 1, max_attempts: int = 3, previous_errors: list = None, previous_code: str = None, mode: str = "refinement", debug: bool = False) -> str:
    """Generate daily trading decision function with error feedback to the LLM
    
    Args:
        feedback: Previous feedback to incorporate
        attempt_num: Current attempt number
        max_attempts: Maximum number of attempts
        previous_errors: List of errors from previous attempts
        previous_code: Code from the previous attempt
        mode: Strategy generation mode ("error_fix", "exploring", or "refinement")
        debug: Enable debug output
    
    Returns:
        Generated trading strategy code
    """
    
    # Initialize previous_errors if None
    if previous_errors is None:
        previous_errors = []
    
    # Initialize variables
    temperature = 0.5  # Default temperature if not set elsewhere
    
    # Logging file setup - ONLY FOR LLM MESSAGES
    log_file = 'log.txt'
    with open(log_file, 'a') as log:
        log.write("\n" + "=" * 40 + "\n")
        log.write(f"Execution Date: {datetime.now()}\n")
    
    # Update the base prompt to include Fear & Greed data
    base_prompt = (
        "You are a professional algorithmic trader creating a Python function for Bitcoin trading. "
        "I need ONLY the complete Python code with NO explanations outside the code comments.\n\n"
        "Write a SINGLE Python function named 'daily_decision' using this EXACT signature:\n"
        "def daily_decision(exchange_rates: list[ExchangeRateItem], fear_greed_data: list[FearGreedItem]) -> int:\n\n"
        "=== INPUT SPECIFICATIONS ===\n"
        "1. ExchangeRateItem PREDEFINED CLASS (DO NOT REDECLARE) with fields:\n"
        " - date: datetime (access via .date)\n"
        " - close: float (closing price EUR)\n"
        " - high: float (daily high EUR)\n"
        " - low: float (daily low EUR)\n"
        " - open: float (opening price EUR)\n"
        " - volume: float (BTC traded)\n"
        "2. FearGreedItem PREDEFINED CLASS (DO NOT REDECLARE) with fields:\n"
        " - date: datetime (access via .date)\n"
        " - index: int (Fear & Greed Index value between 0-100)\n"
        " - index_text: str (textual representation: 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed')\n"
        "3. Both input lists are chronological - exchange_rates[-1] and fear_greed_data[-1] represent today\n"
        "=== OUTPUT REQUIREMENTS ===\n"
        "Return ONLY 0/1/2 integer. NO OTHER OUTPUT:\n"
        "- 0: Hold | 1: Buy today | 2: Sell today\n\n"
        "=== STRATEGY RULES ===\n"
        "1. START with '# Strategy: [Strategy Type] - ' comment explaining your approach in detail, like hardcorded values etc, write it in one line\n"
        "2. Required technical calculations (you must calculate these, but you have complete freedom in how you use them):\n"
        " a) Calculate short-term and long-term moving averages of closing prices (e.g., 3-day, 30-day, or other periods)\n"
        " b) Calculate volatility metrics (e.g., (high - low)/close for recent days)\n"
        " c) Analyze volume patterns (e.g., compare recent volume to longer-term averages)\n"
        " d) Calculate momentum indicators (e.g., RSI using a 14-day period)\n"
        " e) Fear and Greed data is optional, if you dont need it, ignore it.\n"
        "3. BE CREATIVE! You can combine these metrics in novel ways or create additional indicators from the price data.\n"
        "4. Edge cases:\n"
        " a) Raise ValueError if len(exchange_rates) < 30\n"
        " b) Default to 0 if no clear signal\n"
        " c) Handle missing fear_greed_data gracefully (use price-based signals only)\n\n"
        "=== CODING CONSTRAINTS ===\n"
        "1. FUNCTION NAME MUST BE 'daily_decision' EXACTLY\n"
        "2. DO NOT IMPORT MODULES or REDEFINE CLASSES\n"
        "3. Use List Comprehensions for data extraction:\n"
        " closing_prices = [er.close for er in exchange_rates]\n"
        " fear_greed_values = [fg.index for fg in fear_greed_data] if fear_greed_data else []\n"
        "4. Complete all variable assignments\n"
        "5. Handle division zero errors in calculations\n"
        "6. Use f-strings for error messages\n\n"
        "PROVIDE ONLY THE COMPLETE FUNCTION CODE. NO EXPLANATIONS OUTSIDE THE CODE COMMENTS.\n"
    )
    
    # Add error feedback if available along with the code that caused the errors
    if mode == "error_fix":            
        error_section = (
            "\n=== PREVIOUS EXECUTION ERRORS ===\n"
            "The previous strategy had the following errors that you need to fix:\n"
            f"{chr(10).join(['- ' + err for err in previous_errors[:15]])}\n"
            "Please carefully check your code for typos, variable name consistency, and logical errors.\n\n"
            "Here is the code that produced these errors:\n\n"
            f"{previous_code}\n\n"
            "Please fix all errors and make the code work correctly.\n"
        )
        base_prompt += error_section
        
        # If there are errors, prioritize fixing them over exploration/refinement
        strategy_message = "\nFIX ERRORS: Your primary task is to fix the errors in the previous code while preserving its overall strategy."
        
        messages = [{
            "role": "system",
            "content": base_prompt + strategy_message
        }]
        
        # Add a direct instruction to fix the errors
        messages.append({
            "role": "user", 
            "content": (
                f"Fix all the errors in the previous code. Pay special attention to:\n"
                f"1. Variable name typos (like 'closing_rices' instead of 'closing_prices')\n"
                f"2. Undefined variables\n"
                f"3. Syntax errors\n"
                f"4. Logic errors in calculations\n\n"
                f"Return the corrected code that fixes all these issues."
            )
        })
        
        # Set temperature low for error fixing
        temperature = 0.2
        
    else:
        # Regular strategy generation logic (exploration or refinement)
        if mode == "exploring":
            # For exploration attempts, suggest a specific type of strategy to try
            strategy_suggestions = [
                "Mean Reversion - Buy when price deviates significantly from its average and sell when it returns",
                "Momentum - Buy when price movement accelerates in one direction and sell when it slows",
                "Breakout - Buy when price moves above a resistance level and sell when it breaks below support",
                "Volatility-based - Buy or sell based on changes in market volatility patterns",
                "Volume Pattern - Make decisions primarily based on unusual volume activity",
                "Multiple Timeframe Analysis - Compare indicators across different time periods for confirmation",
                "Fibonacci Retracement - Use key Fibonacci levels for decision points",
                "Divergence Strategy - Look for divergence between price and technical indicators",
                "Hybrid Approach - Combine multiple signals with a weighted scoring system",
                "Market Regime Detection - Adapt strategy based on bullish, bearish, or sideways markets"
            ]
            strategy_index = (attempt_num // 4) % len(strategy_suggestions)
            suggested_strategy = strategy_suggestions[strategy_index]
            
            if attempt_num == 1:
                strategy_message = "\nThis is your first attempt. Create a creative, innovative trading strategy."
            else:
                strategy_message = (
                    f"\nThis is attempt #{attempt_num}. This is an EXPLORATION attempt."
                    f"\nYour task is to create a COMPLETELY DIFFERENT strategy than before."
                    f"\nSpecifically, consider creating a '{suggested_strategy}' type of strategy."
                    f"\nBe innovative - develop a unique decision algorithm that uses the required calculations in a novel way."
                    f"\nYou have complete freedom to design the trading logic however you think would be most effective."
                )
        else:  # refinement mode
            strategy_message = (
                f"\nThis is attempt #{attempt_num}. This is a REFINEMENT attempt."
                f"\nFocus on IMPROVING the best performing strategy from previous attempts. "
                f"\nRefine the approach to make it more effective by:"
                f"\n1. Fine-tuning parameters to be more responsive to market conditions"
                f"\n2. Enhancing the decision logic to capture more profitable trades"
                f"\n3. Combining signals in more sophisticated ways"
                f"\n4. Adding filters to reduce false signals"
                f"\n5. Adapting the algorithm to work better in different market conditions"
            )
        
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
        last_strategy = {"description": "", "performance": 0, "code": ""}
        
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
                            
                        # Track most recent strategy (always the last one)
                        if i == len(attempts) - 1:
                            last_strategy["description"] = strategy_desc
                            last_strategy["performance"] = performance
                            last_strategy["code"] = attempt
                    except:
                        pass
                    break
            
            if strategy_desc:
                # Add performance to the strategy description
                strategy_with_performance = f"{strategy_desc} [Performance: €{performance:.2f}]"
                # Limit to 400 characters
                if len(strategy_with_performance) > 400:
                    strategy_with_performance = strategy_with_performance[:397] + "..."
                prior_approaches.append(strategy_with_performance)
        
        # Different feedback approach based on mode
        if mode == "refinement":            
            if last_strategy["description"]:
                # Add important context about the performance
                performance_context = ""
                if int(last_strategy["performance"]) == int(money_if_never_sell):
                    performance_context = (
                        f"\nIMPORTANT: This strategy performed at exactly €{money_if_never_sell}, which means it never executed any trades "
                        f"or had a net zero impact (we always finish with €{money_if_never_sell} if we do nothing). The strategy's decision logic is likely "
                        "too conservative or the thresholds are too strict, causing it to hold throughout the entire period. "
                        "Your improvements should make the strategy more active in finding trading opportunities."
                    )
                else:
                    return_pct = ((last_strategy["performance"] / money_if_never_sell) - 1) * 100
                    performance_context = f"\nThis represents a {return_pct:+.2f}% return on the initial €{money_if_never_sell} investment."
                
                messages.append({
                    "role": "user", 
                    "content": (
                        f"I want you to IMPROVE this strategy which performed at €{last_strategy['performance']:.2f}:\n"
                        f"{last_strategy['description']}\n"
                        f"{performance_context}\n\n"
                        f"Make it more effective by:\n"
                        f"1. Fine-tuning the algorithm parameters\n"
                        f"2. Adjusting the decision logic\n"
                        f"3. Better handling of edge cases\n"
                        f"4. Making it more responsive to market conditions\n\n"
                        f"The core approach should remain similar, but make it more effective."
                    )
                })
                
                # Extract and provide the exact code from the target strategy
                code_lines = []
                in_code_section = False
                for line in last_strategy["code"].split("\n"):
                    if line.startswith("Code:"):
                        in_code_section = True
                        continue
                    if in_code_section and not line.startswith("---"):
                        code_lines.append(line)
                
                # Use previous_code in refinement mode (NEW ADDITION)
                if previous_code:
                    messages.append({
                        "role": "user",
                        "content": f"Here is the exact code to improve:\n\n{previous_code}\n\nMake targeted improvements to this code while keeping the core strategy approach."
                    })
                elif code_lines:
                    messages.append({
                        "role": "user",
                        "content": f"Here is the exact code to improve:\n\n{''.join(code_lines)}\n\nMake targeted improvements to this code while keeping the core strategy approach."
                    })
        elif mode == "exploring":
            # Exploration mode - avoid previous approaches
            if prior_approaches:
                approach_summary = "\n".join(prior_approaches)
                messages.append({
                    "role": "user", 
                    "content": f"I've tried these approaches previously:\n{approach_summary}\n\nPlease create a trading strategy that uses a COMPLETELY DIFFERENT approach than these."
                })
                
                # Add most recent attempt but focus on what didn't work
                performance_line = ""
                for line in last_strategy["code"].split('\n'):
                    if "Final:" in line and "Return:" in line:
                        performance_line = line
                        break
                
                # Add context about starting capital
                if f"Final: €{int(money_if_never_sell)}" in performance_line:
                    performance_note = "\nNote: We start with €1000.00 and invest another 100 each first of the month, this strategy didn't execute any trades or had no net impact."
                else:
                    performance_note = "\nNote: We start with €1000.00 initial capital and invest another 100 each first of the month."
                    
                messages.append({
                    "role": "user", 
                    "content": (
                        f"My most recent attempt had this performance:\n{performance_line}{performance_note}\n\n"
                        f"Analysis of what didn't work well:\n"
                        f"1. The strategy likely wasn't making enough trades or was making trades at suboptimal times\n"
                        f"2. The decision thresholds may have been too strict or not aligned with market patterns\n" 
                        f"3. The indicators need to be combined in a more effective way\n\n"
                        f"Please develop a creative trading strategy with a different decision algorithm."
                    )
                })
        
        # Adjust temperature based on attempt number and mode
        if mode == "refinement":
            # Lower temperature for refinement (more focused)
            temperature = 0.3  
        elif mode == "exploring":
            # Higher temperature for exploration (more creative)
            temperature = min(0.8, 0.5 + (attempt_num * 0.05))  # Slightly higher for more creativity
        
        # Final user instruction
        if mode == "refinement" and attempt_num > 1:
            messages.append({
                "role": "user", 
                "content": f"Generate an improved version of the strategy for attempt #{attempt_num}. Focus on making it more effective while keeping the core approach."
            })
        elif mode == "exploring":
            messages.append({
                "role": "user", 
                "content": f"Generate a creative trading strategy for attempt #{attempt_num}. Make sure it's fundamentally different from previous attempts."
            })
    
    # Debug output for prompt history (console only)
    if debug:
        from rich.text import Text
        from rich.syntax import Syntax
        from rich.box import ROUNDED

        debug_output = Text("LLM Prompt Chain:\n", style="bold cyan")
        debug_output.append(f"\nMode: {mode} (Temperature: {temperature:.2f})\n", style="bold magenta")
        
        for idx, msg in enumerate(messages, 1):
            debug_output.append(f"\n{idx}. {msg['role'].upper()}:\n", style="bold yellow")
            content = msg['content'].replace("\n", "\n ")
            debug_output.append(f" {content}\n", style="bright_white")
        console.print(Panel(
            debug_output,
            title=f"[bold]DEBUG - ATTEMPT {attempt_num}/{max_attempts} ({mode})[/]",
            border_style="bright_blue",
            padding=(1, 2),
            box=ROUNDED,
            width=min(console.width, 140)
        ))
    
    # Log messages to LLM to log file
    with open(log_file, 'a') as log:
        log.write("\n>>>>>>>>>> TO LLM:\n")
        for msg in messages:
            log.write(f"{msg['role'].upper()}: {msg['content']}\n")

    # API call with retry mechanism
    max_retries = 5
    retry_delay = 2  # Initial delay in seconds
    
    for retry in range(max_retries):
        try:
            # Attempt API call
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=16000,
                top_p=0.9,
                frequency_penalty=0,
                presence_penalty=0.1
            )
            
            # Extract and clean response
            raw_response_text = response.choices[0].message.content
            cleaned_code = extract_clean_response(raw_response_text)
            
            # Log success (console only)
            console.print(f"[green]API call successful on attempt {retry + 1}[/green]")
            
            # Log the response to log file (raw LLM response only)
            with open(log_file, 'a') as log:
                log.write("\n>>>>>>>>>> FROM LLM:\n")
                log.write(f"{cleaned_code}\n")
            
            if debug:
                # console.print("\n[bold magenta]RAW LLM RESPONSE:[/bold magenta]")
                # console.print(Panel(raw_response_text, title="Original Response", style="yellow"))
                console.print("\n[bold green]PROCESSED CODE:[/bold green]")
                console.print(Panel(cleaned_code, title="Final Output", style="green"))
            
            # Successful API call, return the response
            return cleaned_code
            
        except Exception as e:
            # Log error (console only)
            error_msg = f"API error on retry {retry + 1}/{max_retries}: {str(e)}"
            console.print(f"[yellow]{error_msg}[/yellow]")
            
            # Check if we've reached max retries
            if retry == max_retries - 1:
                # If this was our last attempt, exit application
                fatal_error = f"FATAL ERROR: Maximum retries ({max_retries}) exceeded. Last error: {str(e)}"
                console.print(f"[bold red]{fatal_error}[/bold red]")
                raise RuntimeError(fatal_error)  # Raise exception to terminate execution
            
            # Exponential backoff with jitter
            jitter = random.uniform(0, 0.5)
            sleep_time = retry_delay * (2 ** retry) + jitter
            console.print(f"[yellow]Retrying in {sleep_time:.2f} seconds...[/yellow]")
            time.sleep(sleep_time)


def extract_clean_response(response_text):
    """
    Extract clean code from the model's response, handling potential thinking tags
    and other formatting particularities
    """
    # Remove thinking tags if present
    if "<think>" in response_text and "</think>" in response_text:
        # Get everything after the thinking section
        response_text = response_text.split("</think>")[-1].strip()
    
    # If that didn't work, try to find the actual Python code
    if not response_text.strip().startswith("# Strategy:"):
        # Look for code block markers
        if "```python" in response_text:
            code_blocks = response_text.split("```python")
            if len(code_blocks) > 1:
                response_text = code_blocks[1]
                if "```" in response_text:
                    response_text = response_text.split("```")[0]
    
    # Remove any markdown code block markers
    cleaned_code = response_text.replace("```python", "").replace("```", "").strip()
    
    # Ensure it starts with the strategy comment
    if not cleaned_code.startswith("# Strategy:"):
        cleaned_code = "# Strategy: Generic Trading Strategy - Generated output needed cleaning\n" + cleaned_code
    
    return cleaned_code

# -----------------------------
# 5. Strategy Validation & Simulation
# -----------------------------
def simulate_trading(decision_func: Callable[[list[ExchangeRateItem], list[FearGreedItem]], int], 
                    exchange_rates: list[ExchangeRateItem], 
                    fear_greed_data: list[FearGreedItem] = None, 
                    debug: bool = False) -> tuple[float, list, list]:
    """
    Simulates a BTC trading strategy from day 30 onward using historical data.

    Args:
        decision_func: Function that analyzes historical data and returns:
                       1 = BUY (use all cash), 2 = SELL (all BTC), 0 = HOLD
        exchange_rates: Historical BTC price data (requires >=30 entries minimum)
        fear_greed_data: Historical Fear & Greed Index data (optional)
        debug: Show trading actions in console when enabled

    Returns:
        Tuple containing:
        - float: Final portfolio value (cash + BTC at last day's close price)
        - list: Daily portfolio values (in EUR) for all trading days
        - list: Errors encountered during simulation
    """
    portfolio = {
        'cash': 1000.00,  # Initial capital in EUR
        'btc': 0.0,       # BTC holdings
        'history': []     # Daily portfolio value in EUR
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
    
    for i in range(30, len(exchange_rates)):  # Start from day 30 (min required data)
        try:
            # Get data up to current day
            current_data = exchange_rates[:i+1]
            today = current_data[-1]
            
            # Add 100 euros on the first day of each month
            if today.date.day == 1:
                portfolio['cash'] += 100.00
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
            
            # Get decision for today
            decision = decision_func(current_data, current_fear_greed)
            
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
                
                console.print(f"[blue]Day {today.date}:[/blue] Decision = {decision}, Price = {today.close:.2f}, Cash = €{portfolio['cash']:.2f}, BTC Cash = {(portfolio['btc'] * today.close):.2f}{fg_info}")
            
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

# 4. Update the validate_strategy function to pass fear & greed data
def validate_strategy(money_if_never_sell: float, code: str, historical_data: list[ExchangeRateItem], fear_greed_data: list[FearGreedItem],  debug: bool = False) -> tuple[bool, str, list]:
    """Validate and backtest strategy; returns simulation outcome with errors if any"""
    namespace = {
        "ExchangeRateItem": ExchangeRateItem,
        "FearGreedItem": FearGreedItem
    }
    
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"Compilation failed: {str(e)}", [f"Compilation error: {str(e)}"]
    
    decision_func = namespace.get("daily_decision")
    if not decision_func:
        return False, "Missing daily_decision function", ["Code doesn't contain a daily_decision function"]

    # Run simulation with error collection
    console.print("\n[bold]💰 Running Trading Simulation[/bold]")
    try:
        final_value, value_history, errors = simulate_trading(decision_func, historical_data, fear_greed_data, debug=debug)
        
        # Format result summary
        returns_pct = ((final_value / money_if_never_sell) - 1) * 100
        peak_value = max(value_history)
        drawdown = (peak_value - min(value_history)) / peak_value * 100
        
        result_summary = (
            f"Start: {money_if_never_sell} → Final: €{final_value:.2f}\n"
            f"Return: {returns_pct:+.2f}%\n"
            f"Peak Value: €{peak_value:.2f}\n"
            f"Max Drawdown: {drawdown:.1f}%"
        )
        
        # Display simulation errors if any
        if errors:
            error_summary = "\n[bold red]Simulation Errors:[/bold red]\n"
            error_summary += "\n".join([f"- {err}" for err in errors[:10]])
            if len(errors) > 10:
                error_summary += f"\n- Plus {len(errors) - 10} more errors..."
            console.print(error_summary)
            
        console.print(Panel.fit(result_summary, title="Simulation Results", style="green" if final_value >= money_if_never_sell else "red"))
        return True, f"{result_summary}", errors
        
    except Exception as e:
        return False, f"Simulation failed: {str(e)}", [f"Simulation system error: {str(e)}"]

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
        f"[bold]📈 AI Trading Bot (with Fear & Greed Index)[/bold]\n"
        f"Target: €{args.threshold:.2f}\n"
        f"Max Attempts: {args.attempts}",
        style="blue"
    ))

    # Load historical price data
    historical_data = load_historical_data()
    
    # Load Fear & Greed data
    fear_greed_data = FearGreedRepo.FearGreedRepo.read_csv_file(datetime(2018, 1, 1), datetime(2024, 9, 1))

    # find out how much money we would have if we always hold (buy btc)
    def no_do_func(data: List[ExchangeRateItem], fear_greed_data: List[FearGreedItem] = None) -> int:
        return 1

    money_if_never_sell, _, _ = simulate_trading(no_do_func, historical_data, fear_greed_data, debug=False)

    best_value = 0.0
    best_code = ""
    feedback_history = []
    recent_errors = []      # Track errors from recent attempts
    previous_code = None   # Store code that caused errors

    for attempt in range(1, args.attempts + 1):
        console.rule(f"[bold]Attempt {attempt}/{args.attempts}")
        
        try:
            # Determine mode based on attempt number, but override if there are errors to fix
            if recent_errors and previous_code:
                mode = "error_fix"
                mode_style = "bright_red"
                console.print(f"[bold {mode_style}]Mode: ERROR FIXING[/bold {mode_style}]")
                console.print("[bold red]Errors detected - switching to error fixing mode[/bold red]")
            else:
                explore_mode = (attempt % 4 == 0 or attempt == 1)
                mode = "exploring" if explore_mode else "refinement"
                mode_style = "bright_yellow" if explore_mode else "bright_magenta"
                mode_display = "EXPLORATION" if explore_mode else "REFINEMENT"
                console.print(f"[bold {mode_style}]Mode: {mode_display}[/bold {mode_style}]")
            
            # If there were errors in previous attempt, display them
            if recent_errors:
                console.print("[bold red]Passing previous errors to LLM for correction:[/bold red]")
                for error in recent_errors[:5]:  # Show first 5 errors
                    console.print(f"[red]- {error}[/red]")
                if len(recent_errors) > 5:
                    console.print(f"[red]- Plus {len(recent_errors) - 5} more errors...[/red]")
            
            # Generate and validate strategy with attempt number
            code = generate_trading_strategy(
                money_if_never_sell,
                feedback="\n".join(feedback_history), 
                attempt_num=attempt,
                max_attempts=args.attempts,
                previous_errors=recent_errors,     # Pass errors to LLM
                previous_code=previous_code,       # Pass code that caused errors
                mode=mode,                         # Pass the mode parameter
                debug=args.debug
            )
            
            # Reset errors and error code for this new attempt
            previous_code = code  # Store current code in case it has errors
            recent_errors = []
            
            # Extract strategy description for display
            strategy_desc = "Unknown"
            for line in code.split('\n'):
                if '# Strategy:' in line:
                    strategy_desc = line.replace('# Strategy:', '').strip()
                    break
                    
            console.print(f"[bold cyan]Approach: [/bold cyan][yellow]{strategy_desc}[/yellow]")
            
            # Validate and simulate
            valid, message, errors = validate_strategy(money_if_never_sell, code, historical_data, fear_greed_data, debug=True)
            
            # Store errors for next attempt if needed
            if errors:
                recent_errors = errors
                console.print(f"[yellow]Found {len(errors)} errors to fix in next attempt[/yellow]")
                # We'll keep previous_code as it is, since it contains the errors
            else:
                # Clear previous_code if no errors
                previous_code = None
            
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
                mode_display = "ERROR FIXING" if mode == "error_fix" else "EXPLORATION" if mode == "exploring" else "REFINEMENT"
                console.print(Panel.fit(
                    f"[green]New Best Strategy: €{best_value:.2f}[/green]\n"
                    f"Approach: {strategy_desc}\n"
                    f"Mode: {mode_display}",
                    style="bold green"
                ))
                
                # Save the best strategy immediately when a new record is hit
                with open("best_strategy.py", "w") as f:
                    f.write(best_code)
                console.print("[green]Saved best strategy to 'best_strategy.py'[/green]")

            if best_value >= args.threshold:
                break

        except Exception as e:
            console.print(Panel(f"[red]Error: {str(e)}[/red]", title="Runtime Error"))
            recent_errors = [f"System error: {str(e)}"]
    
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

# 6. Update the test function to include Fear & Greed data
def test_app():
    # Generate 30 days of September dummy data (2023-09-01 to 2023-09-30)
    exchange_rates = []
    fear_greed_items = []
    start_date = datetime(2023, 9, 1)
    
    for day in range(30):
        current_date = start_date + timedelta(days=day)
        
        # Create exchange rate item
        exchange_rates.append(ExchangeRateItem(
            unix=time.mktime(current_date.timetuple()),
            date=current_date,
            low=25000.0,
            high=25000.0,
            open=25000.0,
            close=25000.0,
            volume=10000.0
        ))
        
        # Create fear & greed item with alternating values
        index_value = 25 if day % 2 == 0 else 75  # Alternate between fear and greed
        index_text = "Fear" if index_value < 50 else "Greed"
        
        fg_item = FearGreedItem()
        fg_item.unix = time.mktime(current_date.timetuple())
        fg_item.date = current_date
        fg_item.index = index_value
        fg_item.index_text = index_text
        
        fear_greed_items.append(fg_item)
    
    # Add our test days (October 1-3 with significant price moves)
    test_prices = [
        (datetime(2023, 10, 1), 1000.0, 30, "Fear"),
        (datetime(2023, 10, 2), 2000.0, 15, "Extreme Fear"),
        (datetime(2023, 10, 3), 3000.0, 60, "Greed"),
    ]
    
    for date, price, fg_index, fg_text in test_prices:
        # Add exchange rate item
        exchange_rates.append(ExchangeRateItem(
            unix=time.mktime(date.timetuple()),
            date=date,
            low=price,
            high=price,
            open=price,
            close=price,
            volume=10000.0
        ))
        
        # Add fear & greed item
        fg_item = FearGreedItem()
        fg_item.unix = time.mktime(date.timetuple())
        fg_item.date = date
        fg_item.index = fg_index
        fg_item.index_text = fg_text
        
        fear_greed_items.append(fg_item)
    
    # Define test decision function that uses both data types
    def decision_func(data: List[ExchangeRateItem], fear_greed_data: List[FearGreedItem] = None) -> int:
        current_day = data[-1].date
        
        # Use both price and sentiment data if available
        fg_info = ""
        if fear_greed_data and len(fear_greed_data) > 0:
            today_fg = next((fg for fg in fear_greed_data if fg.date.date() == current_day.date()), None)
            if today_fg:
                fg_info = f" (F&G: {today_fg.index} - {today_fg.index_text})"
                # Example of using sentiment: Buy in extreme fear
                if today_fg.index < 20 and current_day == datetime(2023, 10, 2):
                    console.print(f"[green]Buying on Extreme Fear{fg_info}[/green]")
                    return 1
        
        if current_day == datetime(2023, 10, 1):
            console.print(f"[yellow]Attempt to sell on Oct 1{fg_info} (no BTC held yet)[/yellow]")
            return 2  # Attempt to sell on Oct 1 (no BTC held yet)
        elif current_day == datetime(2023, 10, 2):
            console.print(f"[green]Buy on Oct 2 at 2000{fg_info}[/green]")
            return 1  # Buy on Oct 2 at 2000
        elif current_day == datetime(2023, 10, 3):
            console.print(f"[blue]Hold on Oct 3{fg_info}[/blue]")
            return 0  # Hold on Oct 3
        
        console.print(f"[blue]Default hold{fg_info}[/blue]")
        return 0  # Default: hold
    
    # Run the simulation with debug output
    final_value, history, errors = simulate_trading(
        decision_func=decision_func,
        exchange_rates=exchange_rates,
        fear_greed_data=fear_greed_items,
        debug=True
    )
    
    print(f"\nTest Results:")
    print(f"Expected final value: €1500.00")
    print(f"Actual final value:   €{final_value:.2f}")
    print(f"Value history: {[round(v, 2) for v in history]}")
    
    if errors:
        print(f"\nSimulation Errors:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()

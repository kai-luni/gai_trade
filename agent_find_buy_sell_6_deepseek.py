# Load environment variables from .env
from typing import Callable, List
from dotenv import load_dotenv
import os
import random
import time

from dto.NetflowDto import NetflowDto
from dto.ExchangeRateItem import ExchangeRateItem
from dto.ObjectsGai import FearGreedItem
from helper.LoggerPrime import LoggerPrime, LogLevel

#imports for test algo
import numpy as np

from tests.test_deepseek_agent import test_app

load_dotenv()

# Initialize logger
logger = LoggerPrime(
    name="trading_bot",
    log_file="trading_bot.log"
)

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
from datetime import datetime, timedelta
from rich.panel import Panel
from openai import AzureOpenAI, OpenAI
from repos import CoinBaseRepo, FearGreedRepo, NetflowRepo

warnings.simplefilter("ignore", DeprecationWarning)

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)
model_name = os.environ.get("DEEPSEEK_V3_NAME")

# -----------------------------
# 2. Core Domain Objects
# -----------------------------

# -----------------------------
# 3. Data Preparation
# -----------------------------

# -----------------------------
# 4. LLM Strategy Generation
# -----------------------------

def generate_trading_strategy(money_if_never_sell, attempts: list = None, attempt_num: int = 1, previous_errors: list = None, previous_code: str = None, mode: str = "refinement", debug: bool = False) -> str:
    """Generate daily trading decision function with error feedback to the LLM

    Args:
        attempts: A list of DTO objects, each with three non-empty fields:
                  - description: str
                  - performance: numeric (int or float)
                  - code: str
                  The function will throw an exception if any of these fields is empty.
        attempt_num: Current attempt number
        previous_errors: List of errors from previous attempts
        previous_code: Code from the previous attempt
        mode: Strategy generation mode ("error_fix", "exploring", or "refinement")
        debug: Enable debug output

    Returns:
        Generated trading strategy code (a complete Python function as a string)
    """
    if previous_errors is None:
        previous_errors = []

    temperature = 0.5

    base_prompt = (
        "You are a REVOLUTIONARY algorithmic trader creating an experimental Python function for Bitcoin trading. "
        "I need you to break conventional trading wisdom and devise something truly ORIGINAL. "
        "I need ONLY the complete Python code with NO explanations outside the code comments.\n\n"
        "Write a Python function named 'daily_decision' using this EXACT signature:\n"
        "def daily_decision(exchange_rates: list[ExchangeRateItem], fear_greed_data: list[FearGreedItem], netflow_data: list[NetflowDto]) -> int:\n\n"
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
        "3. NetflowDto PREDEFINED CLASS (DO NOT REDECLARE) with fields:\n"
        " - date_time: datetime (access via .date_time)\n"
        " - aggregated_exchanges_normalized: float (Amount of Bitcoin flowing in/out of exchanges, normalized from -1 to 1)\n"
        "   Positive values indicate BTC flowing INTO exchanges (bearish - people are preparing to sell)\n"
        "   Negative values indicate BTC flowing OUT OF exchanges (bullish - people are withdrawing to hold)\n"
        "4. All input lists are chronological - exchange_rates[-1], fear_greed_data[-1], and netflow_data[-1] represent today\n"
        "5. The data is checked to be complete, all fields have values. The data is going at least 30 days back, mostly more.\n\n"
        "=== OUTPUT REQUIREMENTS ===\n"
        "Return ONLY 0/1/2 integer. NO OTHER OUTPUT:\n"
        "- 0: Hold | 1: Buy today | 2: Sell today\n"
        "NOTE: The algorithm automatically adds 100 EUR in BTC on account. \n\n"
        "=== INNOVATION CHALLENGE ===\n"
        "Try both, conventional and uncenventional algorithms, anythings goes.\n\n"
        "=== STRATEGY RULES ===\n"
        "1. START with '# Strategy: [Strategy Type] - ' comment explaining your RADICAL approach in detail\n"
        "   - This comment MUST be on its own line ABOVE the function definition\n"
        "   - There MUST be a linebreak after this comment\n"
        "   - The function definition must start on a new line after the strategy comment\n"
        "   - REALLY PUT THOUGHT into naming your strategy something original!\n"
        "2. IMPORTANT: The algorithm receives 100 EUR to invest on the 1st of each month. These funds are added to the available cash balance regardless of the daily_decision output. When returning \"1\" (Buy), the algorithm should use available cash to purchase BTC.\n"
        "3. EMBRACE CREATIVITY:\n"
        " a) Develop adaptive thresholds that evolve with market conditions, not fixed values\n"
        " b) Combine multiple data sources in ways that reveal hidden insights\n"
        " c) Create novel mathematical relationships between technical and sentiment data\n"
        " d) Consider how the RELATIONSHIP between indicators might be more important than the indicators themselves\n"
        "4. You can use sub functions to create sophisticated calculations\n\n"
        "=== CODING CONSTRAINTS ===\n"
        "1. FUNCTION NAME MUST BE 'daily_decision' EXACTLY\n"
        "2. Feel free to add some import :) If the module is missing I will add it.\n"
        "3. Complete all variable assignments\n"
        "4. Handle division zero errors in calculations\n"
        "5. Use f-strings for error messages\n\n"
        "PROVIDE ONLY THE COMPLETE FUNCTION CODE. NO EXPLANATIONS OUTSIDE THE CODE COMMENTS.\n"
    )

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
        strategy_message = "\nFIX ERRORS: Your primary task is to fix the errors in the previous code while preserving its overall strategy."
        messages = [{
            "role": "system",
            "content": base_prompt + strategy_message
        }]
        messages.append({
            "role": "user",
            "content": (
                "Fix all the errors in the previous code. Pay special attention to:\n"
                "1. Variable name typos (like 'closing_rices' instead of 'closing_prices')\n"
                "2. Undefined variables\n"
                "3. Syntax errors\n"
                "4. Logic errors in calculations\n\n"
                "Return the corrected code that fixes all these issues."
            )
        })
        temperature = 0.2
    else:
        if mode == "exploring":
            if attempt_num == 1:
                strategy_message = "\nThis is your first attempt. Create a creative, innovative trading strategy."
            else:
                strategy_message = (
                    f"\nThis is attempt #{attempt_num}. This is an EXPLORATION attempt."
                    "\nYour task is to create a COMPLETELY DIFFERENT strategy than before."
                    "\nBe innovative and experimental - develop a unique decision algorithm."
                    "\nYou have complete freedom in your approach:"
                    "\n1. You DO NOT need to use all available data fields - focus on the most relevant indicators."
                    "\n2. Try fundamentally different approaches than what's been tried before."
                    "\n3. Consider unconventional indicators or unique combinations."
                    "\n4. You can ignore some data fields if they don't fit your approach."
                    "\n5. Think outside the box and create something novel."
                )
        else:
            strategy_message = (
                f"\nThis is attempt #{attempt_num}. This is a REFINEMENT attempt."
                "\nFocus on IMPROVING the best performing strategy from previous attempts."
                "\nRefine the approach to make it more effective by:"
                "\n1. Fine-tuning parameters to be more responsive to market conditions."
                "\n2. Enhancing the decision logic for more profitable trades."
                "\n3. Combining signals in a more sophisticated way."
                "\n4. Adding filters to reduce false signals."
                "\n5. Adapting the algorithm to work better in different market conditions."
            )
        messages = [{
            "role": "system",
            "content": base_prompt + strategy_message
        }]

    def is_valid_field(value):
        if value is None:
            return False
        if isinstance(value, str) and value == "":
            return False
        return True
    # Process feedback from previous attempts using the 'attempts' DTO list without extra processing
    if attempts and attempt_num > 1:
        for attempt_dto in attempts:
            if not all(is_valid_field(attempt_dto.get(key)) for key in ("description", "performance", "code")):
                raise ValueError("Each attempt must have non-empty 'description', 'performance', and 'code' fields.")
        last_attempt = attempts[-1]
        try:
            last_performance = float(last_attempt["performance"])
        except Exception:
            raise ValueError("Performance value must be convertible to float.")
        attempts_summary = "\n".join(
            f"{a['description']} [Performance: €{float(a['performance']):.2f}]"
            for a in attempts
        )
        if mode == "refinement":
            performance_context = ""
            if abs(int(last_performance) - int(money_if_never_sell)) <= 1:
                performance_context = (
                    f"\nIMPORTANT: This strategy performed at effectively €{money_if_never_sell}, indicating no trades were executed. "
                    "Improve the decision logic to capture trading opportunities."
                )
            else:
                return_pct = ((last_performance / money_if_never_sell) - 1) * 100
                performance_context = f"\nThis represents a {return_pct:+.2f}% return on the initial €{money_if_never_sell} investment."
            messages.append({
                "role": "user",
                "content": (
                    f"I want you to IMPROVE this strategy which performed at €{last_performance:.2f}:\n"
                    f"{last_attempt['description']}\n"
                    f"{performance_context}\n\n"
                    "Make it more effective by:\n"
                    "1. Fine-tuning the algorithm parameters\n"
                    "2. Adjusting the decision logic\n"
                    "3. Better handling of edge cases\n"
                    "4. Making it more responsive to market conditions\n\n"
                    "Keep the core approach similar, but enhance its effectiveness."
                )
            })
            if previous_code:
                messages.append({
                    "role": "user",
                    "content": f"Here is the exact code to improve:\n\n{previous_code}\n\nMake targeted improvements while retaining the core strategy."
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Here is the exact code to improve:\n\n{last_attempt['code']}\n\nMake targeted improvements while retaining the core strategy."
                })
        elif mode == "exploring":
            messages.append({
                "role": "user",
                "content": (
                    f"I've tried these approaches previously:\n{attempts_summary}\n\n"
                    "Please create a trading strategy that uses a COMPLETELY DIFFERENT approach than these. "
                    "You may ignore some data fields if they don't fit your approach. Focus on a clear, innovative strategy."
                )
            })
            last_performance = last_attempt["performance"]
            messages.append({
                "role": "user",
                "content": (
                    f"My most recent attempt had this performance:\n"
                    f"Performance: {last_performance} Euro. DCA would have made: {money_if_never_sell} Euro.\n\n"
                    "Analysis of what didn't work well:\n"
                    "1. The strategy likely wasn't making enough trades or made trades at suboptimal times\n"
                    "2. The decision thresholds may have been too strict or misaligned with market patterns\n"
                    "3. The indicators need to be combined more effectively\n\n"
                    "Please develop a creative trading strategy with a different decision algorithm."
                )
            })
        temperature = 0.3 if mode == "refinement" else min(0.95, 0.6 + (attempt_num * 0.1))
        if mode == "refinement" and attempt_num > 1:
            messages.append({
                "role": "user",
                "content": f"Generate an improved version of the strategy for attempt #{attempt_num}. Focus on making it more effective while keeping the core approach."
            })
        elif mode == "exploring":
            messages.append({
                "role": "user",
                "content": (
                    f"Generate a creative trading strategy for attempt #{attempt_num}. "
                    "Ensure it is fundamentally different from previous attempts. "
                    "Focus on the most promising data fields rather than all available data."
                )
            })

    if debug:
        logger.set_level(console_level=LogLevel.DEBUG)
        debug_output = f"\nMode: {mode} (Temperature: {temperature:.2f})\n"
        for idx, msg in enumerate(messages, 1):
            debug_output += f"\n{idx}. {msg['role'].upper()}:\n {msg['content'].replace(chr(10), chr(10)+' ')}\n"
        logger.panel(
            debug_output,
            title=f"DEBUG - ATTEMPT {attempt_num}/3 ({mode})",
            border_style="bright_blue",
            padding=(1, 2),
            width=140
        )

    prompt_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])

    max_retries = 5
    retry_delay = 2

    for retry in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=16000,
                top_p=0.9,
                frequency_penalty=0,
                presence_penalty=0.1
            )
            raw_response_text = response.choices[0].message.content
            cleaned_code = extract_clean_response(raw_response_text)
            logger.success(f"API call successful on attempt {retry + 1}")
            logger.log_llm_interaction(prompt_text, cleaned_code)
            if debug:
                logger.panel(cleaned_code, title="Final Output", style="green")
            return cleaned_code
        except Exception as e:
            error_msg = f"API error on retry {retry + 1}/5: {str(e)}"
            logger.warning(error_msg)
            if retry == max_retries - 1:
                fatal_error = f"FATAL ERROR: Maximum retries (5) exceeded. Last error: {str(e)}"
                logger.critical(fatal_error)
                raise RuntimeError(fatal_error)
            jitter = random.uniform(0, 0.5)
            sleep_time = retry_delay * (2 ** retry) + jitter
            logger.warning(f"Retrying in {sleep_time:.2f} seconds...")
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
                # if debug:
                #     logger.debug(f"First day of month ({today.date}): Added €100.00 to portfolio")
            
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
                logger.success(f"BUY signal on {today.date} at price {today.close:.2f}")
                portfolio['btc'] = portfolio['btc'] + (portfolio['cash'] / today.close)
                portfolio['cash'] = 0.0
            elif decision == 2 and portfolio['btc'] > 0:
                logger.error(f"SELL signal on {today.date} at price {today.close:.2f}")
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
                        if today_nf and today_nf.aggregated_exchanges_normalized is not None:
                            flow_dir = "IN" if today_nf.aggregated_exchanges_normalized > 0 else "OUT"
                            nf_info = f", Netflow: {abs(today_nf.aggregated_exchanges_normalized):.2f} BTC {flow_dir}"
                    except:
                        pass
                
                logger.debug(f"Day {today.date}: Decision = {decision}, Price = {today.close:.2f}, Cash = €{portfolio['cash']:.2f}, BTC Cash = {(portfolio['btc'] * today.close):.2f}{fg_info}{nf_info}")
            
            # Record portfolio value (cash + BTC value)
            current_value = portfolio['cash'] + (portfolio['btc'] * today.close)
            portfolio['history'].append(current_value)
            
        except Exception as e:
            error_msg = f"Error on {today.date}: {str(e)}"
            if debug:
                logger.error(error_msg)
            
            # Add error to collection with date context
            errors.append(error_msg)
            return 0.0, [0.0], errors

    
    return portfolio['history'][-1], portfolio['history'], errors

# 4. Update the validate_strategy function to pass fear & greed data
def validate_strategy(money_if_never_sell: float, code: str, historical_data: list[ExchangeRateItem], fear_greed_data: list[FearGreedItem], netflow_data: list[NetflowDto] = None, debug: bool = False) -> tuple[bool, str, list, float]:
    """
    Validate and backtest strategy; returns simulation outcome with errors if any, and the final value
    
    Args:
        money_if_never_sell: Float value representing how much money would be worth if never selling (DCA baseline)
        code: String containing the trading strategy implementation code
        historical_data: Historical BTC price data (requires >=30 entries minimum)
        fear_greed_data: Historical Fear & Greed Index data (optional)
        netflow_data: Historical Bitcoin exchange flow data (optional)
        debug: Show trading actions in console when enabled
    
    Returns:
        Tuple containing:
        - bool: Success status of the simulation (True if successful, False if errors occurred)
        - str: Result summary or error message
        - list: Errors encountered during simulation (if any)
        - float: Final portfolio value (in EUR)
    """
    namespace = {
        "ExchangeRateItem": ExchangeRateItem,
        "FearGreedItem": FearGreedItem,
        "NetflowDto": NetflowDto
    }
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"Compilation failed: {str(e)}", [f"Compilation error: {str(e)}"], 0.0
    
    decision_func = namespace.get("daily_decision")
    if not decision_func:
        return False, "Missing daily_decision function", ["Code doesn't contain a daily_decision function"], 0.0
    
    # Run simulation with error collection
    logger.info("\n💰 Running Trading Simulation")
    try:
        final_value, value_history, errors = simulate_trading(decision_func, historical_data, fear_greed_data, netflow_data, debug=debug)
        
        # Format result summary
        returns_pct = ((final_value / money_if_never_sell) - 1) * 100
        peak_value = max(value_history)
        drawdown = (peak_value - min(value_history)) / peak_value * 100
        result_summary = (
            f"DCA would have made: {money_if_never_sell} → This strategy: €{final_value:.2f}\n"
            f"Return: {returns_pct:+.2f}%\n"
            f"Peak Value: €{peak_value:.2f}\n"
            f"Max Drawdown: {drawdown:.1f}%"
        )
        
        # Display simulation errors if any
        if errors:
            error_summary = "\nSimulation Errors:\n"
            error_summary += "\n".join([f"- {err}" for err in errors[:10]])
            if len(errors) > 10:
                error_summary += f"\n- Plus {len(errors) - 10} more errors..."
            logger.error(error_summary)
        
        panel_style = "green" if final_value > money_if_never_sell else "red"
        logger.panel(result_summary, title="Simulation Results", style=panel_style)
        logger.info(result_summary)
        
        return True, f"{result_summary}", errors, final_value
    except Exception as e:
        error_msg = f"Simulation failed: {str(e)}"
        return False, error_msg, [error_msg], 0.0

# -----------------------------
# 6. Main Application
# -----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="AI Trading Strategy Generator")
    parser.add_argument("--attempts", type=int, default=1, 
                       help="Maximum number of generation attempts")
    parser.add_argument("--threshold", type=float, default=300000,
                       help="Target final portfolio value (in EUR)")
    parser.add_argument("--test", action="store_true",
                       help="Run validation tests only")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    return parser.parse_args()


def main():
    args = parse_arguments()  
    # Set debug log level if debug flag is enabled
    if args.debug:
        logger.set_level(console_level=LogLevel.DEBUG)
        
    # Test the test function  
    if args.test:
        test_app(simulate_trading)
        return
    
    logger.panel(
        f"📈 AI Trading Bot (with Fear & Greed and Netflow Data)\n"
        f"Target: €{args.threshold:.2f}\n"
        f"Max Attempts: {args.attempts}",
        style="blue"
    )

    # Define a common start and end date for all data sources
    start_date = datetime(2018, 2, 4)
    end_date = datetime(2025, 3, 1)
    # Load historical price data
    raw_data = CoinBaseRepo.CoinBaseRepo.read_csv_to_dict('repos/BTC_EUR.csv')
    historical_data = CoinBaseRepo.CoinBaseRepo.get_exchange_rate_items(start_date, end_date, raw_data)
    
    # Load Fear & Greed data
    fear_greed_data = FearGreedRepo.FearGreedRepo.read_csv_file(start_date, end_date)
    
    # Load Netflow data
    netflow_repo = NetflowRepo.NetflowRepo("repos/BtcNetflowNormalized.csv")
    netflow_data = netflow_repo.get_range(start_date.date(), end_date.date())

    # Determine the baseline (money if always holding)
    def no_do_func(_: List[ExchangeRateItem], __: List[FearGreedItem] = None, ___: List[NetflowDto] = None) -> int:
        return 1

    money_if_never_sell, _, _ = simulate_trading(no_do_func, historical_data, fear_greed_data, netflow_data, debug=False)

    best_value = 0.0
    best_code = ""
    # Now using a list of DTO objects for attempts (each with description, performance, and code)
    attempts = []
    recent_errors = []      # Track errors from recent attempts
    previous_code = None    # Store code that caused errors

    for attempt in range(1, args.attempts + 1):
        logger.rule(f"Attempt {attempt}/{args.attempts}")
        
        try:
            # Determine the mode: exploration on first and every 4th attempt; error_fix if there are recent errors and not exploring
            explore_mode = (attempt % 4 == 0 or attempt == 1)
            if recent_errors and previous_code and not explore_mode:
                mode = "error_fix"
                mode_style = "bright_red"
                logger.info("Mode: ERROR FIXING", style=f"bold {mode_style}")
                logger.error("Errors detected - switching to error fixing mode")
            else:
                mode = "exploring" if explore_mode else "refinement"
                mode_style = "bright_yellow" if explore_mode else "bright_magenta"
                mode_display = "EXPLORATION" if explore_mode else "REFINEMENT"
                if explore_mode and recent_errors:
                    logger.info(f"Mode: {mode_display} (overriding error fix)", style=f"bold {mode_style}")
                else:
                    logger.info(f"Mode: {mode_display}", style=f"bold {mode_style}")
            
            if recent_errors:
                logger.error("Passing previous errors to LLM for correction:")
                for error in recent_errors[:5]:
                    logger.error(f"- {error}")
                if len(recent_errors) > 5:
                    logger.error(f"- Plus {len(recent_errors) - 5} more errors...")
            
            # Call generate_trading_strategy using the new parameters:
            code = generate_trading_strategy(
                money_if_never_sell,
                attempts=attempts,
                attempt_num=attempt,
                previous_errors=recent_errors,
                previous_code=previous_code,
                mode=mode,
                debug=args.debug
            )
            
            # Reset errors and update previous_code for the next iteration
            previous_code = code
            recent_errors = []
            
            # Extract strategy description from the code (assumes a line starting with "# Strategy:")
            strategy_desc = "Unknown"
            for line in code.split('\n'):
                if '# Strategy:' in line:
                    strategy_desc = line.replace('# Strategy:', '').strip()
                    break
                    
            logger.info(f"Approach: {strategy_desc}", style="yellow")
            
            # Validate and simulate the generated strategy
            valid, message, errors, current_value = validate_strategy(money_if_never_sell, code, historical_data, fear_greed_data, netflow_data, debug=True)
            
            if errors:
                recent_errors = errors
                logger.warning(f"Found {len(errors)} errors to fix in next attempt")
            else:
                previous_code = None  # Clear previous_code if the simulation had no errors
            
            # Append the current attempt as a DTO to the attempts list
            attempts.append({
                "description": strategy_desc,
                "performance": current_value,
                "code": code
            })
            
            if valid and current_value > best_value:
                best_value = current_value
                best_code = code
                mode_display = "ERROR FIXING" if mode == "error_fix" else ("EXPLORATION" if mode == "exploring" else "REFINEMENT")
                logger.panel(
                    f"New Best Strategy: €{best_value:.2f}\n"
                    f"Approach: {strategy_desc}\n"
                    f"Mode: {mode_display}",
                    style="bold green"
                )
                # Save the best strategy immediately when a new record is hit
                with open("best_strategy.py", "w") as f:
                    f.write(best_code)
                logger.success("Saved best strategy to 'best_strategy.py'")

            if best_value >= args.threshold:
                logger.success(f"We made it :) Reached {best_value}")
                break

        except Exception as e:
            logger.panel(f"Error: {str(e)}", title="Runtime Error", style="red")
            recent_errors = [f"System error: {str(e)}"]
    
    # Final results
    if best_value > 0:
        logger.panel(f"🏆 Best Result: €{best_value:.2f}\n", style="green")
    else:
        logger.panel("❌ Failed to generate valid strategy", style="bold red")
        sys.exit(1)

if __name__ == "__main__":
    main()
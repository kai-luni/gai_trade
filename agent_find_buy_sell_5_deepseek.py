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
model_name = "deepseek-ai/DeepSeek-R1-fast"

# -----------------------------
# 2. Core Domain Objects
# -----------------------------

# -----------------------------
# 3. Data Preparation
# -----------------------------

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
    
    # Update the base prompt to include Netflow data
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
        " - price: Optional[float] (Bitcoin price at that time, may be None)\n"
        " - aggregated_exchanges: Optional[float] (Amount of Bitcoin flowing in/out of exchanges, may be None)\n"
        "   Positive values indicate BTC flowing INTO exchanges (bearish - people are preparing to sell)\n"
        "   Negative values indicate BTC flowing OUT OF exchanges (bullish - people are withdrawing to hold)\n"
        "4. All input lists are chronological - exchange_rates[-1], fear_greed_data[-1], and netflow_data[-1] represent today\n"
        "5. The data is checked to be complete, all fields have values. The data is going at least 30 days back, mostly more.\n"
        
        "=== OUTPUT REQUIREMENTS ===\n"
        "Return ONLY 0/1/2 integer. NO OTHER OUTPUT:\n"
        "- 0: Hold | 1: Buy today | 2: Sell today\n"
        "NOTE: The algorithm automatically adds 100 EUR in BTC on account. \n\n"
        
        "=== INNOVATION CHALLENGE ===\n"
        "🚀 BREAK THE MOLD! Traditional trading strategies often fail in crypto markets. Develop something that defies conventional wisdom.\n"
        "🧠 Consider these unconventional approaches (or invent your own):\n"
        " - Pattern interruption: Trade AGAINST clear patterns when specific conditions are met\n"
        " - Meta-indicators: Create indicators that measure the reliability of other indicators\n"
        " - Behavioral economics: Model market psychology, not just prices\n"
        " - Contrarian timing: Buy when BOTH technicals AND sentiment suggest selling (and vice versa)\n"
        " - Multi-timeframe adaptive parameters: Dynamically adjust thresholds based on volatility regimes\n"
        " - Sentiment-price divergence: Identify when market sentiment doesn't match price action\n"
        " - 'Hidden' signal amplifiers: Identify when minor signals are more significant than major ones\n"
        " - Incorporate cyclical patterns beyond standard cycles (consider human psychology patterns)\n"
        " - Noise-filtering techniques that identify true signals amid market chaos\n\n"
        
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
        "4. DEFY CONVENTIONAL TRADING WISDOM! The most profitable strategies often appear counterintuitive at first glance.\n"
        "5. You can use sub functions to create sophisticated calculations\n"
        
        "=== CODING CONSTRAINTS ===\n"
        "1. FUNCTION NAME MUST BE 'daily_decision' EXACTLY\n"
        "2. DO NOT IMPORT MODULES or REDEFINE CLASSES\n"
        "3. Use List Comprehensions for data extraction:\n"
        " closing_prices = [er.close for er in exchange_rates]\n"
        " fear_greed_values = [fg.index for fg in fear_greed_data]\n"
        " netflows = [nf.aggregated_exchanges for nf in netflow_data if nf.aggregated_exchanges is not None]\n"
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
            if attempt_num == 1:
                strategy_message = "\nThis is your first attempt. Create a creative, innovative trading strategy."
            else:
                strategy_message = (
                    f"\nThis is attempt #{attempt_num}. This is an EXPLORATION attempt."
                    f"\nYour task is to create a COMPLETELY DIFFERENT strategy than before."
                    f"\nBe innovative and experimental - develop a unique decision algorithm."
                    f"\nYou have complete freedom in your approach:"
                    f"\n1. You DO NOT need to use all available data fields - feel free to focus on specific indicators or data that you think are most relevant."
                    f"\n2. Try fundamentally different approaches than what's been tried before."
                    f"\n3. Consider unconventional indicators, unique combinations, or completely new approaches."
                    f"\n4. You can ignore some data fields entirely if it leads to a clearer or more effective strategy."
                    f"\n5. Think outside the box and create something novel - the goal is to try approaches that are very different from previous attempts."
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
                if len(strategy_with_performance) > 600:
                    strategy_with_performance = strategy_with_performance[:597] + "..."
                prior_approaches.append(strategy_with_performance)
        
        # Different feedback approach based on mode
        if mode == "refinement":            
            if last_strategy["description"]:
                # Add important context about the performance
                performance_context = ""
                if abs(int(last_strategy["performance"]) - int(money_if_never_sell)) <= 1:
                    performance_context = (
                        f"\nIMPORTANT: This strategy performed at effectively €{money_if_never_sell}, which means it never executed any trades "
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
                
                # Use previous_code in refinement mode
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
                    "content": (
                        f"I've tried these approaches previously:\n{approach_summary}\n\n"
                        f"Please create a trading strategy that uses a COMPLETELY DIFFERENT approach than these. "
                        f"Feel free to ignore some data fields if they don't fit your approach. "
                        f"You don't need to use all of: exchange rates, fear greed data, and netflow data. "
                        f"It's better to have a focused strategy using only some data fields than to force-fit all data."
                    )
                })
                
                # Add most recent attempt but focus on what didn't work
                performance_line = ""
                for line in last_strategy["code"].split('\n'):
                    if "Final:" in line:
                        performance_line = line
                        break
                
                performance_note = f"\nPerformance: {performance_line}. DCA would have made: {money_if_never_sell} Euro."
                    
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
            temperature = min(0.95, 0.6 + (attempt_num * 0.1))  # Significantly higher for more creativity
        
        # Final user instruction
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
                    f"Make sure it's fundamentally different from previous attempts. "
                    f"Remember, you don't need to use all available data - focus on what you think will work best. "
                    f"Using fewer data sources but in a more focused way is perfectly valid."
                )
            })
    
    # Debug output for prompt history
    if debug:
        # Set debug log level if debug flag is enabled
        logger.set_level(console_level=LogLevel.DEBUG)
        
        debug_output = f"\nMode: {mode} (Temperature: {temperature:.2f})\n"
        for idx, msg in enumerate(messages, 1):
            debug_output += f"\n{idx}. {msg['role'].upper()}:\n"
            content = msg['content'].replace("\n", "\n ")
            debug_output += f" {content}\n"
        
        logger.panel(
            debug_output,
            title=f"DEBUG - ATTEMPT {attempt_num}/{max_attempts} ({mode})",
            border_style="bright_blue",
            padding=(1, 2),
            width=140
        )
    
    # Log LLM interaction (replaces previous file logging)
    prompt_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])
    
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
            
            # Log success
            logger.success(f"API call successful on attempt {retry + 1}")
            
            # Log the LLM interaction
            logger.log_llm_interaction(prompt_text, cleaned_code)
            
            if debug:
                logger.panel(cleaned_code, title="Final Output", style="green")
            
            # Successful API call, return the response
            return cleaned_code
            
        except Exception as e:
            # Log error
            error_msg = f"API error on retry {retry + 1}/{max_retries}: {str(e)}"
            logger.warning(error_msg)
            
            # Check if we've reached max retries
            if retry == max_retries - 1:
                # If this was our last attempt, exit application
                fatal_error = f"FATAL ERROR: Maximum retries ({max_retries}) exceeded. Last error: {str(e)}"
                logger.critical(fatal_error)
                raise RuntimeError(fatal_error)  # Raise exception to terminate execution
            
            # Exponential backoff with jitter
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
                        if today_nf and today_nf.aggregated_exchanges is not None:
                            flow_dir = "IN" if today_nf.aggregated_exchanges > 0 else "OUT"
                            nf_info = f", Netflow: {abs(today_nf.aggregated_exchanges):.2f} BTC {flow_dir}"
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
            
            current_value = portfolio['cash'] + (portfolio['btc'] * today.close)
            portfolio['history'].append(current_value)

    
    return portfolio['history'][-1], portfolio['history'], errors

# 4. Update the validate_strategy function to pass fear & greed data
def validate_strategy(money_if_never_sell: float, code: str, historical_data: list[ExchangeRateItem], fear_greed_data: list[FearGreedItem], netflow_data: list[NetflowDto] = None, debug: bool = False) -> tuple[bool, str, list]:
    """Validate and backtest strategy; returns simulation outcome with errors if any"""
    namespace = {
        "ExchangeRateItem": ExchangeRateItem,
        "FearGreedItem": FearGreedItem,
        "NetflowDto": NetflowDto
    }
    
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"Compilation failed: {str(e)}", [f"Compilation error: {str(e)}"]
    
    decision_func = namespace.get("daily_decision")
    if not decision_func:
        return False, "Missing daily_decision function", ["Code doesn't contain a daily_decision function"]

    # Run simulation with error collection
    logger.info("\n💰 Running Trading Simulation")
    try:
        final_value, value_history, errors = simulate_trading(decision_func, historical_data, fear_greed_data, netflow_data, debug=debug)
        
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
            error_summary = "\nSimulation Errors:\n"
            error_summary += "\n".join([f"- {err}" for err in errors[:10]])
            if len(errors) > 10:
                error_summary += f"\n- Plus {len(errors) - 10} more errors..."
            logger.error(error_summary)
            
        panel_style = "green" if final_value > money_if_never_sell else "red"
        logger.panel(result_summary, title="Simulation Results", style=panel_style)
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
    parser.add_argument("--threshold", type=float, default=100000,
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
        
    # test the test function  
    if args.test:
        test_app()
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
    # Convert to ExchangeRateItem objects
    historical_data = CoinBaseRepo.CoinBaseRepo.get_exchange_rate_items(
        start_date,  # Broad initial filter
        end_date, 
        raw_data
    )
    
    # Load Fear & Greed data
    fear_greed_data = FearGreedRepo.FearGreedRepo.read_csv_file(start_date, end_date)
    
    # Load Netflow data
    netflow_repo = NetflowRepo.NetflowRepo("repos/ITB_btc_netflows.csv")  # Adjust path as needed
    netflow_data = netflow_repo.get_range(start_date.date(), end_date.date())

    # find out how much money we would have if we always hold (buy btc)
    def no_do_func(data: List[ExchangeRateItem], fear_greed_data: List[FearGreedItem] = None, netflow_data: List[NetflowDto] = None) -> int:
        return 1

    money_if_never_sell, _, _ = simulate_trading(no_do_func, historical_data, fear_greed_data, netflow_data, debug=False)

    best_value = 0.0
    best_code = ""
    feedback_history = []
    recent_errors = []      # Track errors from recent attempts
    previous_code = None   # Store code that caused errors

    for attempt in range(1, args.attempts + 1):
        logger.rule(f"Attempt {attempt}/{args.attempts}")
        
        try:
            # Determine if this should be an exploration attempt (every 4th attempt or first attempt)
            explore_mode = (attempt % 4 == 0 or attempt == 1)

            # Check for errors, but only switch to error_fix if it's not an exploration attempt
            if recent_errors and previous_code and not explore_mode:
                mode = "error_fix"
                mode_style = "bright_red"
                logger.info(f"Mode: ERROR FIXING", style=f"bold {mode_style}")
                logger.error("Errors detected - switching to error fixing mode")
            else:
                # Either no errors or it's an exploration attempt that overrides error fixing
                mode = "exploring" if explore_mode else "refinement"
                mode_style = "bright_yellow" if explore_mode else "bright_magenta"
                mode_display = "EXPLORATION" if explore_mode else "REFINEMENT"
                
                if explore_mode and recent_errors:
                    logger.info(f"Mode: {mode_display} (overriding error fix)", style=f"bold {mode_style}")
                else:
                    logger.info(f"Mode: {mode_display}", style=f"bold {mode_style}")
            
            # If there were errors in previous attempt, display them
            if recent_errors:
                logger.error("Passing previous errors to LLM for correction:")
                for error in recent_errors[:5]:  # Show first 5 errors
                    logger.error(f"- {error}")
                if len(recent_errors) > 5:
                    logger.error(f"- Plus {len(recent_errors) - 5} more errors...")
            
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
                    
            logger.info(f"Approach: {strategy_desc}", style="yellow")
            
            # Validate and simulate
            valid, message, errors = validate_strategy(money_if_never_sell, code, historical_data, fear_greed_data, netflow_data, debug=True)
            
            # Store errors for next attempt if needed
            if errors:
                recent_errors = errors
                logger.warning(f"Found {len(errors)} errors to fix in next attempt")
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
        logger.panel(
            f"🏆 Best Result: €{best_value:.2f}\n"
            f"Saving strategy to 'best_strategy.py'",
            style="green"
        )
        with open("best_strategy.py", "w") as f:
            f.write(best_code)
    else:
        logger.panel(
            "❌ Failed to generate valid strategy",
            style="bold red"
        )
        sys.exit(1)

# 6. Update the test function to include Fear & Greed data
def test_app():
    # Generate 30 days of September dummy data (2023-09-01 to 2023-09-30)
    exchange_rates = []
    fear_greed_items = []
    netflow_items = []
    start_date = datetime(2023, 9, 1)
    
    # Set debug level for test
    logger.set_level(console_level=LogLevel.DEBUG)
    
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
        
        # Create netflow item with alternating flow direction
        netflow_value = 100.0 if day % 3 == 0 else -100.0  # Alternate between inflow and outflow
        
        nf_item = NetflowDto(
            date_time=current_date,
            price=25000.0,
            aggregated_exchanges=netflow_value
        )
        
        netflow_items.append(nf_item)
    
    # Add our test days (October 1-3 with significant price moves)
    test_prices = [
        (datetime(2023, 10, 1), 1000.0, 30, "Fear", 150.0),
        (datetime(2023, 10, 2), 2000.0, 15, "Extreme Fear", -200.0),
        (datetime(2023, 10, 3), 3000.0, 60, "Greed", 80.0),
    ]
    
    for date, price, fg_index, fg_text, netflow in test_prices:
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
        
        # Add netflow item
        nf_item = NetflowDto(
            date_time=date,
            price=price,
            aggregated_exchanges=netflow
        )
        
        netflow_items.append(nf_item)
    
    # Define test decision function that uses all data types
    def decision_func(data: List[ExchangeRateItem], fear_greed_data: List[FearGreedItem] = None, netflow_data: List[NetflowDto] = None) -> int:
        current_day = data[-1].date
        
        # Use both price, sentiment and netflow data if available
        info_str = ""
        
        # Check fear & greed data
        if fear_greed_data and len(fear_greed_data) > 0:
            today_fg = next((fg for fg in fear_greed_data if fg.date.date() == current_day.date()), None)
            if today_fg:
                info_str += f" (F&G: {today_fg.index} - {today_fg.index_text})"
                # Example of using sentiment: Buy in extreme fear
                if today_fg.index < 20 and current_day == datetime(2023, 10, 2):
                    logger.success(f"Buying on Extreme Fear{info_str}")
                    return 1
        
        # Check netflow data
        if netflow_data and len(netflow_data) > 0:
            today_nf = next((nf for nf in netflow_data if nf.date_time.date() == current_day.date()), None)
            if today_nf and today_nf.aggregated_exchanges is not None:
                flow_dir = "IN" if today_nf.aggregated_exchanges > 0 else "OUT"
                info_str += f" (Netflow: {abs(today_nf.aggregated_exchanges):.1f} BTC {flow_dir})"
                
                # Example of using netflow: Sell when large inflows to exchanges (bearish)
                if today_nf.aggregated_exchanges > 100 and current_day == datetime(2023, 10, 1):
                    logger.error(f"Selling on high exchange inflow{info_str}")
                    return 2
                
                # Example of using netflow: Buy when large outflows from exchanges (bullish)
                if today_nf.aggregated_exchanges < -150 and current_day == datetime(2023, 10, 2):
                    logger.success(f"Buying on high exchange outflow{info_str}")
                    return 1
        
        if current_day == datetime(2023, 10, 1):
            logger.warning(f"Attempt to sell on Oct 1{info_str} (no BTC held yet)")
            return 2  # Attempt to sell on Oct 1 (no BTC held yet)
        elif current_day == datetime(2023, 10, 2):
            logger.success(f"Buy on Oct 2 at 2000{info_str}")
            return 1  # Buy on Oct 2 at 2000
        elif current_day == datetime(2023, 10, 3):
            logger.info(f"Hold on Oct 3{info_str}")
            return 0  # Hold on Oct 3
        
        logger.debug(f"Default hold{info_str}")
        return 0  # Default: hold
    
    # Run the simulation with debug output
    final_value, history, errors = simulate_trading(
        decision_func=decision_func,
        exchange_rates=exchange_rates,
        fear_greed_data=fear_greed_items,
        netflow_data=netflow_items,
        debug=True
    )
    
    logger.info("\nTest Results:")
    logger.info(f"Expected final value: €1500.00")
    logger.info(f"Actual final value:   €{final_value:.2f}")
    logger.debug(f"Value history: {[round(v, 2) for v in history]}")
    
    if errors:
        logger.error("\nSimulation Errors:")
        for error in errors:
            logger.error(f"- {error}")

if __name__ == "__main__":
    main()
# Load environment variables from .env
from dotenv import load_dotenv
import os
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
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from openai import AzureOpenAI  # Azure-specific client
from repos import CoinBaseRepo  # Import our repository module

warnings.simplefilter("ignore", DeprecationWarning)

# -----------------------------
# 2. Initialize Azure OpenAI Client
# -----------------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-12-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
model_name = "gpt-4o"  # Use the gpt-4o model

# Initialize rich console for pretty printing
console = Console()

# -----------------------------
# 3. Core Domain Objects
# -----------------------------
class ExchangeRateItem:
    def __init__(self, unix: float, date: datetime, low: float, high: float, 
                 open: float, close: float, volume: float):
        self.unix = unix
        self.date = date
        self.low = low
        self.high = high
        self.open = open
        self.close = close
        self.volume = volume

# -----------------------------
# 4. Data Loading Functions
# -----------------------------
def get_repo_exchange_rate_items() -> list:
    """Read exchange rate items from the repository"""
    start_filter = datetime(2018, 12, 31)
    end_filter = datetime(2024, 1, 7)
    print(f"Loading BTC/EUR data from {start_filter} to {end_filter}...")
    dictlist_btc = CoinBaseRepo.CoinBaseRepo.read_csv_to_dict('repos/BTC_EUR.csv')
    sample_exchange_rates = CoinBaseRepo.CoinBaseRepo.get_exchange_rate_items(
        start_filter, end_filter, dictlist_btc
    )
    print(f"Loaded {len(sample_exchange_rates)} BTC/EUR entries.")
    return sample_exchange_rates

# -----------------------------
# 5. LLM Interaction Functions
# -----------------------------
def generate_simulate_investing_function(feedback: str = "") -> str:
    """Generate Python function using Azure OpenAI with full history context"""
    # Base prompt preserved exactly as provided
    base_prompt = (
        "Write a Python function named 'simulate_investing' that analyzes cryptocurrency exchange rate data. "
        "Requirements:\n"
        "1. Input: List of ExchangeRateItem objects with these fields:\n"
        " - date: datetime object (convert to ISO string with .isoformat())\n"
        " - close: closing price (use for calculations)\n"
        " - high: daily high price\n"
        " - low: daily low price\n"
        " - open: daily opening price\n"
        " - volume: trading volume\n"
        "2. Output: Tuple (buy_date, sell_date) as ISO 8601 strings\n"
        "3. Strategy must:\n"
        " a) Start with # Strategy: comment explaining the approach\n"
        " b) Ensure sell_date > buy_date\n"
        " c) Use exchange_rates[i].date.isoformat() for conversion\n"
        " d) Handle empty/insufficient data (raise ValueError)\n"
        "4. Trading rules:\n"
        " - Your goal is to maximize profit between any two dates\n"
        " - Consider daily closing prices (close field)\n"
        " - You must buy before selling\n"
        " - Choose dates from the given data (no interpolation)\n"
        "\n"
        "Return ONLY the function code with Strategy comment. No explanations. "
        "Never use markdown formatting. Never include test code or examples."
    )

    # Initialize messages with base prompt as system message
    messages = [{
        "role": "system",
        "content": base_prompt + "\nRules:\n1. Return ONLY Python code\n2. No markdown\n3. Include # Strategy comment"
    }]

    # Add feedback history if provided
    if feedback:
        attempts = feedback.split("----------------------------")
        for attempt in attempts:
            if attempt.strip():
                messages.extend([
                    {
                        "role": "user",
                        "content": f"Previous attempt details:\n{attempt.strip()}"
                    },
                    {
                        "role": "assistant",
                        "content": "Understood. I'll improve based on these results."
                    }
                ])

    # Add final instruction
    messages.append({
        "role": "user",
        "content": "Generate the improved Python function according to the requirements."
    })

    # Get and clean response
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )
    
    code = response.choices[0].message.content
    return code.replace("```python", "").replace("```", "").strip()

# -----------------------------
# 5. LLM Interaction Functions
# -----------------------------
def generate_simulate_investing_function(feedback: str = "", debug: bool = False) -> str:
    """Generate Python function using Azure OpenAI with full history context"""
    base_prompt = (
        "Write a Python function named 'simulate_investing' that analyzes cryptocurrency exchange rate data. "
        "Requirements:\n"
        "1. Input: List of ExchangeRateItem objects with these fields:\n"
        " - date: datetime object (convert to ISO string with .isoformat())\n"
        " - close: closing price (use for calculations)\n"
        " - high: daily high price\n"
        " - low: daily low price\n"
        " - open: daily opening price\n"
        " - volume: trading volume\n"
        "2. Output: Tuple (buy_date, sell_date) as ISO 8601 strings\n"
        "3. Strategy must:\n"
        " a) Start with # Strategy: comment explaining the approach\n"
        " b) Ensure sell_date > buy_date\n"
        " c) Use exchange_rates[i].date.isoformat() for conversion\n"
        " d) Handle empty/insufficient data (raise ValueError)\n"
        "4. Trading rules:\n"
        " - Your goal is to maximize profit between any two dates\n"
        " - Consider daily closing prices (close field)\n"
        " - You must buy before selling\n"
        " - Choose dates from the given data (no interpolation)\n"
        "\n"
        "Return ONLY the function code with Strategy comment. No explanations. "
        "Never use markdown formatting. Never include test code or examples."
    )
    messages = [{
        "role": "system",
        "content": base_prompt + "\nRules:\n1. Return ONLY Python code\n2. No markdown\n3. Include # Strategy comment"
    }]
    # Use a unique delimiter to split feedback into complete entries instead of per-line
    if feedback:
        attempts = feedback.split("----------------------------")
        for attempt in attempts:
            if attempt.strip():
                messages.extend([
                    {
                        "role": "user",
                        "content": f"Previous attempt details:\n{attempt.strip()}"
                    },
                    {
                        "role": "assistant",
                        "content": "Understood. I'll improve based on these results."
                    }
                ])
    messages.append({
        "role": "user",
        "content": "Generate the improved Python function according to the requirements."
    })
    if debug:
        debug_output = "[bold cyan]LLM Prompt History:[/]\n"
        for idx, msg in enumerate(messages, 1):
            debug_output += f"\n[bold]{idx}. {msg['role'].title()}:[/]\n{msg['content']}\n"
        console.print(Panel.fit(debug_output, title="DEBUG MODE - FULL PROMPT LOG", style="blue"))
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )
    code = response.choices[0].message.content
    return code.replace("python", "").replace("", "").strip()


# -----------------------------
# 6. Testing Framework
# -----------------------------
def test_generated_function(code: str, exchange_rates: list = None) -> tuple[bool, float]:
    """Test the generated trading function"""
    namespace = {"ExchangeRateItem": ExchangeRateItem}
    
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"Syntax Error: {str(e)}"

    if "simulate_investing" not in namespace:
        return False, "Missing simulate_investing function"
    
    strategy = next((line.split(":", 1)[1].strip() 
                    for line in code.split("\n") 
                    if line.startswith("# Strategy:")), "No strategy described")

    try:
        if exchange_rates is None:
            exchange_rates = get_repo_exchange_rate_items()
            
        buy_date, sell_date = namespace["simulate_investing"](exchange_rates)
        buy_dt = datetime.fromisoformat(buy_date)
        sell_dt = datetime.fromisoformat(sell_date)
    except Exception as e:
        return False, f"Runtime Error: {str(e)}"

    try:
        buy_item = next(r for r in exchange_rates if r.date.date() == buy_dt.date())
        sell_item = next(r for r in exchange_rates if r.date.date() == sell_dt.date())
        if buy_dt >= sell_dt:
            return False, "Sell date must be after buy date"
    except StopIteration:
        return False, "Invalid dates - no matching data found"

    final_value = (100 / buy_item.close) * sell_item.close
    return True, final_value

def run_integration_tests():
    """Run validation tests with sample data"""
    test_data = [
        ExchangeRateItem(
            unix=1,
            date=datetime.fromisoformat("2023-01-01T00:00:00"),
            low=900, high=1100,
            open=1000, close=1000,
            volume=1
        ),
        ExchangeRateItem(
            unix=2,
            date=datetime.fromisoformat("2023-01-02T00:00:00"),
            low=4900, high=5100,
            open=5000, close=5000,
            volume=1
        ),
        ExchangeRateItem(
            unix=3,
            date=datetime.fromisoformat("2023-01-03T00:00:00"),
            low=9900, high=10100,
            open=10000, close=10000,
            volume=1
        ),
    ]

    test_code = """# Strategy: Buy first item, sell last item
def simulate_investing(exchange_rates):
    try:
        buy_date = exchange_rates[0].date.isoformat()
        sell_date = exchange_rates[-1].date.isoformat()
    except IndexError:
        raise ValueError("Insufficient data")
    return (buy_date, sell_date)
"""

    console.print(Panel.fit("Running Integration Tests", style="cyan"))
    
    # Test 1: Successful case
    success, result = test_generated_function(test_code, test_data)
    if success and abs(result - 1000) < 0.01:
        console.print("[green]✓ Basic test passed")
    else:
        console.print(f"[red]✗ Basic test failed (got {result})")

    # Test 2: Error handling
    error_code = "def simulate_investing(_): return ('invalid', 'date')"
    success, result = test_generated_function(error_code, test_data)
    if not success and "invalid" in str(result):
        console.print("[green]✓ Error handling works")
    else:
        console.print(f"[red]✗ Error handling failed")

# -----------------------------
# 7. Main Application
# -----------------------------
def parse_arguments():
    """Configure command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Trading Assistant - Generate and Test Strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Number of generation attempts"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run integration tests instead of generation"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2000,
        help="Minimum profit threshold for success"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full LLM prompts and debugging information"
    )
    return parser.parse_args()
def main():
    """Main execution flow"""
    args = parse_arguments()

    if args.test:
        return run_integration_tests()

    console.print(Panel.fit(
        f"[bold]💸 AI Trading Assistant[/]\n"
        f"Attempts: {args.attempts}\n"
        f"Target: €{args.threshold:.2f}",
        style="blue"
    ))

    success = False
    feedback_history = []

    for attempt in range(1, args.attempts + 1):
        console.rule(f"[bold]Attempt {attempt}/{args.attempts}")
        
        # Generate code with accumulated feedback
        code = generate_simulate_investing_function(
            "\n".join(feedback_history),
            debug=args.debug
        )
        console.print(Panel(code, title="Generated Strategy"))
        
        # Test the strategy
        valid, result = test_generated_function(code)
        
        # Create detailed feedback entry
        result_str = f"€{result:.2f}" if valid else f"Error: {result}"
        feedback_entry = (
            f"Attempt {attempt} Result: {result_str}\n"
            f"Generated Code:\n{code}\n"
            f"----------------------------"
        )
        feedback_history.append(feedback_entry)

        if valid and result >= args.threshold:
            console.print(Panel.fit(
                f"[green]💰 Profit Target Achieved!\nFinal Value: €{result:.2f}[/]",
                style="green"
            ))
            success = True
            break
            
        console.print(Panel(
            f"Result: {result_str}",
            title="Progress",
            style="yellow"
        ))

    if not success:
        console.print(Panel.fit(
            "[red]❌ Failed to achieve profit target[/]\n" +
            "\n".join(feedback_history[-3:]),
            style="red"
        ))
        sys.exit(1)

# -----------------------------
# 8. Entry Point
# -----------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Operation cancelled by user[/]")
        sys.exit(1)
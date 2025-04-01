# Initialize logger
from datetime import datetime, timedelta
import time
from typing import List
from dto.ExchangeRateItem import ExchangeRateItem
from dto.NetflowDto import NetflowDto
from dto.ObjectsGai import FearGreedItem
from helper.LoggerPrime import LogLevel, LoggerPrime


logger = LoggerPrime(
    name="trading_bot",
    log_file="trading_bot.log"
)


def test_app(simulate_trading):
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
            aggregated_exchanges=netflow_value,
            aggregated_exchanges_normalized=1.0
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
            aggregated_exchanges=netflow,
            aggregated_exchanges_normalized=1.0
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
            if today_nf:
                flow_dir = "IN" if today_nf.aggregated_exchanges_normalized > 0 else "OUT"
                info_str += f" (Netflow: {abs(today_nf.aggregated_exchanges):.1f} BTC {flow_dir})"
                
                # Example of using netflow: Sell when large inflows to exchanges (bearish)
                if today_nf.aggregated_exchanges_normalized > 0.6 and current_day == datetime(2023, 10, 1):
                    logger.error(f"Selling on high exchange inflow{info_str}")
                    return 2
                
                # Example of using netflow: Buy when large outflows from exchanges (bullish)
                if today_nf.aggregated_exchanges_normalized < -0.6 and current_day == datetime(2023, 10, 2):
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
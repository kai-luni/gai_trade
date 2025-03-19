from datetime import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class NetflowDto:
    """
    Data Transfer Object for Netflow data.
    Represents cryptocurrency exchange flow data for a specific date.
    """
    date_time: datetime
    price: Optional[float] = None
    aggregated_exchanges: Optional[float] = None
    binance: Optional[float] = None
    bitfinex: Optional[float] = None
    bitstamp: Optional[float] = None
    coinbase: Optional[float] = None
    gemini: Optional[float] = None
    huobi: Optional[float] = None
    kraken: Optional[float] = None
    luno: Optional[float] = None
    okex: Optional[float] = None
    poloniex: Optional[float] = None

    @classmethod
    def from_csv_row(cls, row: dict) -> 'NetflowDto':
        """
        Create a NetflowDto instance from a CSV row dictionary.
        
        Args:
            row: Dictionary containing CSV row data
            
        Returns:
            NetflowDto instance
        """
        # Parse the date time string to datetime object
        date_time = datetime.fromisoformat(row['DateTime'].replace('Z', '+00:00'))
        
        # Helper function to safely convert values to float
        def safe_float(value):
            if value is None or value == '' or value == '**':
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        
        # Convert data to appropriate types, handling missing values
        return cls(
            date_time=date_time,
            price=safe_float(row.get('Price')),
            aggregated_exchanges=safe_float(row.get('Aggregated Exchanges')),
            binance=safe_float(row.get('Binance')),
            bitfinex=safe_float(row.get('Bitfinex')),
            bitstamp=safe_float(row.get('Bitstamp')),
            coinbase=safe_float(row.get('Coinbase')),
            gemini=safe_float(row.get('Gemini')),
            huobi=safe_float(row.get('Huobi')),
            kraken=safe_float(row.get('**Kraken')),
            luno=safe_float(row.get('**Luno')),
            okex=safe_float(row.get('Okex')),
            poloniex=safe_float(row.get('Poloniex'))
        )
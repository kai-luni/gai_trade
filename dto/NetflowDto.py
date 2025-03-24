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
    aggregated_exchanges_normalized: float = None
    aggregated_exchanges: float = None

    @classmethod
    def from_csv_row(cls, row: dict) -> 'NetflowDto':
        """
        Create a NetflowDto instance from a CSV row dictionary.
        
        Args:
            row: Dictionary containing CSV row data
            
        Returns:
            NetflowDto instance
            
        Raises:
            ValueError: If any of the required values (DateTime, Aggregated Exchanges, 
                    or AggrExchNormalized) is missing or cannot be converted to the proper type
        """
        # Check if DateTime exists in the row
        if 'DateTime' not in row or not row['DateTime']:
            raise ValueError("Missing required field: DateTime")
            
        # Parse the date time string to datetime object
        try:
            date_time = datetime.fromisoformat(row['DateTime'].replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid DateTime format: {row.get('DateTime')}")
        
        # Helper function to safely convert values to float, throwing an exception on None
        def safe_float(value, field_name):
            if value is None or value == '' or value == '**':
                raise ValueError(f"Missing required field: {field_name}")
            try:
                return float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid numeric value for {field_name}: {value}")
        
        # Convert data to appropriate types, raising exceptions for missing values
        aggregated_exchanges = safe_float(row.get('Aggregated Exchanges'), 'Aggregated Exchanges')
        aggregated_exchanges_normalized = safe_float(row.get('AggrExchNormalized'), 'AggrExchNormalized')
        
        # All required values are present and properly converted
        return cls(
            date_time=date_time,
            aggregated_exchanges=aggregated_exchanges,
            aggregated_exchanges_normalized=aggregated_exchanges_normalized
        )
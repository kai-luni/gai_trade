import csv
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple, Set
import os
from collections import defaultdict

import pandas as pd

from dto.NetflowDto import NetflowDto



class DuplicateDateError(Exception):
    """Exception raised when multiple entries exist for the same date."""
    pass


class DateNotFoundError(Exception):
    """Exception raised when a requested date entry does not exist."""
    pass


class NetflowRepo:
    """
    Repository for managing cryptocurrency netflow data.
    Provides methods to load data from CSV and query by date range.
    """
    
    def __init__(self, csv_file_path: str):
        """
        Initialize the repository with data from a CSV file.
        
        Args:
            csv_file_path: Path to the CSV file containing netflow data
        
        Raises:
            FileNotFoundError: If the CSV file does not exist
            DuplicateDateError: If multiple entries exist for the same date
        """
        self.csv_file_path = csv_file_path
        self._validate_file_exists()
        self.data_by_date: Dict[date, NetflowDto] = {}
        self._load_data()
    
    def _validate_file_exists(self) -> None:
        """
        Check if the CSV file exists.
        
        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.isfile(self.csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
    
    def _load_data(self) -> None:
        """
        Load data from CSV file into memory.
        
        Raises:
            DuplicateDateError: If multiple entries exist for the same date
        """
        date_count = defaultdict(int)
        
        with open(self.csv_file_path, 'r', newline='') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                dto = NetflowDto.from_csv_row(row)
                entry_date = dto.date_time.date()
                
                # Check for duplicate dates
                date_count[entry_date] += 1
                if date_count[entry_date] <= 1:
                    self.data_by_date[entry_date] = dto
                
                
    
    def get_range(self, from_date: date, to_date: date, raise_if_missing: bool = False) -> List[NetflowDto]:
        """
        Get netflow data for a specific date range (inclusive).
        
        Args:
            from_date: Start date
            to_date: End date
            raise_if_missing: If True, raises DateNotFoundError when any date in the range has no data
            
        Returns:
            List of NetflowDto objects within the specified range, sorted by date
            
        Raises:
            ValueError: If from_date is later than to_date
            DateNotFoundError: If raise_if_missing is True and any date in the range has no data
        """
        if from_date > to_date:
            raise ValueError("from_date must be less than or equal to to_date")
        
        result = []
        current_date = from_date
        missing_dates = []
        
        while current_date <= to_date:
            if current_date in self.data_by_date:
                result.append(self.data_by_date[current_date])
            elif raise_if_missing:
                missing_dates.append(current_date)
            current_date = self._next_day(current_date)
        
        if raise_if_missing and missing_dates:
            raise DateNotFoundError(f"No data found for dates: {', '.join(str(d) for d in missing_dates)}")
        
        # Sort by datetime
        return sorted(result, key=lambda x: x.date_time)
    
    def get_by_date(self, query_date: date, raise_if_not_found: bool = True) -> Optional[NetflowDto]:
        """
        Get netflow data for a specific date.
        
        Args:
            query_date: The date to query for
            raise_if_not_found: If True, raises DateNotFoundError when the date doesn't exist
            
        Returns:
            NetflowDto for the specified date, or None if not found and raise_if_not_found is False
            
        Raises:
            DateNotFoundError: If the date doesn't exist and raise_if_not_found is True
        """
        result = self.data_by_date.get(query_date)
        if result is None and raise_if_not_found:
            raise DateNotFoundError(f"No data found for date: {query_date}")
        return result
    
    def get_all(self) -> List[NetflowDto]:
        """
        Get all netflow data.
        
        Returns:
            List of all NetflowDto objects, sorted by date
        """
        return sorted(self.data_by_date.values(), key=lambda x: x.date_time)
    
    @staticmethod
    def normalize_exchanges_by_year(csv_file, output_file='repos/BtcNetflowNormalized.csv'):
        """
        Reads a CSV file, extracts DateTime and Aggregated Exchanges columns,
        creates a new column AggrExchNormalized by normalizing the Aggregated Exchanges
        to a range of -1 to 1 for each year, and writes the result to a new CSV file.
        
        Parameters:
        -----------
        csv_file : str
            Path to the input CSV file
        output_file : str, optional
            Path to the output CSV file (default: 'BtcNetflowNormalized.csv')
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Read the CSV file
            # For troubleshooting, explicitly set the encoding to utf-8
            df = pd.read_csv(csv_file, header=0, encoding='utf-8')
            print(f"Successfully read {len(df)} rows from {csv_file}")
            
            # Debug: print first few rows to verify data
            print("First few rows of input data:")
            print(df.head())
            
            # Clean column names (remove ** if present and strip whitespace)
            df.columns = [col.replace('*', '').strip() for col in df.columns]
            
            # Extract only the DateTime and Aggregated Exchanges columns
            # Use explicit column names to avoid issues with spaces
            date_col = 'DateTime'
            agg_exch_col = 'Aggregated Exchanges'
            
            # Verify the columns exist in the dataframe
            if date_col not in df.columns or agg_exch_col not in df.columns:
                print(f"Error: Required columns not found. Available columns: {df.columns.tolist()}")
                return False
                
            # Extract only the columns we need
            result_df = df[[date_col, agg_exch_col]].copy()
            
            # Debug: check if extraction worked
            print(f"Extracted {len(result_df)} rows with DateTime and Aggregated Exchanges")
            print(result_df.head())
            
            # Convert DateTime to datetime format
            result_df[date_col] = pd.to_datetime(result_df[date_col])
            
            # Extract year from DateTime
            result_df['Year'] = result_df[date_col].dt.year
            
            # Initialize the normalized column
            result_df['AggrExchNormalized'] = 0.0
            
            # The sign convention for Aggregated Exchanges:
            # Positive values indicate BTC flowing INTO exchanges (bearish - preparing to sell)
            # Negative values indicate BTC flowing OUT OF exchanges (bullish - withdrawing to hold)
            
            # Process each year separately
            years = result_df['Year'].unique()
            print(f"Processing data for {len(years)} different years: {years}")
            
            for year in years:
                year_mask = result_df['Year'] == year
                year_data = result_df.loc[year_mask]
                
                print(f"Processing year {year} with {len(year_data)} rows")
                
                # Get positive and negative values separately
                pos_mask = year_data[agg_exch_col] > 0
                neg_mask = year_data[agg_exch_col] < 0
                zero_mask = year_data[agg_exch_col] == 0
                
                pos_data = year_data.loc[pos_mask]
                neg_data = year_data.loc[neg_mask]
                zero_data = year_data.loc[zero_mask]
                
                print(f"Year {year}: {len(pos_data)} positive values, {len(neg_data)} negative values, {sum(zero_mask)} zeros")
                
                # Process positive values (normalize to 0 to 1)
                if len(pos_data) > 0:
                    pos_min = pos_data[agg_exch_col].min()
                    pos_max = pos_data[agg_exch_col].max()
                    
                    if pos_max > pos_min:
                        # Multiple positive values - normalize to range [0, 1]
                        result_df.loc[pos_data.index, 'AggrExchNormalized'] = (
                            (pos_data[agg_exch_col] - pos_min) / (pos_max - pos_min)
                        )
                    else:
                        # Only one positive value or all are the same
                        result_df.loc[pos_data.index, 'AggrExchNormalized'] = 1.0
                
                # Process negative values (normalize to -1 to 0)
                if len(neg_data) > 0:
                    neg_min = neg_data[agg_exch_col].min()
                    neg_max = neg_data[agg_exch_col].max()
                    
                    if neg_max > neg_min:
                        # Multiple negative values - normalize to range [-1, 0]
                        result_df.loc[neg_data.index, 'AggrExchNormalized'] = (
                            -1.0 + (neg_data[agg_exch_col] - neg_min) / (neg_max - neg_min)
                        )
                    else:
                        # Only one negative value or all are the same
                        result_df.loc[neg_data.index, 'AggrExchNormalized'] = -1.0
                
                # Zero values remain as 0.0
                if len(zero_data) > 0:
                    result_df.loc[zero_data.index, 'AggrExchNormalized'] = 0.0
            
            # Drop the temporary Year column
            result_df = result_df.drop(columns=['Year'])
            
            # Write the result to the output CSV file
            result_df.to_csv(output_file, index=False)
            print(f"Successfully wrote {len(result_df)} rows to {output_file}")
            print("First few rows of output:")
            print(result_df.head())
            
            return True
            
        except Exception as e:
            print(f"Error in normalize_exchanges_by_year: {str(e)}")
            # Print stack trace for debugging
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def _next_day(d: date) -> date:
        """Helper method to get the next day."""
        from datetime import timedelta
        return d + timedelta(days=1)
    

if __name__ == "__main__":
    # Replace 'data.csv' with your actual CSV file path
    success = NetflowRepo.normalize_exchanges_by_year('repos/ITB_btc_netflows.csv')
    
    if success:
        print("Data successfully written to BtcNetflowNormalized.csv")
    else:
        print("Failed to process and write data")
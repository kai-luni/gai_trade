import csv
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple, Set
import os
from collections import defaultdict

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
    def _next_day(d: date) -> date:
        """Helper method to get the next day."""
        from datetime import timedelta
        return d + timedelta(days=1)
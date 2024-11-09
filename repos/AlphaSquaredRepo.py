import csv
from datetime import datetime, timedelta
from typing import List

from dto.RiskItem import RiskEntry


class AlphaSquaredRepo:
    """
    A repository for managing and retrieving risk data entries from a CSV file.
    """

    @staticmethod
    def get_risk_entries(from_date: datetime, to_date: datetime, csv_path: str) -> List[RiskEntry]:
        """
        Reads the CSV file and parses each row to create a RiskEntry object for the given date range.

        Parameters:
        ----------
        from_date : datetime
            The start date of the range (inclusive) as a datetime object.
        to_date : datetime
            The end date of the range (inclusive) as a datetime object.
        csv_path : str
            The path to the CSV file containing risk data.

        Returns:
        -------
        List[RiskEntry]
            A list of RiskEntry objects, each representing a date and corresponding risk value from the CSV file.

        Raises:
        ------
        ValueError
            If the CSV file format is incorrect or if any row is missing the 'Date' or 'Risk' field.
        Exception
            If no entry is found within a 7-day window for any date in the range.
        """
        # Ensure the date range is valid
        if from_date > to_date:
            raise ValueError("from_date must be earlier than or equal to to_date")

        # Read the CSV data into a dictionary for quick lookup
        risk_entries_dict = {}
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                try:
                    date = datetime.strptime(row['Date'], '%Y-%m-%d').date()
                    risk = float(row['Risk'])
                    risk_entries_dict[date] = RiskEntry(row['Date'], risk)
                except (KeyError, ValueError) as e:
                    raise ValueError(f"Error parsing CSV row: {row}. Ensure 'Date' and 'Risk' fields are correct.") from e

        # Check for entries within a 7-day window for each date in the range
        result = []
        current_date = from_date.date()
        end_date = to_date.date()

        while current_date <= end_date:
            # Look for an entry within a 7-day window
            closest_entry = None
            min_difference = timedelta(days=7)

            for entry_date in risk_entries_dict.keys():
                difference = abs(current_date - entry_date)
                if difference <= min_difference:
                    closest_entry = risk_entries_dict[entry_date]
                    min_difference = difference

            if closest_entry:
                # Create a new RiskEntry object with the requested date
                updated_entry = RiskEntry(date=current_date.strftime('%Y-%m-%d'), risk=closest_entry.risk)
                result.append(updated_entry)
            else:
                raise Exception(f"No risk entry found within 7 days for date: {current_date}")

            current_date += timedelta(days=1)

        return result

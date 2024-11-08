import csv
from typing import List

from dto.RiskItem import RiskEntry


class AlphaSquaredRepo:
    """
    A repository for managing and retrieving risk data entries from a CSV file.

    Attributes:
    ----------
    csv_path : str
        The file path to the CSV file containing risk data entries.

    Methods:
    -------
    get_risk_entries() -> List[RiskEntry]:
        Reads the CSV file and returns a list of RiskEntry objects, each representing a single risk entry.
    """
    
    def __init__(self, csv_path: str):
        """
        Initializes the AlphaSquaredRepo with a path to the CSV file.

        Parameters:
        ----------
        csv_path : str
            The path to the CSV file containing risk data in 'Date;Risk' format.
        """
        self.csv_path = csv_path

    def get_risk_entries(self) -> List[RiskEntry]:
        """
        Reads the CSV file and parses each row to create a RiskEntry object.

        Returns:
        -------
        List[RiskEntry]
            A list of RiskEntry objects, each representing a date and corresponding risk value from the CSV file.
        
        Raises:
        ------
        ValueError
            If the CSV file format is incorrect or if any row is missing the 'Date' or 'Risk' field.
        """
        risk_entries = []
        with open(self.csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                date = row['Date']
                risk = float(row['Risk'])
                risk_entries.append(RiskEntry(date, risk))
        return risk_entries

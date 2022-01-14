import csv
from datetime import datetime

class FearGreedDataReader:
    def __init__(self):
        self._all_entries = self.initialize()

    def initialize(self, full_path="data/fear_greed.csv"):
        """load all entries from csv into private list

        Args:
            full_path (str, optional): path to csv data file. Defaults to "data/fear_greed.csv".

        Returns:
            list: all entries in a list
        """
        header=["date", "fear_index", "description"]
        rows = []
        with open(full_path) as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                if len(row) != 3:
                    continue
                date_values = row[0].split("-")
                row[0] = datetime(int(date_values[2]), int(date_values[1]), int(date_values[0]))
                row[1] = int(row[1])
                rows.append(row)
        return rows

    def get_entry_for_day(self, day : int, month : int, year : int):
        """find day entry for given date

        Args:
            day (int): day
            month (int): month
            year (int): year

        Returns:
            list: day entry: date, greed fear index, description
        """
        for day_entry in self._all_entries:
            if day_entry[0].day != day:
                continue
            if day_entry[0].month != month:
                continue
            if day_entry[0].year != year:
                continue
            return day_entry
        return None

    



# if __name__ == "__main__":
#     FearGreedDataReader.initialize()
            
from datetime import datetime


class RiskEntry:
    """
    Represents a single entry of risk data.

    Attributes:
    ----------
    date : datetime.date
        The date associated with the risk entry, converted from a string in 'YYYY-MM-DD' format.
    risk : float
        The risk value associated with this entry.

    Methods:
    -------
    __repr__():
        Returns a string representation of the RiskEntry object, showing its date and risk value.
    """
    
    def __init__(self, date: str, risk: float):
        self.date = datetime.strptime(date, '%Y-%m-%d').date()
        self.risk = risk

    def __repr__(self):
        return f"RiskEntry(date={self.date}, risk={self.risk})"
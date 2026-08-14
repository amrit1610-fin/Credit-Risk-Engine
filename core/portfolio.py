from typing import List, Optional
import pandas as pd

class Loan:
    """
    Object-Oriented representation of a single credit facility.
    Accepts a dynamic dictionary of features to support any ML model input.
    """
    def __init__(self, features_dict: dict, lgd: float = 0.40):
        # We store all 60+ Kaggle columns inside this dictionary
        self.features = features_dict
        
        # Risk Metrics (Calculated later by the engines)
        self.pd: Optional[float] = None
        self.lgd: float = lgd 
        
    @property
    def ead(self) -> float:
        """Exposure at Default."""
        # Safely extract loan_amnt from the dynamic features dictionary
        return float(self.features.get('loan_amnt', 0.0))

    @property
    def loan_amnt(self) -> float:
        """Helper property for the ECL Engine output."""
        return float(self.features.get('loan_amnt', 0.0))

    @property
    def int_rate(self) -> float:
        """Helper property for the ECL Engine output."""
        return float(self.features.get('int_rate', 0.0))

    def to_dict(self):
        """Returns the features dictionary directly for ML prediction."""
        return self.features


class Portfolio:
    """
    A collection of Loan objects.
    """
    def __init__(self, loans: List[Loan]):
        self.loans = loans

    @property
    def total_exposure(self) -> float:
        """Returns the sum of Exposure at Default (EAD) for all loans."""
        return sum(loan.ead for loan in self.loans)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the portfolio of Loan objects back into a pandas DataFrame.
        This is required because Scikit-Learn/XGBoost models expect a 2D matrix/dataframe.
        """
        # We extract the dynamic features dictionary from every loan
        return pd.DataFrame([loan.to_dict() for loan in self.loans])
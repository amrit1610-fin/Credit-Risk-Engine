from typing import List, Optional
import pandas as pd
import random

class Loan:
    """
    Object-Oriented representation of a single credit facility.
    Accepts a dynamic dictionary of features to support any ML model input.
    """
    def __init__(self, features_dict: dict):
        # We store all 60+ Kaggle columns inside this dictionary
        self.features = features_dict
        
        # Risk Metrics (Calculated later by the engines)
        self.pd: Optional[float] = None
        self.lgd: Optional[float] = None 
        
    @property
    def ead(self) -> float:
        """Exposure at Default."""
        # Safely extract funded_amnt or loan_amnt from the dynamic features dictionary
        return float(self.features.get('funded_amnt', self.features.get('loan_amnt', 0.0)))

    @property
    def loan_amnt(self) -> float:
        """Helper property for the ECL Engine output."""
        return float(self.features.get('loan_amnt', 0.0))

    @property
    def int_rate(self) -> float:
        """Helper property for the ECL Engine output."""
        return float(self.features.get('int_rate', 0.0))

    @property
    def sector(self) -> str:
        """Extracts or assigns a sector for correlation modeling."""
        # If 'purpose' isn't in our features, we assign a random sector for the Copula math
        purposes = ['Tech', 'Healthcare', 'Consumer', 'Finance']
        purpose = self.features.get('purpose', random.choice(purposes))
        
        # Simple mapping logic if using real LendingClub 'purpose' column
        if isinstance(purpose, str):
            if purpose in ['credit_card', 'debt_consolidation']:
                return 'Consumer'
            elif purpose in ['medical']:
                return 'Healthcare'
            elif purpose in ['small_business', 'renewable_energy']:
                return 'Tech'
        
        # Fallback to random if the purpose column was label-encoded to an integer
        return random.choice(purposes)

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
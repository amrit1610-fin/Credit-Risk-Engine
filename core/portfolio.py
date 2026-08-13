from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class Loan:
    """
    Object-Oriented representation of a single credit facility.
    Contains the financial metrics required for the ML model to predict PD.
    """
    loan_amnt: float
    term: float
    int_rate: float
    installment: float
    emp_length: float
    home_ownership: int      # Encoded
    annual_inc: float
    verification_status: int # Encoded
    purpose: int             # Encoded
    dti: float
    delinq_2yrs: float
    inq_last_6mths: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    
    # Risk Metrics (Calculated later by the engines)
    pd: Optional[float] = None
    lgd: float = 0.40 # Standard Basel assumption for unsecured retail/corporate: 40% loss
    
    @property
    def ead(self) -> float:
        """
        Exposure at Default. For simple term loans, we use the total loan amount.
        (For revolving credit lines, this would include a Credit Conversion Factor).
        """
        return self.loan_amnt

    def to_dict(self):
        """Converts the loan features into a dictionary for ML prediction."""
        return {
            'loan_amnt': self.loan_amnt, 'term': self.term, 'int_rate': self.int_rate,
            'installment': self.installment, 'emp_length': self.emp_length,
            'home_ownership': self.home_ownership, 'annual_inc': self.annual_inc,
            'verification_status': self.verification_status, 'purpose': self.purpose,
            'dti': self.dti, 'delinq_2yrs': self.delinq_2yrs,
            'inq_last_6mths': self.inq_last_6mths, 'open_acc': self.open_acc,
            'pub_rec': self.pub_rec, 'revol_bal': self.revol_bal,
            'revol_util': self.revol_util, 'total_acc': self.total_acc
        }

class Portfolio:
    """
    A collection of Loan objects.
    Provides methods to easily extract data for matrix/ML operations.
    """
    def __init__(self, loans: List[Loan] = None):
        self.loans = loans if loans else []

    def add_loan(self, loan: Loan):
        self.loans.append(loan)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the entire portfolio into a Pandas DataFrame.
        This is perfectly formatted to be passed directly into our XGBoost model.
        """
        return pd.DataFrame([loan.to_dict() for loan in self.loans])

    @property
    def total_exposure(self) -> float:
        return sum(loan.ead for loan in self.loans)
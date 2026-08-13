import joblib
import pandas as pd
import numpy as np
import os

from core.portfolio import Portfolio, Loan
from models.ecl_calculator import ECLCalculator

class MockPDModel:
    """
    A dummy Machine Learning model that mimics XGBoost/Scikit-Learn.
    It returns random default probabilities so we can test the ECL math.
    """
    def predict_proba(self, X):
        np.random.seed(42) # Ensure consistent test results
        # Generate a random probability of default (PD) between 2% and 15%
        pd_values = np.random.uniform(0.02, 0.15, size=len(X))
        # predict_proba expects a 2D array: [Probability of 0, Probability of 1]
        return np.column_stack((1 - pd_values, pd_values))

def run_test():
    print("--- 1. Setting up Test Environment ---")
    
    # We save our mock model and features to disk so ECLCalculator can load them
    mock_model = MockPDModel()
    mock_features = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'emp_length',
        'home_ownership', 'annual_inc', 'verification_status', 'purpose',
        'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
        'revol_bal', 'revol_util', 'total_acc'
    ]
    
    joblib.dump(mock_model, 'dummy_pd_model.pkl')
    joblib.dump(mock_features, 'dummy_features.pkl')
    print("Created 'dummy_pd_model.pkl' and 'dummy_features.pkl'")
    
    # ---------------------------------------------------------
    # --- 2. Create Dummy Loans
    # ---------------------------------------------------------
    print("\n--- 2. Building Portfolio ---")
    
    # A "Good" Loan (Low risk, low amount)
    loan1 = Loan(
        loan_amnt=10000.0, term=36, int_rate=0.05, installment=300.0,
        emp_length=5, home_ownership=1, annual_inc=60000.0,
        verification_status=1, purpose=2, dti=15.0, delinq_2yrs=0,
        inq_last_6mths=1, open_acc=5, pub_rec=0, revol_bal=5000.0,
        revol_util=0.3, total_acc=10
    )
    
    # A "Bad" Loan (High risk, high amount, past delinquencies)
    loan2 = Loan(
        loan_amnt=50000.0, term=60, int_rate=0.15, installment=1000.0,
        emp_length=2, home_ownership=0, annual_inc=40000.0,
        verification_status=0, purpose=3, dti=35.0, delinq_2yrs=2,
        inq_last_6mths=4, open_acc=12, pub_rec=1, revol_bal=25000.0,
        revol_util=0.85, total_acc=15
    )
    
    portfolio = Portfolio([loan1, loan2])
    print(f"Created Portfolio with {len(portfolio.loans)} loans.")
    print(f"Total Portfolio Exposure (EAD): ${portfolio.total_exposure:,.2f}")
    
    # ---------------------------------------------------------
    # --- 3. Run the Engine
    # ---------------------------------------------------------
    print("\n--- 3. Running ECL Engine ---")
    engine = ECLCalculator(
        portfolio=portfolio, 
        model_path='dummy_pd_model.pkl', 
        features_path='dummy_features.pkl'
    )
    
    results = engine.calculate_risk()
    
    # ---------------------------------------------------------
    # --- 4. Display Results
    # ---------------------------------------------------------
    print("\n--- 4. Final Risk Results ---")
    print(f"Total Portfolio ECL:    ${results['total_ecl']:,.2f}")
    print(f"Blended ECL Percentage: {results['ecl_percentage']:.2f}%\n")
    
    print("--- Loan Level Detail ---")
    print(results['loan_level_data'].to_string(index=False))

    # Cleanup the dummy files so we don't clutter your workspace
    os.remove('dummy_pd_model.pkl')
    os.remove('dummy_features.pkl')

if __name__ == "__main__":
    run_test()
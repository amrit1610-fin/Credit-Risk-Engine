import joblib
import pandas as pd
import numpy as np
from core.portfolio import Portfolio, Loan
from models.base import RiskEngine

class ECLCalculator(RiskEngine):
    """
    Deterministic engine that calculates Expected Credit Loss (ECL)
    under the IFRS 9 / CECL framework.
    """
    def __init__(self, portfolio: Portfolio, model_path: str, features_path: str):
        super().__init__(portfolio)
        
        # Load the pre-trained XGBoost model and the exact feature list
        print(f"Loading PD Model from {model_path}...")
        self.pd_model = joblib.load(model_path)
        self.model_features = joblib.load(features_path)

    def _predict_pd(self):
        """
        Takes the portfolio, converts it to a dataframe, ensures the columns 
        perfectly match what XGBoost expects, and predicts the Probability of Default.
        """
        # 1. Convert our OOP Portfolio into a matrix (Pandas DataFrame)
        df = self.portfolio.to_dataframe()
        
        # 2. Safety Check: If our current portfolio is missing columns the model needs,
        # we fill them with 0 to prevent the model from crashing. 
        # (This happens if a categorical variable like 'State_WY' wasn't in our current batch)
        for col in self.model_features:
            if col not in df.columns:
                df[col] = 0
                
        # 3. Reorder the columns to perfectly match the training data
        X = df[self.model_features]
        
        # 4. Predict probabilities. predict_proba returns [prob_0, prob_1]. We want prob_1 (Default)
        probabilities = self.pd_model.predict_proba(X)[:, 1]
        
        # 5. Write the probabilities back into our OOP Loan objects
        for i, loan in enumerate(self.portfolio.loans):
            loan.pd = probabilities[i]

    def calculate_risk(self) -> dict:
        """
        Executes the ECL formula: PD * LGD * EAD for every loan.
        """
        if not self.portfolio.loans:
            return {"error": "Portfolio is empty."}

        # Step 1: Run the ML model to populate loan.pd
        self._predict_pd()
        
        total_ecl = 0.0
        loan_level_results = []
        
        # Step 2: Calculate ECL for each loan
        for loan in self.portfolio.loans:
            # The Holy Trinity
            prob_default = loan.pd  # Renamed to avoid shadowing pandas (pd)
            lgd = loan.lgd
            ead = loan.ead
            
            # The IFRS 9 Formula
            ecl = prob_default * lgd * ead
            total_ecl += ecl
            
            loan_level_results.append({
                'loan_amnt': loan.loan_amnt,
                'int_rate': loan.int_rate,
                'pd': prob_default,
                'ecl': ecl
            })
            
        # Create a clean summary dictionary to pass to our Dash UI later
        return {
            "total_exposure": self.portfolio.total_exposure,
            "total_ecl": total_ecl,
            "ecl_percentage": (total_ecl / self.portfolio.total_exposure) * 100 if self.portfolio.total_exposure > 0 else 0,
            "loan_level_data": pd.DataFrame(loan_level_results)
        }
import joblib
import pandas as pd
import numpy as np
from core.portfolio import Portfolio, Loan
from models.base import RiskEngine

class ECLCalculator(RiskEngine):
    """
    Deterministic engine that calculates Expected Credit Loss (ECL)
    under the IFRS 9 / CECL framework using dynamic PD and LGD models.
    """
    def __init__(self, portfolio: Portfolio, 
                 pd_model_path: str, pd_features_path: str, 
                 lgd_model_path: str, lgd_features_path: str
                ):
        super().__init__(portfolio)
        
        # Load the pre-trained XGBoost models and the exact feature lists
        print(f"Loading PD Model from {pd_model_path}...")
        self.pd_model = joblib.load(pd_model_path)
        self.pd_features = joblib.load(pd_features_path)

        print(f"Loading LGD Model from {lgd_model_path}...")
        self.lgd_model = joblib.load(lgd_model_path)
        self.lgd_features = joblib.load(lgd_features_path)

    def _predict_risk_parameters(self):
        """
        Takes the portfolio, converts it to a dataframe, ensures the columns 
        perfectly match what XGBoost expects, and predicts PD and LGD.
        """
        # 1. Convert our OOP Portfolio into a matrix (Pandas DataFrame)
        df = self.portfolio.to_dataframe()
        
        # --- PD PREDICTION ---
        pd_df = df.copy()
        for col in self.pd_features:
            if col not in pd_df.columns:
                pd_df[col] = 0
        X_pd = pd_df[self.pd_features]
        probabilities = self.pd_model.predict_proba(X_pd)[:, 1]

        # --- LGD PREDICTION ---
        lgd_df = df.copy()
        for col in self.lgd_features:
            if col not in lgd_df.columns:
                lgd_df[col] = 0
        X_lgd = lgd_df[self.lgd_features]
        # Predict and cap between 0 and 1
        lgd_predictions = np.clip(self.lgd_model.predict(X_lgd), 0.0, 1.0)
        
        # Write the predictions back into our OOP Loan objects
        for i, loan in enumerate(self.portfolio.loans):
            loan.pd = probabilities[i]
            loan.lgd = lgd_predictions[i]

    def calculate_risk(self) -> dict:
        """
        Executes the ECL formula: PD * LGD * EAD for every loan.
        """
        if not self.portfolio.loans:
            return {"error": "Portfolio is empty."}

        # Step 1: Run the ML models to populate loan.pd and loan.lgd
        self._predict_risk_parameters()
        
        total_ecl = 0.0
        loan_level_results = []
        
        # Step 2: Calculate ECL for each loan
        for loan in self.portfolio.loans:
            # The Holy Trinity
            prob_default = loan.pd  
            lgd = loan.lgd
            ead = loan.ead
            
            # The IFRS 9 Formula
            ecl = prob_default * lgd * ead
            total_ecl += ecl
            
            loan_level_results.append({
                'loan_amnt': loan.loan_amnt,
                'int_rate': loan.int_rate,
                'pd': prob_default,
                'lgd': lgd,
                'ecl': ecl
            })
            
        # Create a clean summary dictionary to pass to our Dash UI later
        return {
            "total_exposure": self.portfolio.total_exposure,
            "total_ecl": total_ecl,
            "ecl_percentage": (total_ecl / self.portfolio.total_exposure) * 100 if self.portfolio.total_exposure > 0 else 0,
            "loan_level_data": pd.DataFrame(loan_level_results)
        }
import pandas as pd
import numpy as np
import os

from core.portfolio import Portfolio, Loan
from models.ecl_calculator import ECLCalculator
from models.copula_simulation import CopulaSimulationEngine

def run_test():
    print("--- 1. Setting up Environment ---")
    
    # Check if necessary files exist
    required_files = ['data/live_portfolio.csv', 
                      'models/pd_model_assets/model.pkl', 'models/pd_model_assets/features.pkl', 
                      'models/lgd_model_assets/model.pkl', 'models/lgd_model_assets/features.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"ERROR: Missing required files: {missing_files}")
        print("Please ensure you have downloaded your models and data from Kaggle.")
        return

    # ---------------------------------------------------------
    # Load Real Data Portfolio
    # ---------------------------------------------------------
    print("\n--- 2. Building Portfolio from live_portfolio.csv ---")
    df = pd.read_csv('data/live_portfolio.csv')
    loans = [Loan(features_dict=row.to_dict()) for _, row in df.iterrows()]
    portfolio = Portfolio(loans)
    
    print(f"Loaded Portfolio with {len(portfolio.loans)} loans.")
    print(f"Total Portfolio Exposure (EAD): ${portfolio.total_exposure:,.2f}")
    
    # ---------------------------------------------------------
    # ECL Engine (PD & LGD Models)
    # ---------------------------------------------------------
    print("\n--- 3. Running Deterministic ECL Engine ---")
    ecl_engine = ECLCalculator(
        portfolio=portfolio, 
        pd_model_path='models/pd_model_assets/model.pkl', 
        pd_features_path='models/pd_model_assets/features.pkl',
        lgd_model_path='models/lgd_model_assets/model.pkl',
        lgd_features_path='models/lgd_model_assets/features.pkl'
    )
    
    results = ecl_engine.calculate_risk()
    
    print("\n--- 4. Final Risk Results ---")
    print(f"Total Portfolio ECL:    ${results['total_ecl']:,.2f}")
    print(f"Blended ECL Percentage: {results['ecl_percentage']:.2f}%\n")
    
    print("--- Loan Level Detail ---")
    print(results['loan_level_data'].to_string(index=False))

    # ---------------------------------------------------------
    # Copula Simulation (Tail Risk with Sectors)
    # ---------------------------------------------------------
    print("\n--- 5. Running Copula Simulation (Tail Risk) ---")
    
    num_loans = len(portfolio.loans)
    sector_correlations = {
        'Tech': {'Tech': 0.8, 'Healthcare': 0.3, 'Consumer': 0.4, 'Finance': 0.5},
        'Healthcare': {'Tech': 0.3, 'Healthcare': 0.7, 'Consumer': 0.2, 'Finance': 0.4},
        'Consumer': {'Tech': 0.4, 'Healthcare': 0.2, 'Consumer': 0.6, 'Finance': 0.6},
        'Finance': {'Tech': 0.5, 'Healthcare': 0.4, 'Consumer': 0.6, 'Finance': 0.9}
    }

    base_rho = 0.3 # Base systemic correlation
    corr_matrix = np.zeros((num_loans, num_loans))
    for i in range(num_loans):
        for j in range(num_loans):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                s_i = portfolio.loans[i].sector
                s_j = portfolio.loans[j].sector
                # Blend user baseline with sector specific correlation
                corr_matrix[i, j] = (base_rho + sector_correlations[s_i][s_j]) / 2
    
    sim_engine = CopulaSimulationEngine(
        portfolio=portfolio,
        correlation_matrix=corr_matrix,
        num_simulations=10000
    )
    
    sim_results = sim_engine.calculate_risk()
    
    print("\n--- 6. Portfolio Tail Risk Metrics ---")
    print(f"Expected Loss (Baseline): ${sim_results['expected_loss']:,.2f}")
    print(f"99% Value at Risk (VaR):  ${sim_results['var_99']:,.2f}")
    print(f"97.5% Expected Shortfall: ${sim_results['expected_shortfall']:,.2f}")
    print(f"Max Simulated Loss:       ${sim_results['max_simulated_loss']:,.2f}\n")

if __name__ == "__main__":
    run_test()
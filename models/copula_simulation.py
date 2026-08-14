import numpy as np
import pandas as pd
from scipy.stats import norm
from core.portfolio import Portfolio
from models.base import RiskEngine

class CopulaSimulationEngine(RiskEngine):
    """
    Monte Carlo engine using a Gaussian Copula to simulate correlated defaults.
    Calculates Portfolio Value at Risk (VaR) and Expected Shortfall (ES).
    """
    def __init__(self, portfolio: Portfolio, correlation_matrix: np.ndarray, num_simulations: int = 10000):
        super().__init__(portfolio)
        self.correlation_matrix = correlation_matrix
        self.num_simulations = num_simulations

    def calculate_risk(self) -> dict:
        """
        Executes the Copula simulation to find extreme tail risks.
        """
        if not self.portfolio.loans:
            return {"error": "Portfolio is empty."}
            
        num_loans = len(self.portfolio.loans)
        
        # Ensure the correlation matrix matches the portfolio size
        if self.correlation_matrix.shape != (num_loans, num_loans):
            # If not, default to an identity matrix (independent defaults)
            print("Warning: Correlation matrix size mismatch. Assuming independent defaults.")
            self.correlation_matrix = np.eye(num_loans)

        # 1. Extract PDs and EADs
        pds = np.array([loan.pd for loan in self.portfolio.loans])
        eads = np.array([loan.ead for loan in self.portfolio.loans])
        lgds = np.array([loan.lgd for loan in self.portfolio.loans])
        
        # Ensure PDs are not zero or one to avoid infinity in inverse CDF
        pds = np.clip(pds, 1e-6, 1 - 1e-6)

        # 2. Map PDs to standard normal thresholds
        # If PD is 5%, norm.ppf(0.05) gives the z-score cut-off (-1.645)
        default_thresholds = norm.ppf(pds)

        # 3. Perform Cholesky Decomposition on the correlation matrix
        # This allows us to generate correlated random numbers
        try:
            L = np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            # Fallback if matrix is not positive-definite
            print("Warning: Correlation matrix is not positive-definite. Falling back to independent shocks.")
            L = np.eye(num_loans)

        # 4. Generate independent random normal shocks (Z) for all simulations
        # Shape: (num_simulations, num_loans)
        Z = np.random.standard_normal((self.num_simulations, num_loans))

        # 5. Apply Cholesky matrix to create correlated shocks (X)
        # X = Z * L^T
        correlated_shocks = Z.dot(L.T)

        # 6. Evaluate Defaults
        # A loan defaults if its shock is LESS THAN its threshold
        # default_events is a boolean matrix of shape (num_simulations, num_loans)
        default_events = correlated_shocks < default_thresholds

        # 7. Calculate Portfolio Losses for each simulation
        # Loss = Default Event (0 or 1) * LGD * EAD
        # We sum across the columns (loans) to get the total portfolio loss per simulation
        portfolio_losses = np.sum(default_events * lgds * eads, axis=1)

        # 8. Sort the losses from smallest to largest to find the tail
        sorted_losses = np.sort(portfolio_losses)

        # Calculate 99% Value at Risk (VaR)
        var_99_index = int(self.num_simulations * 0.99)
        var_99 = sorted_losses[var_99_index]

        # Calculate 97.5% Expected Shortfall (ES / CVaR)
        # Average of all losses worse than the 97.5% VaR threshold
        var_975_index = int(self.num_simulations * 0.975)
        expected_shortfall = np.mean(sorted_losses[var_975_index:])

        # Baseline Expected Loss (for reference against VaR)
        expected_loss = np.sum(pds * lgds * eads)

        return {
            "expected_loss": expected_loss,
            "var_99": var_99,
            "expected_shortfall": expected_shortfall,
            "max_simulated_loss": sorted_losses[-1],
            # We return the full array of losses so Dash can plot a histogram
            "simulated_loss_distribution": portfolio_losses 
        }
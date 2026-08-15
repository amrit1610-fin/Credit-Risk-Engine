<div align="center">

# 🏦 Quantitative Credit Risk & Portfolio Stress Testing Engine

An institutional-grade, object-oriented Credit Risk system built in Python. This engine calculates deterministic **Expected Credit Loss (ECL)** under the *IFRS 9* framework using Machine Learning, and utilizes a Monte Carlo Gaussian Copula simulation to calculate tail risks (VaR & Expected Shortfall) for correlated loan portfolios.

</div>

### 🚀 Live Interactive Dashboard
Experience the engine directly in your browser: [Launch Dash Application](https://credit-risk-engine-jmi9.onrender.com/)
(Note: Hosted on Render's free tier. The server may take 45-60 seconds to "wake up" upon initial load).

### 🌟 Key Features
**Machine Learning Risk Parameters**: Utilizes XGBoost models to predict Probability of Default (PD) and Loss Given Default (LGD) based on borrower financial metrics.

**IFRS 9 / CECL Framework**: Calculates baseline Expected Credit Loss (ECL = PD × LGD × EAD) at the facility level.

**Gaussian Copula Simulation**: Emplements Cholesky Decomposition to simulate thousands of correlated macroeconomic scenarios, accurately modeling systemic tail risks where multiple companies default simultaneously.

**Tail Risk Metrics**: Calculates 99% Value at Risk (VaR) and 97.5% Expected Shortfall (ES/CVaR) on the simulated loss distributions.

**Object-Oriented Architecture**: Strict decoupling of Financial Instruments (the Loan object), Machine Learning wrappers, and Mathematical Engines, mimicking Tier-1 bank tech stacks.

**Reactive UI**: A professional, "Dark Glass" interactive dashboard built with Plotly Dash, featuring risk concentration scatter plots and dynamic loss histograms.

### 🧠 The Mathematical & Data Architecture

*1. The Machine Learning Pipelines*

To prevent Data Leakage (a common pitfall in credit risk modeling), the training data pipeline uses a strict "Y-Split" architecture:
    1. PD Model (Classification): Trained on the entire portfolio to predict the binary default event, explicitly masking post-default data (like recoveries).
    2. LGD Model (Regression): Trained only on defaulted loans to predict the continuous recovery rate severity.

*2. The Copula Simulation Engine*
While individual expected losses are deterministic, portfolio risk is driven by correlation.
The engine maps ML-predicted PDs to standard normal Z-score thresholds.
A covariance matrix is constructed based on sectoral overlap.
Geometric Brownian Motion (GBM) / standard normal shocks are generated and multiplied by the Cholesky factor of the correlation matrix.
If a facility's correlated shock breaches its specific Z-score threshold, a default event is registered for that simulation path.

### 📂 Project Structure

```
Credit-Risk-Engine/
│
├── core/                   
│   ├── __init__.py
│   └── portfolio.py               # OOP Classes: 'Loan' and 'Portfolio' (EAD logic)
│
├── data/                 
│   ├── live_portfolio.py          # test csv 
│
├── models/                 
│   ├── __init__.py
│   ├── lgd_model_assets           # contains model and features for LGD
|   ├── pd_model_assets            # contains model and features for PD
│   ├── base.py                    # Abstract Base Class 'RiskEngine'
│   ├── ecl_calculator.py          # Deterministic IFRS 9 Engine (PD * LGD * EAD)
│   └── copula_simulation.py       # Monte Carlo Gaussian Copula Engine (VaR & ES)
│
├── app.py                         # Dash Interactive Web Dashboard
├── main.py                        # CLI Execution & Testing Script
└── requirements.txt               # Deployment dependencies
```

### ⚙️ Installation & Local Usage
To run this engine locally on your machine, follow these steps:
1. Clone the repository
git clone https://github.com/[YOUR_GITHUB_USERNAME]/Credit-Risk-Engine.git
cd Credit-Risk-Engine


2. Install dependencies It is recommended to use a Python virtual environment.
pip install -r requirements.txt


3. Run the CLI Engine To output the portfolio metrics and simulation results directly to your terminal:
python main.py


4. Launch the Web Dashboard To boot up the interactive Dash application locally:
python app.py


Navigate to http://127.0.0.1:8050/ in your browser.
👨‍💻 Author
Amrit
Quantitative Finance Enthusiast | Software Developer
GitHub Profile

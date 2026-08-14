import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os

# Import our custom risk models
from core.portfolio import Portfolio, Loan
from models.ecl_calculator import ECLCalculator
from models.copula_simulation import CopulaSimulationEngine

# --- MOCK ML MODEL SETUP (For seamless testing) ---
class MockPDModel:
    def predict_proba(self, X):
        np.random.seed(42)
        pd_values = np.random.uniform(0.01, 0.15, size=len(X))
        return np.column_stack((1 - pd_values, pd_values))

def setup_mock_environment():
    """Creates dummy files if real models aren't present yet."""
    if not os.path.exists('pd_model.pkl'):
        joblib.dump(MockPDModel(), 'pd_model.pkl')
        mock_features = ['loan_amnt', 'term', 'int_rate', 'installment', 'emp_length',
                         'home_ownership', 'annual_inc', 'verification_status', 'purpose',
                         'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
                         'revol_bal', 'revol_util', 'total_acc']
        joblib.dump(mock_features, 'model_features.pkl')

def generate_dummy_portfolio():
    """Generates a diverse portfolio of 15 loans for visualization."""
    np.random.seed(42)
    loans = []
    for _ in range(15):
        l = Loan(
            loan_amnt=np.random.uniform(10000, 100000),
            term=np.random.choice([36, 60]),
            int_rate=np.random.uniform(0.05, 0.20),
            installment=np.random.uniform(200, 2000),
            emp_length=np.random.randint(0, 10),
            home_ownership=np.random.randint(0, 3),
            annual_inc=np.random.uniform(40000, 150000),
            verification_status=np.random.randint(0, 2),
            purpose=np.random.randint(0, 5),
            dti=np.random.uniform(5, 35),
            delinq_2yrs=np.random.randint(0, 2),
            inq_last_6mths=np.random.randint(0, 3),
            open_acc=np.random.randint(5, 20),
            pub_rec=0,
            revol_bal=np.random.uniform(5000, 50000),
            revol_util=np.random.uniform(0.1, 0.9),
            total_acc=np.random.randint(10, 40)
        )
        loans.append(l)
    return Portfolio(loans)

setup_mock_environment()

# --- DASH APP INITIALIZATION ---
# Using the DARKLY theme for a professional quant/trading desk look
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Credit Risk & Copula Engine"

# Custom CSS for the slider to fix the invisible text issue
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Force the slider marks (text) to be white */
            .rc-slider-mark-text {
                color: white !important;
            }
            /* Make dropdown text visible */
            .Select-value-label {
                color: #333 !important;
            }
            .Select-placeholder {
                color: #888 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- UI LAYOUT ---
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H2("Quantitative Credit Risk Engine", className="text-primary mt-3 mb-4"), width=12)
    ]),

    dbc.Row([
        # SIDEBAR: Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stress Test Parameters", className="fw-bold text-white bg-dark border-bottom"),
                dbc.CardBody([
                    html.Label("Systemic Correlation (ρ)", className="text-white fw-bold mb-2"),
                    dcc.Slider(
                        id='correlation-slider',
                        min=0.0, max=0.9, step=0.05, value=0.3,
                        marks={0: '0.0', 0.5: '0.5', 0.9: '0.9'},
                        className="mb-4"
                    ),
                    
                    html.Label("Monte Carlo Simulations", className="text-white fw-bold mb-2 mt-2"),
                    dcc.Dropdown(
                        id='num-simulations',
                        options=[
                            {'label': '1,000 Paths', 'value': 1000},
                            {'label': '10,000 Paths', 'value': 10000},
                            {'label': '50,000 Paths (Slow)', 'value': 50000}
                        ],
                        value=10000,
                        clearable=False,
                        className="mb-4"
                    ),
                    
                    dbc.Button(
                        "Run Copula Stress Test", 
                        id="run-btn", 
                        color="primary", 
                        className="w-100 fw-bold mt-2"
                    )
                ], className="bg-secondary") # Darker background for the body
            ], className="mb-4 shadow-sm border-secondary"),
            
            # Exposure Summary Card
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Portfolio Exposure", className="text-light"),
                    html.H3(id="total-exposure-text", className="text-white mb-0 fw-bold")
                ], className="bg-secondary")
            ], className="shadow-sm border-secondary")
            
        ], width=3),

        # MAIN CONTENT: Metrics and Charts
        dbc.Col([
            # Top Metrics Row
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Expected Credit Loss (ECL)", className="text-info"),
                    html.H3(id="ecl-text", className="mb-0")
                ]), className="shadow-sm"), width=4),
                
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("99% Value at Risk (VaR)", className="text-warning"),
                    html.H3(id="var-text", className="mb-0")
                ]), className="shadow-sm"), width=4),
                
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("97.5% Expected Shortfall (ES)", className="text-danger"),
                    html.H3(id="es-text", className="mb-0")
                ]), className="shadow-sm"), width=4),
            ], className="mb-4"),

            # Charts Row
            dbc.Row([
                # Loss Distribution Histogram (The Copula Output)
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Portfolio Loss Distribution (Monte Carlo)", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="loss-distribution-chart"))
                ], className="shadow-sm"), width=12)
            ], className="mb-4"),
            
            dbc.Row([
                # Individual Loan ECL Breakdown
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Loan-Level Expected Loss", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="loan-ecl-chart"))
                ], className="shadow-sm"), width=12)
            ])
            
        ], width=9)
    ])
], fluid=True, className="p-4")

# --- CALLBACKS (The Reactive Logic) ---
@app.callback(
    [Output("total-exposure-text", "children"),
     Output("ecl-text", "children"),
     Output("var-text", "children"),
     Output("es-text", "children"),
     Output("loss-distribution-chart", "figure"),
     Output("loan-ecl-chart", "figure")],
    [Input("run-btn", "n_clicks")],
    [State("correlation-slider", "value"),
     State("num-simulations", "value")]
)
def update_dashboard(n_clicks, correlation_rho, n_simulations):
    # 1. Generate/Load Portfolio
    portfolio = generate_dummy_portfolio()
    
    # 2. Run Deterministic ECL Engine
    ecl_engine = ECLCalculator(
        portfolio=portfolio,
        model_path='pd_model.pkl',
        features_path='model_features.pkl'
    )
    ecl_results = ecl_engine.calculate_risk()
    
    # 3. Run Stochastic Copula Engine
    num_loans = len(portfolio.loans)
    # Build a uniform correlation matrix based on user slider input
    corr_matrix = np.full((num_loans, num_loans), correlation_rho)
    np.fill_diagonal(corr_matrix, 1.0)
    
    copula_engine = CopulaSimulationEngine(
        portfolio=portfolio,
        correlation_matrix=corr_matrix,
        num_simulations=n_simulations
    )
    tail_results = copula_engine.calculate_risk()
    
    # --- Format Outputs ---
    total_exposure_str = f"${ecl_results['total_exposure']:,.0f}"
    total_ecl_str = f"${ecl_results['total_ecl']:,.0f}"
    var_str = f"${tail_results['var_99']:,.0f}"
    es_str = f"${tail_results['expected_shortfall']:,.0f}"
    
    # --- Build Histogram (Loss Distribution) ---
    losses = tail_results['simulated_loss_distribution']
    var_99 = tail_results['var_99']
    es_975 = tail_results['expected_shortfall']
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=losses, 
        nbinsx=100, 
        marker_color='#1f77b4',
        name='Simulated Losses'
    ))
    
    # Calculate y-axis limit to place annotations nicely
    hist_counts, _ = np.histogram(losses, bins=100)
    max_count = np.max(hist_counts)
    
    # Add VaR Line (Staggered lower)
    fig_hist.add_vline(x=var_99, line_dash="dash", line_color="orange")
    fig_hist.add_annotation(
        x=var_99, y=max_count * 0.8,
        text="99% VaR",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="orange",
        ax=-40,
        ay=-20,
        font=dict(color="orange", size=12, weight="bold")
    )

    # Add ES Line (Staggered higher)
    fig_hist.add_vline(x=es_975, line_dash="dash", line_color="red")
    fig_hist.add_annotation(
        x=es_975, y=max_count * 0.95,
        text="97.5% ES",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        ax=40,
        ay=-30,
        font=dict(color="red", size=12, weight="bold")
    )
    
    fig_hist.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Portfolio Loss ($)",
        yaxis_title="Frequency (Simulations)",
        font=dict(color="white")
    )
    
    # --- Build Bar Chart (Loan-Level ECL) ---
    df_loans = ecl_results['loan_level_data']
    # Give them simple IDs for the X-axis
    df_loans['Loan ID'] = [f"Loan {i+1}" for i in range(len(df_loans))]
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_loans['Loan ID'],
        y=df_loans['ecl'],
        marker_color='#17becf',
        text=df_loans['ecl'].apply(lambda x: f"${x:,.0f}"),
        textposition='auto',
        textfont=dict(color="white")
    ))
    
    fig_bar.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Loan Facility",
        yaxis_title="Expected Credit Loss ($)",
        font=dict(color="white")
    )

    return total_exposure_str, total_ecl_str, var_str, es_str, fig_hist, fig_bar

if __name__ == '__main__':
    # Run the server on port 8050
    app.run(debug=True, port=8050)
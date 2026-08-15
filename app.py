import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# Import our custom risk models
from core.portfolio import Portfolio, Loan
from models.ecl_calculator import ECLCalculator
from models.copula_simulation import CopulaSimulationEngine

# --- LIVE DATA LOADER ---
def load_live_portfolio(csv_path='./data/live_portfolio.csv'):
    if not os.path.exists(csv_path):
        # Fallback to local directory if not in /data
        if os.path.exists('./live_portfolio.csv'):
            csv_path = './live_portfolio.csv'
        else:
            print(f"Warning: live_portfolio.csv not found. Returning empty portfolio.")
            return Portfolio([])
        
    df = pd.read_csv(csv_path)
    loans = []
    for _, row in df.iterrows():
        loans.append(Loan(features_dict=row.to_dict()))
    return Portfolio(loans)

# --- DASH APP INITIALIZATION ---
# Load Bootstrap Darkly and Inter font for a professional look
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.DARKLY,
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap"
])
app.title = "Credit Risk & Copula Engine"

# --- PROFESSIONAL DARK THEME CSS ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* 1. Global Modern Background */
            body {
                background-color: #0b0f19 !important; /* Deep space blue/black */
                color: #e5e7eb !important;
                font-family: 'Inter', sans-serif !important;
                -webkit-font-smoothing: antialiased;
            }
            
            /* 2. Smooth, Rounded Cards */
            .smooth-card {
                background-color: #111827 !important; /* Soft dark gray/blue */
                border-radius: 16px;
                border: 1px solid #1f2937 !important;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
                padding: 24px;
                height: 100%;
            }
            .smooth-header {
                font-size: 1.05rem;
                font-weight: 600;
                color: #9ca3af;
                margin-bottom: 20px;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }
            .kpi-value {
                font-size: 2.2rem;
                font-weight: 700;
                letter-spacing: -1px;
            }

            /* 3. Slider Styling */
            .rc-slider-track { background-color: #3b82f6 !important; }
            .rc-slider-handle { border: solid 2px #3b82f6 !important; background-color: #111827 !important; opacity: 1 !important; }
            
            /* 4. TRUE Dark Mode Dropdown Fix */
            .Select-control { 
                background-color: #1f2937 !important; 
                border: 1px solid #374151 !important; 
                border-radius: 8px !important; 
            }
            .Select-value-label { color: #f3f4f6 !important; } /* White text for selection */
            .Select-menu-outer { 
                background-color: #1f2937 !important; 
                border: 1px solid #374151 !important; 
                border-radius: 8px !important; 
                margin-top: 4px;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            }
            .Select-option { background-color: #1f2937 !important; color: #f3f4f6 !important; }
            .Select-option.is-focused { background-color: #374151 !important; }
            .Select-option.is-selected { background-color: #4b5563 !important; }
            .Select-placeholder { color: #9ca3af !important; }
            .Select-arrow { border-top-color: #9ca3af !important; }
            
            /* 5. Gradient Button */
            .run-btn-custom {
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
                border: none !important;
                border-radius: 8px !important;
                color: white !important;
                font-weight: 600 !important;
                padding: 12px !important;
                transition: all 0.2s ease;
                box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
            }
            .run-btn-custom:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
            }
            .run-btn-custom:active {
                transform: translateY(0);
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
    # Page Header
    dbc.Row([
        dbc.Col(
            html.H2("Quantitative Risk Engine", 
                    style={'color': '#f3f4f6', 'fontWeight': '700', 'paddingTop': '20px', 'paddingBottom': '10px'}), 
            width=12)
    ]),

    # Top Row: KPI Cards
    dbc.Row([
        dbc.Col(html.Div([
            html.Div("Total Exposure", className="smooth-header"),
            html.Div(id="total-exposure-text", className="kpi-value", style={'color': '#3b82f6'}) # Blue
        ], className="smooth-card"), width=3),
        
        dbc.Col(html.Div([
            html.Div("Expected Credit Loss", className="smooth-header"),
            html.Div(id="ecl-text", className="kpi-value", style={'color': '#10b981'}) # Green
        ], className="smooth-card"), width=3),
        
        dbc.Col(html.Div([
            html.Div("99% Value at Risk", className="smooth-header"),
            html.Div(id="var-text", className="kpi-value", style={'color': '#f59e0b'}) # Yellow
        ], className="smooth-card"), width=3),
        
        dbc.Col(html.Div([
            html.Div("97.5% Expected Shortfall", className="smooth-header"),
            html.Div(id="es-text", className="kpi-value", style={'color': '#ef4444'}) # Red
        ], className="smooth-card"), width=3),
    ], className="mb-4"),

    # Middle Row: Controls, Distribution, and Sector Donut
    dbc.Row([
        # Left Panel: Controls
        dbc.Col(html.Div([
            html.Div("Simulation Controls", className="smooth-header"),
            
            html.Label("Systemic Correlation (ρ)", className="mb-2", style={'color': '#d1d5db', 'fontSize': '14px'}),
            dcc.Slider(
                id='correlation-slider',
                min=0.0, max=0.9, step=0.05, value=0.3,
                marks={
                    0: {'label': '0.0', 'style': {'color': '#9ca3af'}}, 
                    0.5: {'label': '0.5', 'style': {'color': '#9ca3af'}}, 
                    0.9: {'label': '0.9', 'style': {'color': '#9ca3af'}}
                },
                tooltip={
                    "placement": "bottom",
                    "always_visible": False,
                    "style": {
                        "backgroundColor": "#1f2937",
                        "color": "#f9fafb",
                        "border": "1px solid #374151",
                        "borderRadius": "6px",
                        "boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.5)"
                    }
                },
                className="mb-4"
            ),
            
            html.Label("Monte Carlo Simulations", className="text-light fw-bold mb-2 mt-2"),
            dbc.Select(
                id='num-simulations',
                options=[
                    {'label': '1,000 Paths', 'value': 1000},
                    {'label': '10,000 Paths', 'value': 10000},
                    {'label': '50,000 Paths (Slow)', 'value': 50000}
                ],
                value=10000,
                className="mb-4"
            ),
            
            html.Br(),
            dbc.Button("Run Stress Test", id="run-btn", className="w-100 mt-2 run-btn-custom")
            
        ], className="smooth-card"), width=3),

        # Center Panel: Loss Distribution Histogram
        dbc.Col(html.Div([
            html.Div("Portfolio Loss Distribution", className="smooth-header"),
            html.Div("Portfolio Loss Distribution", className="smooth-header"),
            dcc.Graph(id="loss-distribution-chart", style={'height': '320px'})
        ], className="smooth-card"), width=6),
        
        # Right Panel: Sector Breakdown Donut
        dbc.Col(html.Div([
            html.Div("ECL by Sector", className="smooth-header"),
            dcc.Graph(id="sector-donut-chart", style={'height': '320px'})
        ], className="smooth-card"), width=3),
        
    ], className="mb-4"),
    
    # Bottom Row: Loan Details and Risk Concentration
    dbc.Row([
        # Left Panel: Bar chart of ECL per loan
        dbc.Col(html.Div([
            html.Div("Facility-Level Expected Loss", className="smooth-header"),
            dcc.Graph(id="loan-ecl-chart", style={'height': '350px'})
        ], className="smooth-card"), width=6),
        
        # Right Panel: Scatter plot of Risk Concentration
        dbc.Col(html.Div([
            html.Div("Risk Concentration (PD vs Exposure)", className="smooth-header"),
            dcc.Graph(id="scatter-risk-chart", style={'height': '350px'})
        ], className="smooth-card"), width=6),
    ], className="mb-4")
    
], fluid=True, style={'padding': '30px'})

# --- HELPER FUNCTION: CLEAN CHART LAYOUTS ---
def get_clean_layout(x_title="", y_title=""):
    """Returns a unified, clean, grid-free layout for all charts."""
    return dict(
        template="plotly_dark", 
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=10, b=40), 
        xaxis=dict(title=x_title, showgrid=False, zeroline=False, color="#9ca3af"),
        yaxis=dict(title=y_title, showgrid=False, zeroline=False, color="#9ca3af"),
        font=dict(family="Inter, sans-serif", color="#d1d5db")
    )

# --- CALLBACKS ---
@app.callback(
    [Output("total-exposure-text", "children"),
     Output("ecl-text", "children"),
     Output("var-text", "children"),
     Output("es-text", "children"),
     Output("loss-distribution-chart", "figure"),
     Output("sector-donut-chart", "figure"),
     Output("loan-ecl-chart", "figure"),
     Output("scatter-risk-chart", "figure")],
    [Input("run-btn", "n_clicks")],
    [State("correlation-slider", "value"),
     State("num-simulations", "value")]
)
def update_dashboard(n_clicks, correlation_rho, n_simulations):
    # Ensure n_simulations is an integer (dbc.Select passes strings)
    n_simulations = int(n_simulations)
    
    # 1. Load Portfolio
    portfolio = load_live_portfolio('data/live_portfolio.csv')
    if not portfolio.loans:
        empty_fig = go.Figure().update_layout(get_clean_layout())
        return "No Data", "No Data", "No Data", "No Data", empty_fig, empty_fig, empty_fig, empty_fig
    
    # 2. Run Deterministic ECL Engine
    ecl_engine = ECLCalculator(
        portfolio=portfolio,
        pd_model_path='models/pd_model_assets/model.pkl',
        pd_features_path='models/pd_model_assets/features.pkl',
        lgd_model_path='models/lgd_model_assets/model.pkl',
        lgd_features_path='models/lgd_model_assets/features.pkl'
    )
    ecl_results = ecl_engine.calculate_risk()
    
    # 3. Run Stochastic Copula Engine
    num_loans = len(portfolio.loans)
    sector_correlations = {
        'Tech': {'Tech': 0.8, 'Healthcare': 0.3, 'Consumer': 0.4, 'Finance': 0.5},
        'Healthcare': {'Tech': 0.3, 'Healthcare': 0.7, 'Consumer': 0.2, 'Finance': 0.4},
        'Consumer': {'Tech': 0.4, 'Healthcare': 0.2, 'Consumer': 0.6, 'Finance': 0.6},
        'Finance': {'Tech': 0.5, 'Healthcare': 0.4, 'Consumer': 0.6, 'Finance': 0.9}
    }

    corr_matrix = np.zeros((num_loans, num_loans))
    for i in range(num_loans):
        for j in range(num_loans):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                s_i = portfolio.loans[i].sector
                s_j = portfolio.loans[j].sector
                # Fallback to base rho if sector missing
                sc = sector_correlations.get(s_i, {}).get(s_j, correlation_rho)
                corr_matrix[i, j] = (correlation_rho + sc) / 2
    
    copula_engine = CopulaSimulationEngine(
        portfolio=portfolio,
        correlation_matrix=corr_matrix,
        num_simulations=n_simulations
    )
    tail_results = copula_engine.calculate_risk()
    
    # --- Format KPI Outputs ---
    total_exposure_str = f"${ecl_results['total_exposure']:,.0f}"
    total_ecl_str = f"${ecl_results['total_ecl']:,.0f}"
    var_str = f"${tail_results['var_99']:,.0f}"
    es_str = f"${tail_results['expected_shortfall']:,.0f}"
    
    # Extract dataframe and add sector info for plotting
    df_loans = ecl_results['loan_level_data']
    df_loans['Sector'] = [loan.sector for loan in portfolio.loans]
    # Changed from 'Facility' to 'Loan' for clarity
    df_loans['Loan ID'] = [f"Loan {i+1}" for i in range(len(df_loans))]
    
    # --- CHART 1: Histogram (Loss Distribution) ---
    losses = tail_results['simulated_loss_distribution']
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=losses, nbinsx=100, marker_color='#3b82f6', name='Simulations'))
    
    # Add VaR and ES lines
    max_count = np.max(np.histogram(losses, bins=100)[0]) if len(losses) > 0 else 100
    fig_hist.add_vline(x=tail_results['var_99'], line_dash="dash", line_color="#f59e0b")
    fig_hist.add_annotation(x=tail_results['var_99'], y=max_count * 0.8, text="99% VaR", showarrow=True, arrowhead=2, arrowcolor="#f59e0b", ax=-40, ay=-30, font=dict(color="#f59e0b", weight="bold"))
    
    fig_hist.add_vline(x=tail_results['expected_shortfall'], line_dash="dash", line_color="#ef4444")
    fig_hist.add_annotation(x=tail_results['expected_shortfall'], y=max_count * 0.95, text="97.5% ES", showarrow=True, arrowhead=2, arrowcolor="#ef4444", ax=40, ay=-30, font=dict(color="#ef4444", weight="bold"))
    
    fig_hist.update_layout(get_clean_layout(x_title="Portfolio Loss ($)", y_title="Frequency"))
    fig_hist.update_layout(showlegend=False)

    # --- CHART 2: Donut Chart (ECL by Sector) ---
    sector_grouped = df_loans.groupby('Sector')['ecl'].sum().reset_index()
    fig_donut = go.Figure(data=[go.Pie(
        labels=sector_grouped['Sector'], 
        values=sector_grouped['ecl'], 
        hole=.6,
        marker=dict(colors=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']),
        textinfo='percent'
    )])
    fig_donut.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    # --- CHART 3: Bar Chart (Loan Level ECL) ---
    df_sorted = df_loans.sort_values(by='ecl', ascending=False).head(15) # Top 15 risks
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_sorted['Loan ID'], y=df_sorted['ecl'], 
        marker_color='#10b981',
        text=df_sorted['ecl'].apply(lambda x: f"${x/1000:,.1f}k"),
        textposition='outside', textfont=dict(color="#d1d5db")
    ))
    fig_bar.update_layout(get_clean_layout(x_title="", y_title="Expected Loss ($)"))
    # Hide x-axis line specifically for bar chart to look cleaner
    fig_bar.update_xaxes(showline=False, tickangle=-45)

    # --- CHART 4: Scatter Plot (Risk Concentration) ---
    # Size based on ECL, capped for visual clarity
    bubble_sizes = np.clip((df_loans['ecl'] / df_loans['ecl'].max()) * 60, 10, 60)
    
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df_loans['loan_amnt'], 
        y=df_loans['pd'] * 100, # Convert PD to percentage
        mode='markers',
        text=df_loans['Loan ID'] + "<br>Sector: " + df_loans['Sector'],
        marker=dict(
            size=bubble_sizes,
            color=df_loans['ecl'],
            colorscale='Inferno', # Smooth heat map colors
            showscale=True,
            colorbar=dict(title="ECL ($)", thickness=15, outlinewidth=0),
            opacity=0.8,
            line=dict(width=1, color='#1f2937')
        )
    ))
    fig_scatter.update_layout(get_clean_layout(x_title="Exposure at Default ($)", y_title="Probability of Default (%)"))

    return total_exposure_str, total_ecl_str, var_str, es_str, fig_hist, fig_donut, fig_bar, fig_scatter

if __name__ == '__main__':
    app.run(debug=True, port=8050)
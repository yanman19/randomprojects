import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table, callback_context, no_update
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.optimize import minimize_scalar
import json

# Initial defaults
default_current_age = 26
default_death_age = 90
default_retirement_age = 50
default_net_worth = 35000
default_tax_rate = 50
default_investment_growth_rate = 3
default_pre_tax_income = 300000
default_pre_tax_income_growth = 10
default_health_deterioration_rate = 1.31
default_retirement_balance = 45000
default_retirement_growth_rate = 3
default_annual_contribution = 24000
default_current_spending_percentage = 0.75

# Initial life purchases
default_life_purchases = [
    {"name": "Dream Vacation", "cost": 10000, "age": 35},
    {"name": "Home Renovation", "cost": 50000, "age": 45}
]

# Create Dash app
app = Dash(__name__, meta_tags=[
    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
])
app.title = "Die With Zero - YANUS Style v7"

# Enhanced Theme Colors
theme = {
    'background': '#121212',
    'paper_bg': '#1E1E1E',
    'card_bg': '#252525',
    'text': '#E0E0E0',
    'primary': '#4CAF50',    # Green primary
    'secondary': '#2196F3',  # Blue secondary
    'accent': '#FF9800',     # Orange accent
    'danger': '#F44336',     # Red for negative values
    'border': '#424242',
    'input_bg': '#333333',
    'chart_grid': '#333333',
}

# CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');
            
            body {
                font-family: 'Roboto', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #121212;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                padding: 20px 0;
                margin-bottom: 30px;
                border-bottom: 1px solid #424242;
            }
            
            .input-row {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .input-group {
                flex: 1 1 200px;
                margin-bottom: 15px;
            }
            
            .input-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: 500;
                color: #E0E0E0;
            }
            
            .input-field {
                width: 100%;
                padding: 10px;
                border-radius: 4px;
                border: 1px solid #424242;
                background-color: #333333;
                color: #FFFFFF;
                transition: border-color 0.3s;
            }
            
            .input-field:focus {
                border-color: #4CAF50;
                outline: none;
            }
            
            .submit-btn {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: 500;
                transition: background-color 0.3s;
            }
            
            .submit-btn:hover {
                background-color: #45a049;
            }
            
            .optimize-btn {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: 500;
                transition: background-color 0.3s;
            }
            
            .optimize-btn:hover {
                background-color: #F57C00;
            }
            
            .chart-container {
                background-color: #252525;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 25px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .chart-row {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .chart-col {
                flex: 1 1 calc(50% - 20px);
                min-width: 300px;
            }
            
            .table-container {
                background-color: #252525;
                border-radius: 8px;
                padding: 15px;
                margin-top: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .optimization-container {
                background-color: #2D3748;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                border-left: 4px solid #FF9800;
            }
            
            .optimization-result {
                font-size: 24px;
                font-weight: bold;
                color: #FF9800;
                text-align: center;
                margin: 10px 0;
            }
            
            .optimization-explanation {
                color: #E0E0E0;
                margin-top: 10px;
                font-size: 14px;
                line-height: 1.5;
            }
            
            /* Life Purchases Styling */
            .life-purchases-container {
                background-color: #252525;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 25px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-left: 4px solid #2196F3;
            }
            
            .life-purchase-item {
                display: flex;
                background-color: #333333;
                border-radius: 4px;
                padding: 10px;
                margin-bottom: 10px;
                align-items: center;
            }
            
            .life-purchase-name {
                flex: 2;
                font-weight: 500;
            }
            
            .life-purchase-cost, .life-purchase-age {
                flex: 1;
                text-align: center;
            }
            
            .life-purchase-actions {
                flex: 0 0 40px;
                text-align: right;
            }
            
            .life-purchase-add {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 15px;
                align-items: flex-end;
            }
            
            .purchase-btn {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: 500;
                transition: background-color 0.3s;
            }
            
            .purchase-btn:hover {
                background-color: #1E88E5;
            }
            
            .delete-btn {
                background-color: #F44336;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: background-color 0.3s;
            }
            
            .delete-btn:hover {
                background-color: #E53935;
            }
            
            @media (max-width: 768px) {
                .chart-col {
                    flex: 1 1 100%;
                }
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

# App layout with improved UI
app.layout = html.Div(className='container', children=[
    html.Div(className='header', children=[
        html.H1("Die With Zero | YANUS-Style v7", style={'color': theme['primary'], 'margin': '0'}),
        html.P("Next: Solve for what year to retire and/or at what dollar amount of net worth is enough.", style={'color': theme['text'], 'marginTop': '5px'})
    ]),
    
    # Input form with better organization
    html.Div(className='form-container', children=[
        html.Div(className='input-row', children=[
            html.Div(className='input-group', children=[
                html.Label("Current Age"),
                dcc.Input(
                    id='current-age-input', 
                    type='number', 
                    value=default_current_age, 
                    min=18, 
                    max=120,
                    className='input-field'
                )
            ]),
            html.Div(className='input-group', children=[
                html.Label("Death Age"),
                dcc.Input(
                    id='death-age-input', 
                    type='number', 
                    value=default_death_age, 
                    min=18, 
                    max=120,
                    className='input-field'
                )
            ]),
            html.Div(className='input-group', children=[
                html.Label("Retirement Age"),
                dcc.Input(
                    id='retirement-age-input', 
                    type='number', 
                    value=default_retirement_age, 
                    min=18, 
                    max=120,
                    className='input-field'
                )
            ]),
        ]),
        
        html.Div(className='input-row', children=[
            html.Div(className='input-group', children=[
                html.Label("Net Worth ($)"),
                dcc.Input(
                    id='net-worth-input', 
                    type='number', 
                    value=default_net_worth, 
                    min=0,
                    className='input-field'
                )
            ]),
            html.Div(className='input-group', children=[
                html.Label("Tax Rate (%)"),
                dcc.Input(
                    id='tax-rate-input', 
                    type='number', 
                    value=default_tax_rate, 
                    min=0, 
                    max=100,
                    className='input-field'
                )
            ]),
            html.Div(className='input-group', children=[
                html.Label("Investment Growth (%)"),
                dcc.Input(
                    id='investment-growth-rate-input', 
                    type='number', 
                    value=default_investment_growth_rate, 
                    min=0, 
                    max=100,
                    className='input-field'
                )
            ]),
        ]),
        
        html.Div(className='input-row', children=[
            html.Div(className='input-group', children=[
                html.Label("Pre-Tax Income ($)"),
                dcc.Input(
                    id='pre-tax-income-input', 
                    type='number', 
                    value=default_pre_tax_income, 
                    min=0,
                    className='input-field'
                )
            ]),
            html.Div(className='input-group', children=[
                html.Label("Income Growth (%)"),
                dcc.Input(
                    id='pre-tax-income-growth-input', 
                    type='number', 
                    value=default_pre_tax_income_growth, 
                    min=0, 
                    max=100,
                    className='input-field'
                )
            ]),
            html.Div(className='input-group', children=[
                html.Label("Health Decline Rate (%)"),
                dcc.Input(
                    id='health-deterioration-rate-input', 
                    type='number', 
                    value=default_health_deterioration_rate, 
                    min=0, 
                    max=100,
                    className='input-field'
                )
            ]),
        ]),
        
        html.Div(className='input-row', children=[
            html.Div(className='input-group', children=[
                html.Label("Retirement Balance ($)"),
                dcc.Input(
                    id='retirement-balance-input', 
                    type='number', 
                    value=default_retirement_balance, 
                    min=0,
                    className='input-field'
                )
            ]),
            html.Div(className='input-group', children=[
                html.Label("Retirement Growth (%)"),
                dcc.Input(
                    id='retirement-growth-rate-input', 
                    type='number', 
                    value=default_retirement_growth_rate, 
                    min=0, 
                    max=100,
                    className='input-field'
                )
            ]),
            html.Div(className='input-group', children=[
                html.Label("Annual Contribution ($)"),
                dcc.Input(
                    id='annual-contribution-input', 
                    type='number', 
                    value=default_annual_contribution, 
                    min=0,
                    className='input-field'
                )
            ]),
        ]),
        
        html.Div(className='input-row', children=[
            html.Div(className='input-group', children=[
                html.Label("% of Income Spent"),
                dcc.Input(
                    id='current-spending-percentage-input', 
                    type='number', 
                    value=default_current_spending_percentage, 
                    min=0, 
                    max=1, 
                    step=0.01,
                    className='input-field'
                )
            ]),
            html.Div(className='input-group', style={'display': 'flex', 'alignItems': 'flex-end'}, children=[
                html.Button(
                    'Calculate Results', 
                    id='submit-inputs-button', 
                    n_clicks=0,
                    className='submit-btn'
                )
            ]),
            html.Div(className='input-group', style={'display': 'flex', 'alignItems': 'flex-end'}, children=[
                html.Button(
                    'Optimize Spending', 
                    id='optimize-button', 
                    n_clicks=0,
                    className='optimize-btn'
                )
            ]),
        ]),
    ]),
    
    # Life Purchases Section (New)
    html.Div(className='life-purchases-container', children=[
        html.H3("Life Purchases", style={'color': theme['secondary'], 'marginTop': '0'}),
        html.P("Add significant purchases to see how they impact your financial trajectory", 
               style={'color': theme['text'], 'fontSize': '14px', 'marginBottom': '20px'}),
        
        # Life Purchases List
        html.Div(id='life-purchases-list'),
        
        # Add new purchase form
        html.Div(className='life-purchase-add', children=[
            html.Div(className='input-group', style={'flex': '2'}, children=[
                html.Label("Purchase Name"),
                dcc.Input(
                    id='purchase-name-input', 
                    type='text', 
                    placeholder="e.g. Dream Vacation",
                    className='input-field'
                )
            ]),
            html.Div(className='input-group', style={'flex': '1'}, children=[
                html.Label("Cost ($)"),
                dcc.Input(
                    id='purchase-cost-input', 
                    type='number', 
                    placeholder="10000",
                    min=0,
                    className='input-field'
                )
            ]),
            html.Div(className='input-group', style={'flex': '1'}, children=[
                html.Label("Age"),
                dcc.Input(
                    id='purchase-age-input', 
                    type='number', 
                    placeholder="35",
                    min=18,
                    max=120,
                    className='input-field'
                )
            ]),
            html.Button(
                'Add Purchase', 
                id='add-purchase-button', 
                n_clicks=0,
                className='purchase-btn'
            )
        ]),
        
        # Store for life purchases
        dcc.Store(id='life-purchases-store', data=default_life_purchases),
    ]),
    
    # Optimization Results Section (Hidden by default)
    html.Div(id='optimization-container', className='optimization-container', style={'display': 'none'}, children=[
        html.H3("Optimization Results", style={'color': theme['accent'], 'marginTop': '0'}),
        html.Div(id='optimization-result', className='optimization-result'),
        html.Div(id='optimization-explanation', className='optimization-explanation'),
    ]),
    
    html.Div(id='summary-stats', className='chart-container', children=[
        html.Div(id='summary-cards', className='chart-row', style={'marginTop': '10px'})
    ]),
    
    # Charts in a grid layout
    html.Div(className='chart-row', children=[
        html.Div(className='chart-col', children=[
            html.Div(className='chart-container', children=[
                dcc.Graph(id='wealth-chart')
            ]),
        ]),
        html.Div(className='chart-col', children=[
            html.Div(className='chart-container', children=[
                dcc.Graph(id='post-tax-income-chart')
            ]),
        ]),
    ]),
    
    html.Div(className='chart-row', children=[
        html.Div(className='chart-col', children=[
            html.Div(className='chart-container', children=[
                dcc.Graph(id='spending-chart')
            ]),
        ]),
        html.Div(className='chart-col', children=[
            html.Div(className='chart-container', children=[
                dcc.Graph(id='health-chart')
            ]),
        ]),
    ]),

    html.Div(className='chart-row', children=[
        html.Div(className='chart-col', style={'flex': '1 1 100%'}, children=[
            html.Div(className='chart-container', children=[
                dcc.Graph(id='percent-income-spent-chart')
            ]),
        ]),
    ]),
    
    # Data table with improved styling and export button
    html.Div(className='table-container', children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '15px'}, children=[
            html.H3("Detailed Projections", style={'color': theme['text'], 'margin': '0'}),
            html.Button(
                'Export to CSV', 
                id='export-button',
                className='submit-btn',
                style={'backgroundColor': theme['secondary'], 'fontSize': '14px', 'padding': '8px 16px'}
            ),
        ]),
        dash_table.DataTable(
            id='answer-table',
            style_table={
                'overflowX': 'auto',
                'overflowY': 'auto',
                'maxHeight': '600px',
                'backgroundColor': theme['card_bg'],
                'borderRadius': '4px',
                'border': f'1px solid {theme["border"]}',
            },
            style_cell={
                'textAlign': 'right',
                'backgroundColor': theme['card_bg'],
                'color': theme['text'],
                'border': f'1px solid {theme["border"]}',
                'padding': '12px 15px',
                'fontFamily': 'Roboto, sans-serif',
                'minWidth': '100px',
                'width': '150px',
                'maxWidth': '200px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_header={
                'backgroundColor': theme['paper_bg'],
                'color': theme['primary'],
                'fontWeight': 'bold',
                'textAlign': 'center',
                'padding': '15px 10px',
                'borderBottom': f'2px solid {theme["primary"]}',
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Age'},
                    'fontWeight': 'bold',
                    'backgroundColor': theme['paper_bg'],
                    'textAlign': 'center',
                    'width': '80px',
                },
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgba(0, 0, 0, 0.05)',
                },
            ],
            fixed_rows={'headers': True},
            page_action='none',  # No pagination, show all rows
            export_format='csv',
            export_headers='display',
        )
    ]),
    
    # Hidden div to store optimization data
    html.Div(id='optimization-data', style={'display': 'none'})
])

# Helper function to calculate financial projection
def calculate_projection(current_age, death_age, retirement_age, net_worth, tax_rate, growth_rate, pre_tax_income,
                       income_growth, health_rate, retirement_balance, retirement_growth, annual_contribution, 
                       current_spending_percentage, life_purchases):
    total_years = death_age - current_age + 1
    ages = list(range(current_age, death_age + 1))

    # Income Calculations
    pre_tax_income_values = [pre_tax_income * (1 + income_growth / 100) ** i if ages[i] < retirement_age else 0 for i in range(total_years)]
    post_tax_income = [round(inc * (1 - tax_rate / 100)) for inc in pre_tax_income_values]

    # Health Calculations
    health_values = [max(100 - health_rate * i, 0) for i in range(total_years)]

    # Retirement Account Calculations
    retirement_balances = [retirement_balance]
    for i in range(1, total_years):
        if ages[i] <= 65:
            if ages[i] < retirement_age:
                retirement_balances.append(round(retirement_balances[-1] * (1 + retirement_growth / 100) + annual_contribution))
            else:
                retirement_balances.append(round(retirement_balances[-1] * (1 + retirement_growth / 100)))
        else:
            retirement_balances.append(0)  # Retirement balance is 0 after age 65

    # Spending Calculations
    spending = []
    retirement_spending = 0

    for i in range(total_years):
        if ages[i] < retirement_age:  # Pre-retirement spending
            spending_percentage = current_spending_percentage + (0.5 - current_spending_percentage) * (ages[i] - current_age) / (retirement_age - current_age)
            spend = round(post_tax_income[i] * spending_percentage)
            retirement_spending = spend  # Capture spending to calculate post-retirement
        else:  # Post-retirement spending
            spend = round(retirement_spending * (1 - (ages[i] - retirement_age) / (death_age - retirement_age)))
        spending.append(-spend)
    
    # Life Purchases - Add major purchases at specified ages
    life_purchase_costs = [0] * total_years
    life_purchase_markers = []  # For annotations in charts
    
    for purchase in life_purchases:
        purchase_age = purchase.get('age')
        purchase_cost = purchase.get('cost')
        purchase_name = purchase.get('name')
        
        if purchase_age is not None and purchase_cost is not None:
            # Find the index for this age in our array
            if purchase_age >= current_age and purchase_age <= death_age:
                idx = purchase_age - current_age
                life_purchase_costs[idx] += -purchase_cost
                
                # Add marker for chart visualization
                life_purchase_markers.append({
                    'age': purchase_age,
                    'name': purchase_name,
                    'cost': purchase_cost
                })

    # Annual Savings (Post-Tax Income - Spending - Life Purchases)
    annual_savings = [(post_tax_income[i] + spending[i] + life_purchase_costs[i]) for i in range(total_years)]

    # Wealth Calculations
    boy_net_worth = []
    investment_growth = []
    eoy_net_worth = []
    
    # First year: BOY = initial net worth
    current_net_worth = net_worth
    boy_net_worth.append(current_net_worth)
    
    # Calculate for each year
    for i in range(total_years):
        # Calculate investment growth
        growth = current_net_worth * growth_rate / 100
        investment_growth.append(round(growth))
        
        # Calculate EOY net worth
        new_wealth = current_net_worth + annual_savings[i] + growth
        
        # Add retirement account at age 65
        if ages[i] == 65:
            new_wealth += retirement_balances[i]
            
        eoy_net_worth.append(round(new_wealth))
        
        # Next year's BOY is this year's EOY
        if i < total_years - 1:
            current_net_worth = new_wealth
            boy_net_worth.append(round(current_net_worth))
            
    # Ensure retirement account is added to EOY net worth at age 65
    age_65_index = next((i for i, age in enumerate(ages) if age == 65), None)
    if age_65_index is not None:
        eoy_net_worth[age_65_index] += retirement_balances[age_65_index]

    # Calculate Percentage of Income Spent
    percent_income_spent = [round((abs(spending[i]) / post_tax_income[i]) * 100, 2) if post_tax_income[i] > 0 else 0 for i in range(total_years)]
    
    return {
        'ages': ages,
        'boy_net_worth': boy_net_worth,
        'pre_tax_income_values': pre_tax_income_values,
        'post_tax_income': post_tax_income,
        'spending': spending,
        'life_purchase_costs': life_purchase_costs,
        'life_purchase_markers': life_purchase_markers,
        'annual_savings': annual_savings,
        'investment_growth': investment_growth,
        'retirement_balances': retirement_balances,
        'eoy_net_worth': eoy_net_worth,
        'health_values': health_values,
        'percent_income_spent': percent_income_spent,
        'final_net_worth': eoy_net_worth[-1]
    }

# Function to optimize for die with zero
def optimize_spending(current_age, death_age, retirement_age, net_worth, tax_rate, growth_rate, pre_tax_income,
                     income_growth, health_rate, retirement_balance, retirement_growth, annual_contribution, life_purchases):
    
    def objective_function(spending_percentage):
        projection = calculate_projection(
            current_age, death_age, retirement_age, net_worth, tax_rate, growth_rate, pre_tax_income,
            income_growth, health_rate, retirement_balance, retirement_growth, annual_contribution, 
            spending_percentage, life_purchases
        )
        # Return absolute value of final net worth to minimize
        return abs(projection['final_net_worth'])
    
    # Find optimal spending percentage using scipy's minimize_scalar
    result = minimize_scalar(
        objective_function,
        bounds=(0.1, 0.95),  # Search between 10% and 95% spending
        method='bounded',
        options={'xatol': 0.0001}  # Tolerance for optimization
    )
    
    optimal_percentage = result.x
    
    # Calculate projection with optimal percentage
    optimal_projection = calculate_projection(
        current_age, death_age, retirement_age, net_worth, tax_rate, growth_rate, pre_tax_income,
        income_growth, health_rate, retirement_balance, retirement_growth, annual_contribution, 
        optimal_percentage, life_purchases
    )
    
    return {
        'optimal_percentage': optimal_percentage,
        'final_net_worth': optimal_projection['final_net_worth'],
        'projection': optimal_projection
    }

# Summary cards Output callback
@app.callback(
    Output('summary-cards', 'children'),
    Input('submit-inputs-button', 'n_clicks'),
    [State('current-age-input', 'value'),
     State('death-age-input', 'value'),
     State('retirement-age-input', 'value'),
     State('net-worth-input', 'value'),
     State('life-purchases-store', 'data')]
)
def update_summary_cards(n_clicks, current_age, death_age, retirement_age, net_worth, life_purchases):
    work_years = retirement_age - current_age
    retirement_years = death_age - retirement_age
    
    # Calculate total life purchases
    total_life_purchases = sum(purchase.get('cost', 0) for purchase in life_purchases)
    
    card_style = {
        'flex': '1 1 calc(20% - 20px)',  # Changed from 25% to 20% to accommodate 5 cards
        'backgroundColor': theme['card_bg'],
        'borderRadius': '8px',
        'padding': '15px',
        'textAlign': 'center',
        'minWidth': '150px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'border': f'1px solid {theme["border"]}'
    }
    
    value_style = {
        'fontSize': '28px',
        'fontWeight': 'bold',
        'margin': '10px 0',
        'color': theme['primary']
    }
    
    label_style = {
        'color': theme['text'],
        'fontSize': '14px'
    }
    
    # New card for life purchases with a different color
    life_purchase_value_style = {
        'fontSize': '28px',
        'fontWeight': 'bold',
        'margin': '10px 0',
        'color': theme['secondary']  # Different color for life purchases
    }
    
    return [
        html.Div(style=card_style, children=[
            html.Div(style=value_style, children=f"{current_age}"),
            html.Div(style=label_style, children="Current Age")
        ]),
        html.Div(style=card_style, children=[
            html.Div(style=value_style, children=f"{work_years}"),
            html.Div(style=label_style, children="Working Years")
        ]),
        html.Div(style=card_style, children=[
            html.Div(style=value_style, children=f"{retirement_years}"),
            html.Div(style=label_style, children="Retirement Years")
        ]),
        html.Div(style=card_style, children=[
            html.Div(style=value_style, children=f"${net_worth:,}"),
            html.Div(style=label_style, children="Current Net Worth")
        ]),
        html.Div(style=card_style, children=[
            html.Div(style=life_purchase_value_style, children=f"${total_life_purchases:,}"),
            html.Div(style=label_style, children="Life Purchases")
        ])
    ]

# Add a callback for the export button
@app.callback(
    Output('answer-table', 'export_format'),
    Input('export-button', 'n_clicks'),
    prevent_initial_call=True
)
def export_data(n_clicks):
    return 'csv'

# Life Purchases List callback
@app.callback(
    Output('life-purchases-list', 'children'),
    Input('life-purchases-store', 'data')
)
def update_life_purchases_list(purchases):
    if not purchases:
        return html.Div("No life purchases added yet. Add your first purchase below.",
                      style={'color': theme['text'], 'fontStyle': 'italic', 'marginBottom': '15px'})
    
    purchase_items = []
    for i, purchase in enumerate(purchases):
        purchase_items.append(
            html.Div(className='life-purchase-item', children=[
                html.Div(className='life-purchase-name', children=purchase.get('name', 'Unnamed')),
                html.Div(className='life-purchase-cost', children=f"${purchase.get('cost', 0):,}"),
                html.Div(className='life-purchase-age', children=f"Age {purchase.get('age', 'N/A')}"),
                html.Div(className='life-purchase-actions', children=[
                    html.Button('×', 
                               id={'type': 'delete-purchase', 'index': i}, 
                               className='delete-btn',
                               title='Delete this purchase')
                ])
            ])
        )
    
    return html.Div(children=purchase_items)

# Add new purchase callback
@app.callback(
    [Output('life-purchases-store', 'data'),
     Output('purchase-name-input', 'value'),
     Output('purchase-cost-input', 'value'),
     Output('purchase-age-input', 'value')],
    [Input('add-purchase-button', 'n_clicks'),
     Input({'type': 'delete-purchase', 'index': dash.ALL}, 'n_clicks')],
    [State('life-purchases-store', 'data'),
     State('purchase-name-input', 'value'),
     State('purchase-cost-input', 'value'),
     State('purchase-age-input', 'value')]
)
def manage_purchases(add_clicks, delete_clicks, purchases, name, cost, age):
    ctx = callback_context
    
    if not ctx.triggered:
        return purchases, "", None, None
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle add purchase
    if triggered_id == 'add-purchase-button':
        if name and cost is not None and age is not None:
            new_purchase = {
                "name": name,
                "cost": cost,
                "age": age
            }
            updated_purchases = purchases + [new_purchase]
            return updated_purchases, "", None, None
        return purchases, name, cost, age
    
    # Handle delete purchase
    try:
        # Parse the JSON from the triggered ID
        trigger_dict = json.loads(triggered_id)
        if trigger_dict.get('type') == 'delete-purchase':
            index = trigger_dict.get('index')
            updated_purchases = [p for i, p in enumerate(purchases) if i != index]
            return updated_purchases, dash.no_update, dash.no_update, dash.no_update
    except:
        pass
    
    return purchases, dash.no_update, dash.no_update, dash.no_update

# Store optimization results
@app.callback(
    Output('optimization-data', 'children'),
    Input('optimize-button', 'n_clicks'),
    [State('current-age-input', 'value'),
     State('death-age-input', 'value'),
     State('retirement-age-input', 'value'),
     State('net-worth-input', 'value'),
     State('tax-rate-input', 'value'),
     State('investment-growth-rate-input', 'value'),
     State('pre-tax-income-input', 'value'),
     State('pre-tax-income-growth-input', 'value'),
     State('health-deterioration-rate-input', 'value'),
     State('retirement-balance-input', 'value'),
     State('retirement-growth-rate-input', 'value'),
     State('annual-contribution-input', 'value'),
     State('life-purchases-store', 'data')]
)
def run_optimization(n_clicks, current_age, death_age, retirement_age, net_worth, tax_rate, growth_rate, pre_tax_income,
                    income_growth, health_rate, retirement_balance, retirement_growth, annual_contribution, life_purchases):
    if n_clicks == 0:
        return ""
        
    # Run optimization
    result = optimize_spending(
        current_age, death_age, retirement_age, net_worth, tax_rate, growth_rate, pre_tax_income,
        income_growth, health_rate, retirement_balance, retirement_growth, annual_contribution, life_purchases
    )
    
    return str(result['optimal_percentage'])

# Update optimization result display
@app.callback(
    [Output('optimization-container', 'style'),
     Output('optimization-result', 'children'),
     Output('optimization-explanation', 'children'),
     Output('current-spending-percentage-input', 'value')],
    [Input('optimization-data', 'children'),
     Input('submit-inputs-button', 'n_clicks')]
)
def update_optimization_display(optimization_data, submit_clicks):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'submit-inputs-button':
        # Keep the optimization container hidden when regular submit is clicked
        return {'display': 'none'}, "", "", dash.no_update
    
    if not optimization_data:
        return {'display': 'none'}, "", "", dash.no_update
    
    try:
        optimal_percentage = float(optimization_data)
        
        # Update the UI with optimization results
        result_text = f"Optimal Spending: {optimal_percentage:.2%}"
        
        explanation = html.Div([
            html.P([
                "With this spending percentage, you'll gradually adjust your spending over your lifetime to perfectly exhaust your wealth by the end of your life projection. ",
                "This optimizes the 'Die With Zero' principle, allowing you to enjoy your money during your lifetime without leaving excess wealth unspent."
            ]),
            html.P([
                "This calculation includes your life purchases. ",
                "The optimizer has automatically updated your spending percentage in the form above. ",
                "Click 'Calculate Results' to see the updated projections."
            ])
        ])
        
        # Round to 2 decimal places for display in the input field
        rounded_percentage = round(optimal_percentage * 100) / 100
        return {'display': 'block'}, result_text, explanation, rounded_percentage
        
    except:
        return {'display': 'none'}, "", "", dash.no_update

@app.callback(
    [Output('wealth-chart', 'figure'),
     Output('post-tax-income-chart', 'figure'),
     Output('percent-income-spent-chart', 'figure'),
     Output('spending-chart', 'figure'),
     Output('health-chart', 'figure'),
     Output('answer-table', 'data')],
    Input('submit-inputs-button', 'n_clicks'),
    [State('current-age-input', 'value'),
     State('death-age-input', 'value'),
     State('retirement-age-input', 'value'),
     State('net-worth-input', 'value'),
     State('tax-rate-input', 'value'),
     State('investment-growth-rate-input', 'value'),
     State('pre-tax-income-input', 'value'),
     State('pre-tax-income-growth-input', 'value'),
     State('health-deterioration-rate-input', 'value'),
     State('retirement-balance-input', 'value'),
     State('retirement-growth-rate-input', 'value'),
     State('annual-contribution-input', 'value'),
     State('current-spending-percentage-input', 'value'),
     State('life-purchases-store', 'data')]
)
def update_outputs(n_clicks, current_age, death_age, retirement_age, net_worth, tax_rate, growth_rate, pre_tax_income,
                   income_growth, health_rate, retirement_balance, retirement_growth, annual_contribution, 
                   current_spending_percentage, life_purchases):
    
    projection = calculate_projection(
        current_age, death_age, retirement_age, net_worth, tax_rate, growth_rate, pre_tax_income,
        income_growth, health_rate, retirement_balance, retirement_growth, annual_contribution, 
        current_spending_percentage, life_purchases
    )
    
    ages = projection['ages']
    boy_net_worth = projection['boy_net_worth']
    pre_tax_income_values = projection['pre_tax_income_values']
    post_tax_income = projection['post_tax_income']
    spending = projection['spending']
    life_purchase_costs = projection['life_purchase_costs']
    life_purchase_markers = projection['life_purchase_markers']
    annual_savings = projection['annual_savings']
    investment_growth = projection['investment_growth']
    retirement_balances = projection['retirement_balances']
    eoy_net_worth = projection['eoy_net_worth']
    health_values = projection['health_values']
    percent_income_spent = projection['percent_income_spent']

    # Table
    def format_number(value):
        return f"({abs(value):,})" if value < 0 else f"{value:,}"

    # Add life purchases to table display
    life_purchases_display = [format_number(val) if val != 0 else "" for val in life_purchase_costs]
    
    answer_table = pd.DataFrame({
        'Age': ages,
        'BOY Net Worth': [round(val) for val in boy_net_worth],
        'Pre-Tax Income': [round(val) for val in pre_tax_income_values],
        'Post-Tax Income': post_tax_income,
        'Tax Rate (%)': [f"{tax_rate}%" for _ in ages],
        'Regular Spending': [format_number(val) for val in spending],
        'Life Purchases': life_purchases_display,
        'Annual Savings': annual_savings,
        'Investment Growth': investment_growth,
        'Retirement Balance': retirement_balances,
        'EOY Net Worth': eoy_net_worth
    })
    
    # Enhanced Chart Layout
    chart_layout = dict(
        plot_bgcolor=theme['paper_bg'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(family="Roboto, sans-serif", color=theme['text'], size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor=theme['chart_grid'],
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            gridcolor=theme['chart_grid'],
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
        ),
        hovermode="x unified"
    )

    # Wealth Chart with improved styling and life purchase markers
    wealth_chart = go.Figure()
    wealth_chart.add_trace(go.Scatter(
        x=ages, 
        y=boy_net_worth, 
        name='Net Worth',
        line=dict(color=theme['primary'], width=3),
        fill='tozeroy',
        fillcolor=f'rgba(76, 175, 80, 0.1)',
    ))
    wealth_chart.add_trace(go.Scatter(
        x=ages,
        y=retirement_balances,
        name='Retirement',
        line=dict(color=theme['secondary'], width=2, dash='dash'),
    ))
    
    # Add life purchase markers to wealth chart
    for marker in life_purchase_markers:
        wealth_chart.add_annotation(
            x=marker['age'],
            y=boy_net_worth[marker['age'] - current_age],
            text=f"{marker['name']}: ${marker['cost']:,}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=theme['secondary'],
            font=dict(size=10, color=theme['secondary']),
            bgcolor=theme['paper_bg'],
            bordercolor=theme['secondary'],
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
    
    wealth_chart.update_layout(
        title=dict(text="Wealth Trajectory", font=dict(size=18)),
        xaxis_title="Age",
        yaxis_title="Amount ($)",
        **chart_layout
    )
    # Add retirement age vertical line
    wealth_chart.add_vline(
        x=retirement_age, 
        line_dash="dash", 
        line_color=theme['accent'],
        annotation_text="Retirement",
        annotation_position="top right"
    )
    
    # Add a horizontal line at zero
    wealth_chart.add_hline(
        y=0,
        line_dash="solid",
        line_color="rgba(255, 255, 255, 0.3)",
        line_width=1,
    )
    
    # Highlight the final net worth value with rounding to 2 decimal places
    wealth_chart.add_annotation(
        x=ages[-1],
        y=boy_net_worth[-1],
        text=f"Final: ${round(boy_net_worth[-1], 2):,}",
        showarrow=True,
        arrowhead=1,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=theme['accent'],
        font=dict(size=12, color=theme['accent']),
        bordercolor=theme['accent'],
        borderwidth=2,
        borderpad=4,
        bgcolor=theme['card_bg'],
        opacity=0.8
    )

    # Post-Tax Income Chart
    post_tax_income_chart = go.Figure()
    post_tax_income_chart.add_trace(go.Scatter(
        x=ages, 
        y=post_tax_income, 
        name='Post-Tax Income',
        line=dict(color=theme['secondary'], width=3),
        fill='tozeroy',
        fillcolor=f'rgba(33, 150, 243, 0.1)',
    ))
    post_tax_income_chart.update_layout(
        title=dict(text="Post-Tax Income", font=dict(size=18)),
        xaxis_title="Age",
        yaxis_title="Income ($)",
        **chart_layout
    )

    # Percent Income Spent Chart
    percent_income_spent_chart = go.Figure()
    percent_income_spent_chart.add_trace(go.Scatter(
        x=ages, 
        y=percent_income_spent, 
        name='% Income Spent',
        line=dict(color=theme['accent'], width=3),
        fill='tozeroy',
        fillcolor=f'rgba(255, 152, 0, 0.1)',
    ))
    percent_income_spent_chart.update_layout(
        title=dict(text="Percentage of Income Spent", font=dict(size=18)),
        xaxis_title="Age",
        yaxis_title="% of Income",
        **chart_layout
    )

    # Spending Chart (now includes both regular spending and life purchases)
    spending_chart = go.Figure()
    
    # Regular spending
    spending_chart.add_trace(go.Scatter(
        x=ages, 
        y=[abs(s) for s in spending], 
        name='Regular Spending',
        line=dict(color=theme['danger'], width=3),
        fill='tozeroy',
        fillcolor=f'rgba(244, 67, 54, 0.1)',
    ))
    
    # Life purchases (show as bars)
    life_purchase_y = [abs(c) if c != 0 else None for c in life_purchase_costs]
    spending_chart.add_trace(go.Bar(
        x=ages,
        y=life_purchase_y,
        name='Life Purchases',
        marker_color=theme['secondary'],
        opacity=0.7
    ))
    
    spending_chart.update_layout(
        title=dict(text="Spending Over Time", font=dict(size=18)),
        xaxis_title="Age",
        yaxis_title="Spending ($)",
        barmode='stack',
        **chart_layout
    )

    # Health Chart
    health_chart = go.Figure()
    health_chart.add_trace(go.Scatter(
        x=ages, 
        y=health_values, 
        name='Health',
        line=dict(color='#9C27B0', width=3),
        fill='tozeroy',
        fillcolor='rgba(156, 39, 176, 0.1)',
    ))
    health_chart.update_layout(
        title=dict(text="Health Trajectory", font=dict(size=18)),
        xaxis_title="Age",
        yaxis_title="Health (%)",
        **chart_layout
    )

    # Format table data
    for column in ['BOY Net Worth', 'Pre-Tax Income', 'Post-Tax Income', 'Annual Savings', 'Investment Growth', 'Retirement Balance', 'EOY Net Worth']:
        answer_table[column] = answer_table[column].apply(format_number)

    return wealth_chart, post_tax_income_chart, percent_income_spent_chart, spending_chart, health_chart, answer_table.to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True)

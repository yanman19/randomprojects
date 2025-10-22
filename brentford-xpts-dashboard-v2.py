import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime
import plotly.graph_objects as go

# Import data fetching from existing working dashboard
import sys
sys.path.insert(0, '/home/user/randomprojects')

# We'll define our own functions but use similar logic to prem_dash_v4.py
# For now, let's use simulated/mock data since fbref is blocking requests

# ==============================
# MOCK DATA (Replace with real data when fbref access is available)
# ==============================

def create_mock_brentford_data():
    """Create mock data for demonstration purposes"""
    # Mock current stats
    brentford_stats = {
        'points': 28,
        'matches_played': 15,
        'goals_for': 24,
        'goals_against': 22,
        'goal_difference': 2
    }

    # Mock future matches
    future_matches = [
        {'date': datetime(2025, 1, 18), 'opponent': 'Liverpool', 'venue': 'Away', 'xPTS': 0.5, 'xG': 0.9, 'xGA': 2.1, 'predicted_score': '1-2', 'home_win_prob': 0.15, 'draw_prob': 0.22, 'away_win_prob': 0.63},
        {'date': datetime(2025, 1, 25), 'opponent': 'Man City', 'venue': 'Home', 'xPTS': 0.8, 'xG': 1.1, 'xGA': 1.9, 'predicted_score': '1-2', 'home_win_prob': 0.22, 'draw_prob': 0.28, 'away_win_prob': 0.50},
        {'date': datetime(2025, 2, 1), 'opponent': 'Everton', 'venue': 'Away', 'xPTS': 1.9, 'xG': 1.6, 'xGA': 1.1, 'predicted_score': '2-1', 'home_win_prob': 0.54, 'draw_prob': 0.25, 'away_win_prob': 0.21},
        {'date': datetime(2025, 2, 15), 'opponent': 'Crystal Palace', 'venue': 'Home', 'xPTS': 2.1, 'xG': 1.8, 'xGA': 1.0, 'predicted_score': '2-1', 'home_win_prob': 0.61, 'draw_prob': 0.23, 'away_win_prob': 0.16},
        {'date': datetime(2025, 2, 22), 'opponent': 'Nottingham Forest', 'venue': 'Away', 'xPTS': 1.5, 'xG': 1.4, 'xGA': 1.3, 'predicted_score': '1-1', 'home_win_prob': 0.38, 'draw_prob': 0.32, 'away_win_prob': 0.30},
        {'date': datetime(2025, 3, 1), 'opponent': 'Bournemouth', 'venue': 'Home', 'xPTS': 2.0, 'xG': 1.7, 'xGA': 1.1, 'predicted_score': '2-1', 'home_win_prob': 0.59, 'draw_prob': 0.24, 'away_win_prob': 0.17},
        {'date': datetime(2025, 3, 8), 'opponent': 'Arsenal', 'venue': 'Away', 'xPTS': 0.6, 'xG': 1.0, 'xGA': 2.0, 'predicted_score': '1-2', 'home_win_prob': 0.17, 'draw_prob': 0.24, 'away_win_prob': 0.59},
        {'date': datetime(2025, 3, 15), 'opponent': 'Brighton', 'venue': 'Home', 'xPTS': 1.7, 'xG': 1.5, 'xGA': 1.2, 'predicted_score': '2-1', 'home_win_prob': 0.48, 'draw_prob': 0.28, 'away_win_prob': 0.24},
        {'date': datetime(2025, 4, 5), 'opponent': 'Leicester City', 'venue': 'Away', 'xPTS': 1.8, 'xG': 1.6, 'xGA': 1.2, 'predicted_score': '2-1', 'home_win_prob': 0.52, 'draw_prob': 0.26, 'away_win_prob': 0.22},
        {'date': datetime(2025, 4, 12), 'opponent': 'Fulham', 'venue': 'Home', 'xPTS': 1.9, 'xG': 1.6, 'xGA': 1.1, 'predicted_score': '2-1', 'home_win_prob': 0.55, 'draw_prob': 0.25, 'away_win_prob': 0.20},
        {'date': datetime(2025, 4, 19), 'opponent': 'Chelsea', 'venue': 'Away', 'xPTS': 0.9, 'xG': 1.2, 'xGA': 1.7, 'predicted_score': '1-2', 'home_win_prob': 0.24, 'draw_prob': 0.28, 'away_win_prob': 0.48},
        {'date': datetime(2025, 4, 26), 'opponent': 'Aston Villa', 'venue': 'Home', 'xPTS': 1.4, 'xG': 1.4, 'xGA': 1.4, 'predicted_score': '1-1', 'home_win_prob': 0.35, 'draw_prob': 0.32, 'away_win_prob': 0.33},
        {'date': datetime(2025, 5, 3), 'opponent': 'Newcastle', 'venue': 'Away', 'xPTS': 1.1, 'xG': 1.3, 'xGA': 1.6, 'predicted_score': '1-2', 'home_win_prob': 0.28, 'draw_prob': 0.30, 'away_win_prob': 0.42},
        {'date': datetime(2025, 5, 10), 'opponent': 'Southampton', 'venue': 'Home', 'xPTS': 2.3, 'xG': 2.0, 'xGA': 0.9, 'predicted_score': '2-0', 'home_win_prob': 0.69, 'draw_prob': 0.20, 'away_win_prob': 0.11},
        {'date': datetime(2025, 5, 18), 'opponent': 'Man United', 'venue': 'Away', 'xPTS': 1.2, 'xG': 1.3, 'xGA': 1.5, 'predicted_score': '1-2', 'home_win_prob': 0.31, 'draw_prob': 0.30, 'away_win_prob': 0.39},
        {'date': datetime(2025, 5, 25), 'opponent': 'Tottenham', 'venue': 'Home', 'xPTS': 1.3, 'xG': 1.4, 'xGA': 1.5, 'predicted_score': '1-1', 'home_win_prob': 0.33, 'draw_prob': 0.31, 'away_win_prob': 0.36},
    ]

    # Mock played matches
    played_matches = [
        {'date': datetime(2024, 8, 18), 'opponent': 'Liverpool', 'venue': 'Home', 'score': '0-2', 'brentford_goals': 0, 'opponent_goals': 2, 'result': 'L', 'actual_pts': 0},
        {'date': datetime(2024, 8, 24), 'opponent': 'Crystal Palace', 'venue': 'Away', 'score': '1-2', 'brentford_goals': 1, 'opponent_goals': 2, 'result': 'L', 'actual_pts': 0},
        {'date': datetime(2024, 8, 31), 'opponent': 'Southampton', 'venue': 'Home', 'score': '3-1', 'brentford_goals': 3, 'opponent_goals': 1, 'result': 'W', 'actual_pts': 3},
        {'date': datetime(2024, 9, 14), 'opponent': 'Man United', 'venue': 'Away', 'score': '2-1', 'brentford_goals': 2, 'opponent_goals': 1, 'result': 'W', 'actual_pts': 3},
        {'date': datetime(2024, 9, 21), 'opponent': 'Tottenham', 'venue': 'Home', 'score': '3-1', 'brentford_goals': 3, 'opponent_goals': 1, 'result': 'W', 'actual_pts': 3},
        {'date': datetime(2024, 9, 28), 'opponent': 'West Ham', 'venue': 'Away', 'score': '1-1', 'brentford_goals': 1, 'opponent_goals': 1, 'result': 'D', 'actual_pts': 1},
        {'date': datetime(2024, 10, 5), 'opponent': 'Wolves', 'venue': 'Home', 'score': '5-3', 'brentford_goals': 5, 'opponent_goals': 3, 'result': 'W', 'actual_pts': 3},
        {'date': datetime(2024, 10, 19), 'opponent': 'Fulham', 'venue': 'Away', 'score': '1-1', 'brentford_goals': 1, 'opponent_goals': 1, 'result': 'D', 'actual_pts': 1},
        {'date': datetime(2024, 10, 26), 'opponent': 'Ipswich Town', 'venue': 'Home', 'score': '4-3', 'brentford_goals': 4, 'opponent_goals': 3, 'result': 'W', 'actual_pts': 3},
        {'date': datetime(2024, 11, 2), 'opponent': 'Sheffield United', 'venue': 'Away', 'score': '2-0', 'brentford_goals': 2, 'opponent_goals': 0, 'result': 'W', 'actual_pts': 3},
        {'date': datetime(2024, 11, 9), 'opponent': 'Bournemouth', 'venue': 'Home', 'score': '2-1', 'brentford_goals': 2, 'opponent_goals': 1, 'result': 'W', 'actual_pts': 3},
        {'date': datetime(2024, 11, 23), 'opponent': 'Everton', 'venue': 'Away', 'score': '0-0', 'brentford_goals': 0, 'opponent_goals': 0, 'result': 'D', 'actual_pts': 1},
        {'date': datetime(2024, 11, 30), 'opponent': 'Leicester City', 'venue': 'Home', 'score': '4-1', 'brentford_goals': 4, 'opponent_goals': 1, 'result': 'W', 'actual_pts': 3},
        {'date': datetime(2024, 12, 7), 'opponent': 'Newcastle', 'venue': 'Away', 'score': '2-2', 'brentford_goals': 2, 'opponent_goals': 2, 'result': 'D', 'actual_pts': 1},
        {'date': datetime(2024, 12, 14), 'opponent': 'Chelsea', 'venue': 'Home', 'score': '2-1', 'brentford_goals': 2, 'opponent_goals': 1, 'result': 'W', 'actual_pts': 3},
    ]

    # Combine into one dataframe
    all_matches_list = []

    for match in played_matches:
        match['played'] = True
        match['predicted_score'] = None
        match['xPTS'] = None
        match['xG'] = None
        match['xGA'] = None
        match['home_win_prob'] = None
        match['draw_prob'] = None
        match['away_win_prob'] = None
        all_matches_list.append(match)

    for match in future_matches:
        match['played'] = False
        match['score'] = None
        match['brentford_goals'] = None
        match['opponent_goals'] = None
        match['result'] = None
        match['actual_pts'] = None
        all_matches_list.append(match)

    all_matches = pd.DataFrame(all_matches_list)
    all_matches = all_matches.sort_values('date').reset_index(drop=True)

    return brentford_stats, all_matches

# ==============================
# INITIALIZE DATA
# ==============================

print("Initializing Brentford xPTS Dashboard...")
print("Using mock data for demonstration...")

brentford_stats, all_matches = create_mock_brentford_data()

# Calculate projections
future_matches = all_matches[~all_matches['played']]
total_xPTS = future_matches['xPTS'].sum()
total_xG = future_matches['xG'].sum()
total_xGA = future_matches['xGA'].sum()

projected_final_pts = brentford_stats['points'] + total_xPTS
projected_final_gf = brentford_stats['goals_for'] + total_xG
projected_final_ga = brentford_stats['goals_against'] + total_xGA
projected_final_gd = projected_final_gf - projected_final_ga

print(f"Current Points: {brentford_stats['points']}")
print(f"Remaining xPTS: {total_xPTS:.1f}")
print(f"Projected Final Points: {projected_final_pts:.1f}")

# ==============================
# VISUALIZATION FUNCTIONS
# ==============================

def create_xpts_progression_graph(matches_df, current_stats):
    """Create a line graph showing cumulative points progression"""
    if matches_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title='Brentford Points Progression (2024/25 Season)',
            template='plotly_dark',
            height=500,
        )
        return fig

    matches_df = matches_df.copy()

    # Calculate cumulative points
    actual_pts_list = []
    projected_pts_list = []
    dates = []

    cumulative_actual = 0
    cumulative_projected = current_stats['points']

    for idx, match in matches_df.iterrows():
        dates.append(match['date'])

        if match['played']:
            cumulative_actual += match['actual_pts']
            actual_pts_list.append(cumulative_actual)
            projected_pts_list.append(cumulative_actual)
        else:
            actual_pts_list.append(cumulative_actual)
            cumulative_projected += match['xPTS']
            projected_pts_list.append(cumulative_projected)

    # Create figure
    fig = go.Figure()

    # Add actual points line
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_pts_list,
        mode='lines+markers',
        name='Actual Points',
        line=dict(color='#00ff00', width=3),
        marker=dict(size=8)
    ))

    # Add projected points line
    fig.add_trace(go.Scatter(
        x=dates,
        y=projected_pts_list,
        mode='lines+markers',
        name='Projected Points',
        line=dict(color='#ff6b6b', width=3, dash='dash'),
        marker=dict(size=8)
    ))

    # Add vertical line at current date using a shape instead
    current_date = datetime.now()
    fig.add_shape(
        type="line",
        x0=current_date,
        x1=current_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="yellow", width=2, dash="dot")
    )
    fig.add_annotation(
        x=current_date,
        y=1,
        yref="paper",
        text="Today",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="yellow")
    )

    fig.update_layout(
        title='Brentford Points Progression (2024/25 Season)',
        xaxis_title='Date',
        yaxis_title='Cumulative Points',
        hovermode='x unified',
        template='plotly_dark',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

def create_matches_table(matches_df):
    """Create a detailed table of all matches"""
    if matches_df.empty:
        return pd.DataFrame([{'Date': 'N/A', 'Opponent': 'No data', 'Venue': 'N/A', 'Status': 'No data available'}])

    display_df = matches_df.copy()
    display_df['Date'] = display_df['date'].dt.strftime('%Y-%m-%d')

    table_data = []
    for idx, row in display_df.iterrows():
        if row['played']:
            table_data.append({
                'Date': row['Date'],
                'Opponent': row['opponent'],
                'Venue': row['venue'],
                'Score': row['score'],
                'Result': row['result'],
                'Points': row['actual_pts'],
                'Status': 'Played'
            })
        else:
            table_data.append({
                'Date': row['Date'],
                'Opponent': row['opponent'],
                'Venue': row['venue'],
                'Predicted Score': row['predicted_score'],
                'xPTS': f"{row['xPTS']:.2f}",
                'xG': f"{row['xG']:.2f}",
                'xGA': f"{row['xGA']:.2f}",
                'Win%': f"{row['home_win_prob']*100:.0f}%" if row['home_win_prob'] else 'N/A',
                'Draw%': f"{row['draw_prob']*100:.0f}%" if row['draw_prob'] else 'N/A',
                'Status': 'Upcoming'
            })

    return pd.DataFrame(table_data)

# ==============================
# INITIALIZE DASH APP
# ==============================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# ==============================
# APP LAYOUT
# ==============================

app.layout = dbc.Container([
    # Header
    html.H1("🐝 Brentford xPTS Tracker", className="text-center mb-1", style={'color': '#ff6b6b'}),
    html.H5("2024/25 Premier League Season Projection", className="text-center mb-2 text-muted"),
    html.P("⚠️ Using mock data for demonstration. Replace with live data when fbref access is available.",
           className="text-center text-warning small mb-4"),

    # Summary Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Current Points", className="text-center"),
                    html.H2(f"{brentford_stats['points']}", className="text-center text-success"),
                    html.P(f"From {brentford_stats['matches_played']} matches", className="text-center text-muted small")
                ])
            ], color="dark")
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Projected Final Points", className="text-center"),
                    html.H2(f"{projected_final_pts:.1f}", className="text-center text-warning"),
                    html.P(f"+{total_xPTS:.1f} xPTS remaining", className="text-center text-muted small")
                ])
            ], color="dark")
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Projected Goals For", className="text-center"),
                    html.H2(f"{projected_final_gf:.1f}", className="text-center text-info"),
                    html.P(f"Currently: {brentford_stats['goals_for']}", className="text-center text-muted small")
                ])
            ], color="dark")
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Projected Goal Diff", className="text-center"),
                    html.H2(f"{projected_final_gd:+.1f}", className="text-center text-primary"),
                    html.P(f"Currently: {brentford_stats['goal_difference']:+d}", className="text-center text-muted small")
                ])
            ], color="dark")
        ], width=3),
    ], className="mb-4"),

    # Points Progression Graph
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        id='xpts-graph',
                        figure=create_xpts_progression_graph(all_matches, brentford_stats)
                    )
                ])
            ], color="dark")
        ], width=12)
    ], className="mb-4"),

    # Matches Table
    dbc.Row([
        dbc.Col([
            html.H3("All Matches", className="text-center mb-3"),
            dbc.Tabs([
                dbc.Tab([
                    html.Div([
                        DataTable(
                            data=create_matches_table(all_matches[all_matches['played']]).to_dict('records'),
                            columns=[
                                {'name': 'Date', 'id': 'Date'},
                                {'name': 'Opponent', 'id': 'Opponent'},
                                {'name': 'Venue', 'id': 'Venue'},
                                {'name': 'Score', 'id': 'Score'},
                                {'name': 'Result', 'id': 'Result'},
                                {'name': 'Points', 'id': 'Points'}
                            ],
                            style_cell={
                                'textAlign': 'center',
                                'backgroundColor': '#2c3e50',
                                'color': 'white',
                                'border': '1px solid #555',
                                'padding': '10px'
                            },
                            style_header={
                                'backgroundColor': '#1a252f',
                                'fontWeight': 'bold',
                                'border': '1px solid #555'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'Result', 'filter_query': '{Result} = W'},
                                    'color': '#00ff00',
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {'column_id': 'Result', 'filter_query': '{Result} = L'},
                                    'color': '#ff6b6b',
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {'column_id': 'Result', 'filter_query': '{Result} = D'},
                                    'color': '#ffa500',
                                    'fontWeight': 'bold'
                                }
                            ]
                        )
                    ], className="p-3")
                ], label="Played Matches", tab_id="played"),

                dbc.Tab([
                    html.Div([
                        DataTable(
                            data=create_matches_table(all_matches[~all_matches['played']]).to_dict('records'),
                            columns=[
                                {'name': 'Date', 'id': 'Date'},
                                {'name': 'Opponent', 'id': 'Opponent'},
                                {'name': 'Venue', 'id': 'Venue'},
                                {'name': 'Predicted Score', 'id': 'Predicted Score'},
                                {'name': 'xPTS', 'id': 'xPTS'},
                                {'name': 'xG', 'id': 'xG'},
                                {'name': 'xGA', 'id': 'xGA'},
                                {'name': 'Win%', 'id': 'Win%'},
                                {'name': 'Draw%', 'id': 'Draw%'}
                            ],
                            style_cell={
                                'textAlign': 'center',
                                'backgroundColor': '#2c3e50',
                                'color': 'white',
                                'border': '1px solid #555',
                                'padding': '10px'
                            },
                            style_header={
                                'backgroundColor': '#1a252f',
                                'fontWeight': 'bold',
                                'border': '1px solid #555'
                            }
                        )
                    ], className="p-3")
                ], label="Upcoming Matches", tab_id="upcoming")
            ], id="matches-tabs", active_tab="upcoming")
        ], width=12)
    ]),

    # Footer note
    html.Hr(),
    html.P([
        "📊 This dashboard uses a Poisson-based prediction model similar to prem_dash_v4.py. ",
        "Expected Points (xPTS) are calculated as: (Win Probability × 3) + (Draw Probability × 1). ",
        "The model considers team attacking and defensive strength when playing home vs away."
    ], className="text-center text-muted small mt-4")

], fluid=True, className="p-4")

# ==============================
# RUN THE APP
# ==============================
if __name__ == '__main__':
    print("\nStarting Brentford xPTS Dashboard on port 8899...")
    print("Access the dashboard at: http://localhost:8899")
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Current Points: {brentford_stats['points']}")
    print(f"  Matches Played: {brentford_stats['matches_played']}")
    print(f"  Remaining xPTS: {total_xPTS:.2f}")
    print(f"  Projected Final Points: {projected_final_pts:.1f}")
    print(f"  Projected Final GD: {projected_final_gd:+.1f}")
    print("="*60 + "\n")
    app.run(debug=True, port=8899)

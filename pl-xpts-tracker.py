import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ==============================
# MOCK DATA FOR ALL PL TEAMS
# ==============================

PREMIER_LEAGUE_TEAMS = [
    'Liverpool', 'Arsenal', 'Man City', 'Chelsea', 'Newcastle',
    'Man United', 'Tottenham', 'Brighton', 'Aston Villa', 'West Ham',
    'Brentford', 'Fulham', 'Crystal Palace', 'Bournemouth', 'Nottingham Forest',
    'Everton', 'Leicester City', 'Wolves', 'Ipswich Town', 'Southampton'
]

def generate_team_current_stats():
    """Generate realistic current stats for all teams (as of matchday 15)"""
    # Realistic points distribution for top 20 teams at matchday 15
    team_data = {
        'Liverpool': {'points': 34, 'mp': 15, 'w': 11, 'd': 1, 'l': 3, 'gf': 32, 'ga': 14},
        'Arsenal': {'points': 31, 'mp': 15, 'w': 9, 'd': 4, 'l': 2, 'gf': 28, 'ga': 13},
        'Man City': {'points': 30, 'mp': 15, 'w': 9, 'd': 3, 'l': 3, 'gf': 30, 'ga': 16},
        'Chelsea': {'points': 28, 'mp': 15, 'w': 8, 'd': 4, 'l': 3, 'gf': 29, 'ga': 16},
        'Newcastle': {'points': 26, 'mp': 15, 'w': 8, 'd': 2, 'l': 5, 'gf': 24, 'ga': 17},
        'Man United': {'points': 25, 'mp': 15, 'w': 7, 'd': 4, 'l': 4, 'gf': 21, 'ga': 18},
        'Tottenham': {'points': 24, 'mp': 15, 'w': 7, 'd': 3, 'l': 5, 'gf': 27, 'ga': 19},
        'Brighton': {'points': 24, 'mp': 15, 'w': 7, 'd': 3, 'l': 5, 'gf': 23, 'ga': 20},
        'Aston Villa': {'points': 23, 'mp': 15, 'w': 6, 'd': 5, 'l': 4, 'gf': 22, 'ga': 21},
        'Brentford': {'points': 23, 'mp': 15, 'w': 7, 'd': 2, 'l': 6, 'gf': 26, 'ga': 23},
        'Fulham': {'points': 22, 'mp': 15, 'w': 6, 'd': 4, 'l': 5, 'gf': 21, 'ga': 19},
        'West Ham': {'points': 20, 'mp': 15, 'w': 5, 'd': 5, 'l': 5, 'gf': 18, 'ga': 20},
        'Bournemouth': {'points': 19, 'mp': 15, 'w': 5, 'd': 4, 'l': 6, 'gf': 19, 'ga': 22},
        'Crystal Palace': {'points': 18, 'mp': 15, 'w': 5, 'd': 3, 'l': 7, 'gf': 17, 'ga': 21},
        'Nottingham Forest': {'points': 18, 'mp': 15, 'w': 5, 'd': 3, 'l': 7, 'gf': 16, 'ga': 22},
        'Everton': {'points': 15, 'mp': 15, 'w': 4, 'd': 3, 'l': 8, 'gf': 14, 'ga': 23},
        'Wolves': {'points': 12, 'mp': 15, 'w': 3, 'd': 3, 'l': 9, 'gf': 18, 'ga': 28},
        'Leicester City': {'points': 11, 'mp': 15, 'w': 2, 'd': 5, 'l': 8, 'gf': 16, 'ga': 28},
        'Ipswich Town': {'points': 9, 'mp': 15, 'w': 2, 'd': 3, 'l': 10, 'gf': 13, 'ga': 29},
        'Southampton': {'points': 5, 'mp': 15, 'w': 1, 'd': 2, 'l': 12, 'gf': 10, 'ga': 32}
    }

    # Calculate additional stats
    for team in team_data:
        stats = team_data[team]
        stats['gd'] = stats['gf'] - stats['ga']

    return team_data

def generate_remaining_fixtures():
    """Generate remaining fixtures for all teams (23 matches per team)"""
    base_date = datetime.now() + timedelta(days=7)
    all_fixtures = []

    # Team strength ratings (higher = stronger)
    team_strength = {
        'Liverpool': 90, 'Arsenal': 87, 'Man City': 88, 'Chelsea': 82, 'Newcastle': 78,
        'Man United': 76, 'Tottenham': 75, 'Brighton': 74, 'Aston Villa': 73, 'Brentford': 70,
        'Fulham': 68, 'West Ham': 67, 'Bournemouth': 64, 'Crystal Palace': 62,
        'Nottingham Forest': 61, 'Everton': 58, 'Wolves': 55, 'Leicester City': 53,
        'Ipswich Town': 50, 'Southampton': 45
    }

    # Generate round-robin fixtures (simplified - each team plays remaining 23 games)
    matchday = 16
    for i in range(23):
        matchday_date = base_date + timedelta(days=i*7)

        # Create matchups (simplified random pairing)
        teams_copy = PREMIER_LEAGUE_TEAMS.copy()
        np.random.seed(42 + i)  # Consistent randomness
        np.random.shuffle(teams_copy)

        # Pair teams
        for j in range(0, len(teams_copy), 2):
            if j + 1 < len(teams_copy):
                home_team = teams_copy[j]
                away_team = teams_copy[j + 1]

                # Calculate xG based on team strength
                home_strength = team_strength[home_team]
                away_strength = team_strength[away_team]

                # Home advantage
                home_xg = 1.0 + (home_strength - away_strength) / 50 + 0.3
                away_xg = 1.0 + (away_strength - home_strength) / 50

                # Ensure minimum values
                home_xg = max(0.5, min(3.0, home_xg))
                away_xg = max(0.5, min(3.0, away_xg))

                # Calculate probabilities (simplified)
                strength_diff = home_strength - away_strength
                home_win_prob = 0.35 + (strength_diff / 200)
                away_win_prob = 0.25 - (strength_diff / 250)
                draw_prob = 1.0 - home_win_prob - away_win_prob

                # Clamp probabilities
                home_win_prob = max(0.10, min(0.75, home_win_prob))
                away_win_prob = max(0.10, min(0.65, away_win_prob))
                draw_prob = 1.0 - home_win_prob - away_win_prob

                # Calculate xPTS for each team
                home_xpts = home_win_prob * 3 + draw_prob * 1
                away_xpts = away_win_prob * 3 + draw_prob * 1

                # Predict score
                home_goals = round(home_xg)
                away_goals = round(away_xg)
                if home_goals == away_goals:
                    if home_win_prob > away_win_prob:
                        home_goals += 1
                    elif away_win_prob > home_win_prob:
                        away_goals += 1

                all_fixtures.append({
                    'date': matchday_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_xg': round(home_xg, 2),
                    'away_xg': round(away_xg, 2),
                    'home_xpts': round(home_xpts, 2),
                    'away_xpts': round(away_xpts, 2),
                    'home_win_prob': round(home_win_prob, 3),
                    'draw_prob': round(draw_prob, 3),
                    'away_win_prob': round(away_win_prob, 3),
                    'predicted_score': f"{home_goals}-{away_goals}",
                    'matchday': matchday
                })

        matchday += 1

    return pd.DataFrame(all_fixtures)

def calculate_team_projections(team_stats, fixtures_df):
    """Calculate end-of-season projections for all teams"""
    projections = []

    for team in PREMIER_LEAGUE_TEAMS:
        current = team_stats[team]

        # Get remaining fixtures for this team
        home_fixtures = fixtures_df[fixtures_df['home_team'] == team]
        away_fixtures = fixtures_df[fixtures_df['away_team'] == team]

        # Calculate remaining xPTS and xG
        remaining_xpts = home_fixtures['home_xpts'].sum() + away_fixtures['away_xpts'].sum()
        remaining_xgf = home_fixtures['home_xg'].sum() + away_fixtures['away_xg'].sum()
        remaining_xga = home_fixtures['away_xg'].sum() + away_fixtures['home_xg'].sum()

        # Project final stats
        projections.append({
            'team': team,
            'current_pts': current['points'],
            'current_mp': current['mp'],
            'current_gf': current['gf'],
            'current_ga': current['ga'],
            'current_gd': current['gd'],
            'remaining_xpts': round(remaining_xpts, 1),
            'remaining_xgf': round(remaining_xgf, 1),
            'remaining_xga': round(remaining_xga, 1),
            'projected_pts': round(current['points'] + remaining_xpts, 1),
            'projected_gf': round(current['gf'] + remaining_xgf, 1),
            'projected_ga': round(current['ga'] + remaining_xga, 1),
            'projected_gd': round(current['gd'] + remaining_xgf - remaining_xga, 1)
        })

    return pd.DataFrame(projections).sort_values('projected_pts', ascending=False)

def get_team_matches(team, fixtures_df, team_stats):
    """Get all matches (played and upcoming) for a specific team"""
    all_matches = []

    # For simplicity, we'll generate some played matches
    played_count = team_stats[team]['mp']
    base_date = datetime.now() - timedelta(days=played_count * 7)

    # Generate played matches (simplified)
    current = team_stats[team]
    for i in range(played_count):
        # Distribute wins/draws/losses across played matches
        if i < current['w']:
            result = 'W'
            pts = 3
            gf = 2
            ga = 1
        elif i < current['w'] + current['d']:
            result = 'D'
            pts = 1
            gf = 1
            ga = 1
        else:
            result = 'L'
            pts = 0
            gf = 1
            ga = 2

        all_matches.append({
            'date': base_date + timedelta(days=i*7),
            'opponent': 'Opponent',  # Simplified
            'venue': 'Home' if i % 2 == 0 else 'Away',
            'played': True,
            'score': f"{gf}-{ga}",
            'result': result,
            'actual_pts': pts,
            'predicted_score': None,
            'xPTS': None,
            'xG': None,
            'xGA': None,
            'home_win_prob': None,
            'draw_prob': None,
            'away_win_prob': None
        })

    # Get upcoming fixtures
    home_fixtures = fixtures_df[fixtures_df['home_team'] == team].copy()
    away_fixtures = fixtures_df[fixtures_df['away_team'] == team].copy()

    for _, fixture in home_fixtures.iterrows():
        all_matches.append({
            'date': fixture['date'],
            'opponent': fixture['away_team'],
            'venue': 'Home',
            'played': False,
            'score': None,
            'result': None,
            'actual_pts': None,
            'predicted_score': fixture['predicted_score'],
            'xPTS': fixture['home_xpts'],
            'xG': fixture['home_xg'],
            'xGA': fixture['away_xg'],
            'home_win_prob': fixture['home_win_prob'],
            'draw_prob': fixture['draw_prob'],
            'away_win_prob': fixture['away_win_prob']
        })

    for _, fixture in away_fixtures.iterrows():
        all_matches.append({
            'date': fixture['date'],
            'opponent': fixture['home_team'],
            'venue': 'Away',
            'played': False,
            'score': None,
            'result': None,
            'actual_pts': None,
            'predicted_score': f"{fixture['predicted_score'].split('-')[1]}-{fixture['predicted_score'].split('-')[0]}",
            'xPTS': fixture['away_xpts'],
            'xG': fixture['away_xg'],
            'xGA': fixture['home_xg'],
            'home_win_prob': fixture['away_win_prob'],
            'draw_prob': fixture['draw_prob'],
            'away_win_prob': fixture['home_win_prob']
        })

    return pd.DataFrame(all_matches).sort_values('date').reset_index(drop=True)

# ==============================
# INITIALIZE DATA
# ==============================

print("Initializing Premier League xPTS Tracker...")
print("Generating data for all 20 teams...")

team_stats = generate_team_current_stats()
fixtures_df = generate_remaining_fixtures()
projections_df = calculate_team_projections(team_stats, fixtures_df)

print(f"✓ Generated {len(fixtures_df)} remaining fixtures")
print(f"✓ Calculated projections for all {len(PREMIER_LEAGUE_TEAMS)} teams")

# ==============================
# VISUALIZATION FUNCTIONS
# ==============================

def create_league_table(projections_df):
    """Create the projected league table"""
    table_df = projections_df.copy()
    table_df['position'] = range(1, len(table_df) + 1)

    # Reorder columns
    display_df = table_df[[
        'position', 'team', 'current_pts', 'remaining_xpts', 'projected_pts',
        'current_gd', 'projected_gd'
    ]].copy()

    display_df.columns = [
        'Pos', 'Team', 'Current Pts', 'Remaining xPTS', 'Projected Pts',
        'Current GD', 'Projected GD'
    ]

    return display_df

def create_team_progression_graph(team_matches, team, current_stats):
    """Create points progression graph for a specific team"""
    if team_matches.empty:
        fig = go.Figure()
        fig.update_layout(title=f'{team} Points Progression', template='plotly_dark', height=400)
        return fig

    team_matches = team_matches.copy()

    actual_pts_list = []
    projected_pts_list = []
    dates = []

    cumulative_actual = 0
    cumulative_projected = current_stats['points']

    for _, match in team_matches.iterrows():
        dates.append(match['date'])

        if match['played']:
            cumulative_actual += match['actual_pts']
            actual_pts_list.append(cumulative_actual)
            projected_pts_list.append(cumulative_actual)
        else:
            actual_pts_list.append(cumulative_actual)
            cumulative_projected += match['xPTS']
            projected_pts_list.append(cumulative_projected)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=actual_pts_list,
        mode='lines+markers',
        name='Actual Points',
        line=dict(color='#00ff00', width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=projected_pts_list,
        mode='lines+markers',
        name='Projected Points',
        line=dict(color='#ff6b6b', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    current_date = datetime.now()
    fig.add_shape(
        type="line", x0=current_date, x1=current_date,
        y0=0, y1=1, yref="paper",
        line=dict(color="yellow", width=2, dash="dot")
    )

    fig.update_layout(
        title=f'{team} - 2024/25 Season Progression',
        xaxis_title='Date',
        yaxis_title='Points',
        hovermode='x unified',
        template='plotly_dark',
        height=400,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig

def create_team_matches_table(team_matches):
    """Create matches table for a team"""
    if team_matches.empty:
        return pd.DataFrame([{'Date': 'N/A', 'Opponent': 'No data'}])

    display_df = team_matches.copy()
    display_df['Date'] = display_df['date'].dt.strftime('%Y-%m-%d')

    table_data = []
    for _, row in display_df.iterrows():
        if row['played']:
            table_data.append({
                'Date': row['Date'],
                'Opponent': row['opponent'],
                'Venue': row['venue'],
                'Score': row['score'],
                'Result': row['result'],
                'Points': row['actual_pts']
            })
        else:
            table_data.append({
                'Date': row['Date'],
                'Opponent': row['opponent'],
                'Venue': row['venue'],
                'Predicted Score': row['predicted_score'],
                'xPTS': f"{row['xPTS']:.2f}",
                'xG': f"{row['xG']:.2f}",
                'xGA': f"{row['xGA']:.2f}"
            })

    return pd.DataFrame(table_data)

# ==============================
# DASH APP
# ==============================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# ==============================
# LAYOUTS
# ==============================

def create_home_layout():
    """Create the home page with league table"""
    league_table_df = create_league_table(projections_df)

    return dbc.Container([
        html.H1("⚽ Premier League xPTS Tracker", className="text-center mb-3 mt-3"),
        html.H4("2024/25 Season Projections", className="text-center text-muted mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Projected Final Standings", className="text-center")),
                    dbc.CardBody([
                        DataTable(
                            data=league_table_df.to_dict('records'),
                            columns=[{'name': col, 'id': col} for col in league_table_df.columns],
                            style_cell={
                                'textAlign': 'center',
                                'backgroundColor': '#2c3e50',
                                'color': 'white',
                                'border': '1px solid #555',
                                'padding': '12px',
                                'fontSize': '14px'
                            },
                            style_header={
                                'backgroundColor': '#1a252f',
                                'fontWeight': 'bold',
                                'border': '1px solid #555',
                                'fontSize': '15px'
                            },
                            style_data_conditional=[
                                # Top 4 (Champions League)
                                {'if': {'filter_query': '{Pos} <= 4'}, 'backgroundColor': '#1e3a5f', 'fontWeight': 'bold'},
                                # 5th (Europa League)
                                {'if': {'filter_query': '{Pos} = 5'}, 'backgroundColor': '#2d4a2f'},
                                # Bottom 3 (Relegation)
                                {'if': {'filter_query': '{Pos} >= 18'}, 'backgroundColor': '#4a2d2d'}
                            ],
                            page_size=20
                        )
                    ])
                ], color="dark")
            ], width=10)
        ], justify='center'),

        html.Hr(),
        html.Div([
            html.P("🔵 Top 4: UEFA Champions League", className="text-info"),
            html.P("🟢 5th: UEFA Europa League", className="text-success"),
            html.P("🔴 Bottom 3: Relegation", className="text-danger")
        ], className="text-center small")

    ], fluid=True)

def create_team_layout(team):
    """Create individual team page"""
    team_matches = get_team_matches(team, fixtures_df, team_stats)
    team_proj = projections_df[projections_df['team'] == team].iloc[0]
    current = team_stats[team]

    played_matches = team_matches[team_matches['played']]
    upcoming_matches = team_matches[~team_matches['played']]

    return dbc.Container([
        html.H2(f"{team}", className="text-center mb-3 mt-3"),

        # Summary cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Current Points"),
                        html.H2(f"{current['points']}", className="text-success"),
                        html.P(f"Position: {list(projections_df['team']).index(team) + 1}", className="small text-muted")
                    ])
                ], color="dark")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Projected Points"),
                        html.H2(f"{team_proj['projected_pts']:.1f}", className="text-warning"),
                        html.P(f"+{team_proj['remaining_xpts']:.1f} xPTS", className="small text-muted")
                    ])
                ], color="dark")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Current GD"),
                        html.H2(f"{current['gd']:+d}", className="text-info"),
                        html.P(f"{current['gf']}-{current['ga']}", className="small text-muted")
                    ])
                ], color="dark")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Projected GD"),
                        html.H2(f"{team_proj['projected_gd']:+.1f}", className="text-primary"),
                        html.P(f"{team_proj['projected_gf']:.0f}-{team_proj['projected_ga']:.0f}", className="small text-muted")
                    ])
                ], color="dark")
            ], width=3)
        ], className="mb-4"),

        # Progression graph
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            figure=create_team_progression_graph(team_matches, team, current)
                        )
                    ])
                ], color="dark")
            ])
        ], className="mb-4"),

        # Matches tables
        dbc.Row([
            dbc.Col([
                html.H4("Matches", className="text-center mb-3"),
                dbc.Tabs([
                    dbc.Tab([
                        DataTable(
                            data=create_team_matches_table(played_matches).to_dict('records'),
                            columns=[
                                {'name': 'Date', 'id': 'Date'},
                                {'name': 'Opponent', 'id': 'Opponent'},
                                {'name': 'Venue', 'id': 'Venue'},
                                {'name': 'Score', 'id': 'Score'},
                                {'name': 'Result', 'id': 'Result'},
                                {'name': 'Points', 'id': 'Points'}
                            ],
                            style_cell={'textAlign': 'center', 'backgroundColor': '#2c3e50', 'color': 'white', 'padding': '10px'},
                            style_header={'backgroundColor': '#1a252f', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {'if': {'column_id': 'Result', 'filter_query': '{Result} = W'}, 'color': '#00ff00', 'fontWeight': 'bold'},
                                {'if': {'column_id': 'Result', 'filter_query': '{Result} = L'}, 'color': '#ff6b6b', 'fontWeight': 'bold'},
                                {'if': {'column_id': 'Result', 'filter_query': '{Result} = D'}, 'color': '#ffa500', 'fontWeight': 'bold'}
                            ],
                            page_size=15
                        )
                    ], label=f"Played ({len(played_matches)})"),

                    dbc.Tab([
                        DataTable(
                            data=create_team_matches_table(upcoming_matches).to_dict('records'),
                            columns=[
                                {'name': 'Date', 'id': 'Date'},
                                {'name': 'Opponent', 'id': 'Opponent'},
                                {'name': 'Venue', 'id': 'Venue'},
                                {'name': 'Predicted Score', 'id': 'Predicted Score'},
                                {'name': 'xPTS', 'id': 'xPTS'},
                                {'name': 'xG', 'id': 'xG'},
                                {'name': 'xGA', 'id': 'xGA'}
                            ],
                            style_cell={'textAlign': 'center', 'backgroundColor': '#2c3e50', 'color': 'white', 'padding': '10px'},
                            style_header={'backgroundColor': '#1a252f', 'fontWeight': 'bold'},
                            page_size=25
                        )
                    ], label=f"Upcoming ({len(upcoming_matches)})")
                ])
            ])
        ])

    ], fluid=True)

# Main layout with tabs for each team
app.layout = dbc.Container([
    dcc.Tabs(id='main-tabs', value='home', children=[
        dcc.Tab(label='🏠 Home', value='home'),
        *[dcc.Tab(label=team, value=team) for team in PREMIER_LEAGUE_TEAMS]
    ]),
    html.Div(id='page-content')
], fluid=True, style={'padding': '0'})

# ==============================
# CALLBACKS
# ==============================

@app.callback(
    Output('page-content', 'children'),
    Input('main-tabs', 'value')
)
def render_content(tab):
    if tab == 'home':
        return create_home_layout()
    else:
        return create_team_layout(tab)

# ==============================
# RUN APP
# ==============================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Premier League xPTS Tracker - Starting on port 8900")
    print("="*70)
    print(f"✓ {len(PREMIER_LEAGUE_TEAMS)} teams loaded")
    print(f"✓ {len(fixtures_df)} remaining fixtures generated")
    print(f"✓ Projected winner: {projections_df.iloc[0]['team']} ({projections_df.iloc[0]['projected_pts']:.1f} pts)")
    print("="*70 + "\n")
    app.run(debug=True, port=8900)

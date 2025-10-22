import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import pandas as pd
import numpy as np
import scipy.stats as stats
import requests
from datetime import datetime
import plotly.graph_objects as go
from io import StringIO
import time

# ==============================
# DATA FETCHING (from prem_dash_v4.py)
# ==============================

def fetch_team_stats():
    """Fetch team performance data from fbref.com for Premier League"""
    print("Waiting 10 seconds before fetching (rate limiting)...")
    time.sleep(10)
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    # Try multiple times with exponential backoff
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            dfs = pd.read_html(StringIO(response.text))
            print("Successfully fetched team stats!")
            break
        except Exception as e:
            if attempt < 2:
                wait_time = (attempt + 1) * 10
                print(f"Attempt {attempt + 1} failed: {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise

    df = dfs[1]
    df.columns = [
        "Rk","Squad",
        "Home_MP", "Home_W", "Home_D", "Home_L", "Home_GF", "Home_GA", "Home_GD", "Home_Pts",'Home_Pts/MP', "Home_xG", "Home_xGA", "Home_xGD", "Home_xGD_per_90",
        "Away_MP", "Away_W", "Away_D", "Away_L", "Away_GF", "Away_GA", "Away_GD", "Away_Pts",'Away_Pts/MP', "Away_xG", "Away_xGA", "Away_xGD", "Away_xGD_per_90"]

    home_df = df[[
        "Rk", "Squad",
        "Home_MP", "Home_W", "Home_D", "Home_L", "Home_GF", "Home_GA",
        "Home_GD", "Home_Pts", "Home_xG", "Home_xGA", "Home_xGD", "Home_xGD_per_90"
    ]].copy()

    away_df = df[[
        "Rk", "Squad",
        "Away_MP", "Away_W", "Away_D", "Away_L", "Away_GF", "Away_GA",
        "Away_GD", "Away_Pts", "Away_xG", "Away_xGA", "Away_xGD", "Away_xGD_per_90"
    ]].copy()

    rename_home = {
        "Home_MP": "MP", "Home_W": "W", "Home_D": "D", "Home_L": "L",
        "Home_GF": "GF", "Home_GA": "GA", "Home_GD": "GD", "Home_Pts": "Pts",
        "Home_xG": "xG", "Home_xGA": "xGA", "Home_xGD": "xGD", "Home_xGD_per_90": "xGD_per_90"
    }

    rename_away = {
        "Away_MP": "MP", "Away_W": "W", "Away_D": "D", "Away_L": "L",
        "Away_GF": "GF", "Away_GA": "GA", "Away_GD": "GD", "Away_Pts": "Pts",
        "Away_xG": "xG", "Away_xGA": "xGA", "Away_xGD": "xGD", "Away_xGD_per_90": "xGD_per_90"
    }

    home_df.rename(columns=rename_home, inplace=True)
    away_df.rename(columns=rename_away, inplace=True)

    # Calculate weighted xG (70% xG, 30% actual goals)
    home_df['wxG'] = (home_df['xG'] * 0.7 + home_df['GF'] * 0.3).round(2)
    home_df['wxGA'] = (home_df['xGA'] * 0.7 + home_df['GA'] * 0.3).round(2)
    away_df['wxG'] = (away_df['xG'] * 0.7 + away_df['GF'] * 0.3).round(2)
    away_df['wxGA'] = (away_df['xGA'] * 0.7 + away_df['GA'] * 0.3).round(2)

    # Normalize per 90 minutes
    home_df['Normalized wxG/90'] = (home_df['wxG'] / home_df['MP']).round(2)
    away_df['Normalized wxG/90'] = (away_df['wxG'] / away_df['MP']).round(2)
    home_df['Normalized wxGA/90'] = (home_df['wxGA'] / home_df['MP']).round(2)
    away_df['Normalized wxGA/90'] = (away_df['wxGA'] / away_df['MP']).round(2)

    return home_df, away_df, df

def fetch_pl_schedule():
    """Fetch Premier League schedule"""
    print("Waiting 10 seconds before fetching schedule (rate limiting)...")
    time.sleep(10)
    url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            matches_df = pd.read_html(StringIO(response.text))[0]
            print("Successfully fetched schedule!")
            break
        except Exception as e:
            if attempt < 2:
                wait_time = (attempt + 1) * 10
                print(f"Attempt {attempt + 1} failed: {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise

    # Convert date
    matches_df['Date'] = pd.to_datetime(matches_df['Date'], errors='coerce')

    # Determine if match has been played
    matches_df['Played'] = ~matches_df['Score'].isna()

    return matches_df

# ==============================
# PREDICTION ENGINE (from prem_dash_v4.py)
# ==============================

def expected_goals(xG_team, xGA_opp, league_avg_xG):
    """Calculate expected goals based on team and opponent statistics"""
    return (xG_team / league_avg_xG) * (xGA_opp / league_avg_xG) * league_avg_xG

def poisson_prob_matrix(lambda_A, lambda_B, max_goals=10):
    """Generate a matrix of Poisson probabilities for different scorelines"""
    prob_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            prob_matrix[i, j] = stats.poisson.pmf(i, lambda_A) * stats.poisson.pmf(j, lambda_B)
    return prob_matrix

def adjust_draw_probability(home_win_prob, draw_prob, away_win_prob):
    """Adjust draw probability"""
    total_prob = home_win_prob + draw_prob + away_win_prob

    if home_win_prob > away_win_prob:
        favored_win_prob = home_win_prob
        underdog_win_prob = away_win_prob
    else:
        favored_win_prob = away_win_prob
        underdog_win_prob = home_win_prob

    adjusted_draw_prob = max(draw_prob, min(favored_win_prob * 0.75, 0.35))
    normalization_factor = total_prob / (favored_win_prob + adjusted_draw_prob + underdog_win_prob)

    return (
        home_win_prob * normalization_factor,
        adjusted_draw_prob * normalization_factor,
        away_win_prob * normalization_factor
    )

def find_most_likely_scores(prob_matrix, outcome_type, max_scores=3):
    """Find the most likely scores for a specific outcome type"""
    scores = []

    if outcome_type == 'home_win':
        mask = np.tril(np.ones_like(prob_matrix), -1).astype(bool)
    elif outcome_type == 'draw':
        mask = np.eye(prob_matrix.shape[0], dtype=bool)
    elif outcome_type == 'away_win':
        mask = np.triu(np.ones_like(prob_matrix), 1).astype(bool)
    else:
        raise ValueError("outcome_type must be 'home_win', 'draw', or 'away_win'")

    masked_probs = prob_matrix.copy()
    masked_probs[~mask] = 0

    for _ in range(min(max_scores, np.sum(mask))):
        if np.max(masked_probs) == 0:
            break

        idx = np.unravel_index(np.argmax(masked_probs), prob_matrix.shape)
        home_goals, away_goals = idx
        probability = masked_probs[idx]

        scores.append((f"{home_goals}-{away_goals}", probability))
        masked_probs[idx] = 0

    return scores

def match_outcome_prob(home_df, away_df, home_team, away_team, league_avg_xG):
    """Calculate match outcome probabilities for a single match"""
    try:
        home_xG = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxG/90'].values[0]
        home_xGA = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxGA/90'].values[0]
        away_xG = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxG/90'].values[0]
        away_xGA = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxGA/90'].values[0]

        lambda_A = expected_goals(home_xG, away_xGA, league_avg_xG)
        lambda_B = expected_goals(away_xG, home_xGA, league_avg_xG)

        # Calculate outcome probabilities
        prob_matrix = poisson_prob_matrix(lambda_A, lambda_B)
        home_win_prob = np.sum(np.tril(prob_matrix, -1))
        draw_prob = np.sum(np.diag(prob_matrix))
        away_win_prob = np.sum(np.triu(prob_matrix, 1))

        home_win_prob_adj, draw_prob_adj, away_win_prob_adj = adjust_draw_probability(
            home_win_prob, draw_prob, away_win_prob
        )

        # Determine predicted outcome
        if home_win_prob_adj > draw_prob_adj and home_win_prob_adj > away_win_prob_adj:
            predicted_outcome = f"{home_team} Win"
            outcome_type = 'home_win'
        elif away_win_prob_adj > home_win_prob_adj and away_win_prob_adj > draw_prob_adj:
            predicted_outcome = f"{away_team} Win"
            outcome_type = 'away_win'
        else:
            predicted_outcome = "Draw"
            outcome_type = 'draw'

        # Get most likely score
        most_likely_scores = find_most_likely_scores(prob_matrix, outcome_type)
        predicted_score = most_likely_scores[0][0] if most_likely_scores else "N/A"

        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_outcome': predicted_outcome,
            'predicted_score': predicted_score,
            'home_win_prob': round(home_win_prob_adj, 3),
            'draw_prob': round(draw_prob_adj, 3),
            'away_win_prob': round(away_win_prob_adj, 3),
            'home_xG': round(lambda_A, 2),
            'away_xG': round(lambda_B, 2)
        }
    except Exception as e:
        print(f"Error predicting {home_team} vs {away_team}: {e}")
        return None

# ==============================
# DATA PROCESSING
# ==============================

def get_current_standings(standings_df):
    """Extract current standings for all teams"""
    teams_data = {}

    for _, row in standings_df.iterrows():
        team = row['Squad']
        teams_data[team] = {
            'points': int(row['Home_Pts'] + row['Away_Pts']),
            'mp': int(row['Home_MP'] + row['Away_MP']),
            'w': int(row['Home_W'] + row['Away_W']),
            'd': int(row['Home_D'] + row['Away_D']),
            'l': int(row['Home_L'] + row['Away_L']),
            'gf': int(row['Home_GF'] + row['Away_GF']),
            'ga': int(row['Home_GA'] + row['Away_GA']),
            'gd': int(row['Home_GF'] + row['Away_GF'] - row['Home_GA'] - row['Away_GA'])
        }

    return teams_data

def predict_remaining_fixtures(schedule_df, home_df, away_df, league_avg_xG):
    """Predict all remaining (unplayed) fixtures"""
    future_fixtures = schedule_df[~schedule_df['Played']].copy()
    predictions = []

    print(f"Predicting {len(future_fixtures)} remaining fixtures...")

    for idx, match in future_fixtures.iterrows():
        home_team = match['Home']
        away_team = match['Away']

        prediction = match_outcome_prob(home_df, away_df, home_team, away_team, league_avg_xG)

        if prediction:
            predictions.append({
                'date': match['Date'],
                'home_team': home_team,
                'away_team': away_team,
                'predicted_score': prediction['predicted_score'],
                'home_xG': prediction['home_xG'],
                'away_xG': prediction['away_xG'],
                'home_win_prob': prediction['home_win_prob'],
                'draw_prob': prediction['draw_prob'],
                'away_win_prob': prediction['away_win_prob'],
                'home_xPTS': round(prediction['home_win_prob'] * 3 + prediction['draw_prob'] * 1, 2),
                'away_xPTS': round(prediction['away_win_prob'] * 3 + prediction['draw_prob'] * 1, 2)
            })

    return pd.DataFrame(predictions)

def calculate_team_projections(current_standings, predictions_df):
    """Calculate end-of-season projections for all teams"""
    projections = []

    for team in current_standings.keys():
        current = current_standings[team]

        # Get remaining fixtures for this team
        home_fixtures = predictions_df[predictions_df['home_team'] == team]
        away_fixtures = predictions_df[predictions_df['away_team'] == team]

        # Calculate remaining xPTS and xG
        remaining_xpts = home_fixtures['home_xPTS'].sum() + away_fixtures['away_xPTS'].sum()
        remaining_xgf = home_fixtures['home_xG'].sum() + away_fixtures['away_xG'].sum()
        remaining_xga = home_fixtures['away_xG'].sum() + away_fixtures['home_xG'].sum()

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

    return pd.DataFrame(projections).sort_values('projected_pts', ascending=False).reset_index(drop=True)

def get_team_matches(team, schedule_df, predictions_df, current_standings):
    """Get all matches (played and upcoming) for a specific team"""
    all_matches = []

    # Get played matches
    played = schedule_df[schedule_df['Played']].copy()
    team_played_home = played[played['Home'] == team]
    team_played_away = played[played['Away'] == team]

    for _, match in team_played_home.iterrows():
        # Parse score
        try:
            score_parts = str(match['Score']).split('–')
            if len(score_parts) != 2:
                score_parts = str(match['Score']).split('-')
            home_goals = int(score_parts[0])
            away_goals = int(score_parts[1])

            if home_goals > away_goals:
                result = 'W'
                pts = 3
            elif home_goals < away_goals:
                result = 'L'
                pts = 0
            else:
                result = 'D'
                pts = 1

            all_matches.append({
                'date': match['Date'],
                'opponent': match['Away'],
                'venue': 'Home',
                'played': True,
                'score': f"{home_goals}-{away_goals}",
                'result': result,
                'actual_pts': pts,
                'predicted_score': None,
                'xPTS': None,
                'xG': None,
                'xGA': None
            })
        except:
            pass

    for _, match in team_played_away.iterrows():
        try:
            score_parts = str(match['Score']).split('–')
            if len(score_parts) != 2:
                score_parts = str(match['Score']).split('-')
            home_goals = int(score_parts[0])
            away_goals = int(score_parts[1])

            if away_goals > home_goals:
                result = 'W'
                pts = 3
            elif away_goals < home_goals:
                result = 'L'
                pts = 0
            else:
                result = 'D'
                pts = 1

            all_matches.append({
                'date': match['Date'],
                'opponent': match['Home'],
                'venue': 'Away',
                'played': True,
                'score': f"{home_goals}-{away_goals}",
                'result': result,
                'actual_pts': pts,
                'predicted_score': None,
                'xPTS': None,
                'xG': None,
                'xGA': None
            })
        except:
            pass

    # Get upcoming matches
    upcoming_home = predictions_df[predictions_df['home_team'] == team]
    upcoming_away = predictions_df[predictions_df['away_team'] == team]

    for _, match in upcoming_home.iterrows():
        all_matches.append({
            'date': match['date'],
            'opponent': match['away_team'],
            'venue': 'Home',
            'played': False,
            'score': None,
            'result': None,
            'actual_pts': None,
            'predicted_score': match['predicted_score'],
            'xPTS': match['home_xPTS'],
            'xG': match['home_xG'],
            'xGA': match['away_xG']
        })

    for _, match in upcoming_away.iterrows():
        score_reversed = f"{match['predicted_score'].split('-')[1]}-{match['predicted_score'].split('-')[0]}"
        all_matches.append({
            'date': match['date'],
            'opponent': match['home_team'],
            'venue': 'Away',
            'played': False,
            'score': None,
            'result': None,
            'actual_pts': None,
            'predicted_score': score_reversed,
            'xPTS': match['away_xPTS'],
            'xG': match['away_xG'],
            'xGA': match['home_xG']
        })

    return pd.DataFrame(all_matches).sort_values('date').reset_index(drop=True)

# ==============================
# INITIALIZE DATA
# ==============================

print("="*70)
print("Premier League xPTS Tracker - Initializing with REAL DATA")
print("="*70)

try:
    print("Fetching team statistics from fbref.com...")
    home_df, away_df, standings_df = fetch_team_stats()

    print("Calculating league averages...")
    league_avg_xG_home = home_df['Normalized wxG/90'].mean()
    league_avg_xG_away = away_df['Normalized wxG/90'].mean()
    league_avg_xG = (league_avg_xG_home + league_avg_xG_away) / 2

    print("Fetching Premier League schedule...")
    schedule_df = fetch_pl_schedule()

    print("Extracting current standings...")
    current_standings = get_current_standings(standings_df)

    print("Predicting remaining fixtures...")
    predictions_df = predict_remaining_fixtures(schedule_df, home_df, away_df, league_avg_xG)

    print("Calculating team projections...")
    projections_df = calculate_team_projections(current_standings, predictions_df)

    PREMIER_LEAGUE_TEAMS = list(current_standings.keys())

    print(f"✓ Loaded data for {len(PREMIER_LEAGUE_TEAMS)} teams")
    print(f"✓ Predicted {len(predictions_df)} remaining fixtures")
    print(f"✓ Projected winner: {projections_df.iloc[0]['team']} ({projections_df.iloc[0]['projected_pts']:.1f} pts)")

    data_loaded = True

except Exception as e:
    print(f"ERROR loading real data: {e}")
    import traceback
    traceback.print_exc()
    print("\nFalling back to empty data...")
    data_loaded = False
    PREMIER_LEAGUE_TEAMS = []
    projections_df = pd.DataFrame()
    current_standings = {}

# ==============================
# VISUALIZATION FUNCTIONS
# ==============================

def create_league_table(projections_df):
    """Create the projected league table"""
    if projections_df.empty:
        return pd.DataFrame([{'Message': 'No data available'}])

    table_df = projections_df.copy()
    table_df['position'] = range(1, len(table_df) + 1)

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
    if not data_loaded:
        return dbc.Container([
            html.H1("⚽ Premier League xPTS Tracker", className="text-center mb-3 mt-3"),
            dbc.Alert("Failed to load data from fbref.com. Please try again later.", color="danger")
        ], fluid=True)

    league_table_df = create_league_table(projections_df)

    return dbc.Container([
        html.H1("⚽ Premier League xPTS Tracker", className="text-center mb-3 mt-3"),
        html.H4("2024/25 Season Projections (REAL DATA)", className="text-center text-success mb-4"),

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
                                {'if': {'filter_query': '{Pos} <= 4'}, 'backgroundColor': '#1e3a5f', 'fontWeight': 'bold'},
                                {'if': {'filter_query': '{Pos} = 5'}, 'backgroundColor': '#2d4a2f'},
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
            html.P("🔴 Bottom 3: Relegation", className="text-danger"),
            html.P(f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", className="text-muted small")
        ], className="text-center")

    ], fluid=True)

def create_team_layout(team):
    """Create individual team page"""
    if not data_loaded:
        return dbc.Container([
            dbc.Alert("Data not loaded", color="danger")
        ], fluid=True)

    team_matches = get_team_matches(team, schedule_df, predictions_df, current_standings)
    team_proj = projections_df[projections_df['team'] == team].iloc[0]
    current = current_standings[team]

    played_matches = team_matches[team_matches['played']]
    upcoming_matches = team_matches[~team_matches['played']]

    return dbc.Container([
        html.H2(f"{team}", className="text-center mb-3 mt-3"),

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

# Main layout
if data_loaded:
    app.layout = dbc.Container([
        dcc.Tabs(id='main-tabs', value='home', children=[
            dcc.Tab(label='🏠 Home', value='home'),
            *[dcc.Tab(label=team, value=team) for team in PREMIER_LEAGUE_TEAMS]
        ]),
        html.Div(id='page-content')
    ], fluid=True, style={'padding': '0'})
else:
    app.layout = create_home_layout()

# ==============================
# CALLBACKS
# ==============================

if data_loaded:
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
    print("Starting Premier League xPTS Tracker on port 8900")
    print("="*70 + "\n")
    app.run(debug=True, port=8900)

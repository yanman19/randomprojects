import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import pandas as pd
import numpy as np
import scipy.stats as stats
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from io import StringIO
import time

# ==============================
# REALISTIC MOCK DATA (2025/26 Season as of Oct 2025)
# ==============================

MOCK_CURRENT_STANDINGS = {
    'Liverpool': {'points': 47, 'mp': 20, 'w': 14, 'd': 5, 'l': 1, 'gf': 47, 'ga': 19},
    'Arsenal': {'points': 43, 'mp': 20, 'w': 12, 'd': 7, 'l': 1, 'gf': 42, 'ga': 18},
    'Chelsea': {'points': 40, 'mp': 20, 'w': 12, 'd': 4, 'l': 4, 'gf': 43, 'ga': 24},
    'Nottingham Forest': {'points': 37, 'mp': 20, 'w': 11, 'd': 4, 'l': 5, 'gf': 29, 'ga': 23},
    'Newcastle': {'points': 35, 'mp': 20, 'w': 10, 'd': 5, 'l': 5, 'gf': 34, 'ga': 22},
    'Man City': {'points': 35, 'mp': 20, 'w': 10, 'd': 5, 'l': 5, 'gf': 39, 'ga': 29},
    'Bournemouth': {'points': 33, 'mp': 20, 'w': 10, 'd': 3, 'l': 7, 'gf': 32, 'ga': 27},
    'Aston Villa': {'points': 32, 'mp': 20, 'w': 9, 'd': 5, 'l': 6, 'gf': 30, 'ga': 30},
    'Brighton': {'points': 31, 'mp': 20, 'w': 8, 'd': 7, 'l': 5, 'gf': 32, 'ga': 29},
    'Fulham': {'points': 30, 'mp': 20, 'w': 8, 'd': 6, 'l': 6, 'gf': 29, 'ga': 26},
    'Brentford': {'points': 28, 'mp': 20, 'w': 8, 'd': 4, 'l': 8, 'gf': 35, 'ga': 32},
    'Tottenham': {'points': 27, 'mp': 20, 'w': 8, 'd': 3, 'l': 9, 'gf': 40, 'ga': 28},
    'West Ham': {'points': 25, 'mp': 20, 'w': 7, 'd': 4, 'l': 9, 'gf': 24, 'ga': 35},
    'Man United': {'points': 23, 'mp': 20, 'w': 6, 'd': 5, 'l': 9, 'gf': 23, 'ga': 28},
    'Crystal Palace': {'points': 23, 'mp': 20, 'w': 6, 'd': 5, 'l': 9, 'gf': 21, 'ga': 28},
    'Everton': {'points': 20, 'mp': 20, 'w': 5, 'd': 5, 'l': 10, 'gf': 18, 'ga': 29},
    'Wolves': {'points': 16, 'mp': 20, 'w': 4, 'd': 4, 'l': 12, 'gf': 29, 'ga': 42},
    'Ipswich Town': {'points': 16, 'mp': 20, 'w': 4, 'd': 4, 'l': 12, 'gf': 21, 'ga': 37},
    'Leicester City': {'points': 14, 'mp': 20, 'w': 3, 'd': 5, 'l': 12, 'gf': 23, 'ga': 44},
    'Southampton': {'points': 6, 'mp': 20, 'w': 1, 'd': 3, 'l': 16, 'gf': 14, 'ga': 45}
}

def create_mock_team_stats():
    """Create mock home/away DataFrames from current standings"""
    home_data = []
    away_data = []

    for team, stats in MOCK_CURRENT_STANDINGS.items():
        # Approximate home/away split (roughly 60/40 for home advantage)
        home_mp = stats['mp'] // 2
        away_mp = stats['mp'] - home_mp

        home_pts = int(stats['points'] * 0.55)
        away_pts = stats['points'] - home_pts

        home_gf = int(stats['gf'] * 0.55)
        away_gf = stats['gf'] - home_gf

        home_ga = int(stats['ga'] * 0.45)
        away_ga = stats['ga'] - home_ga

        # Calculate xG (approximate based on goals with some variation)
        home_xg = home_gf * 0.95 + np.random.uniform(-2, 2)
        home_xga = home_ga * 0.95 + np.random.uniform(-2, 2)
        away_xg = away_gf * 0.95 + np.random.uniform(-2, 2)
        away_xga = away_ga * 0.95 + np.random.uniform(-2, 2)

        home_data.append({
            'Squad': team,
            'MP': home_mp,
            'GF': home_gf,
            'GA': home_ga,
            'Pts': home_pts,
            'xG': round(home_xg, 1),
            'xGA': round(home_xga, 1),
            'wxG': round(home_xg * 0.7 + home_gf * 0.3, 2),
            'wxGA': round(home_xga * 0.7 + home_ga * 0.3, 2),
            'Normalized wxG/90': round((home_xg * 0.7 + home_gf * 0.3) / home_mp, 2) if home_mp > 0 else 0,
            'Normalized wxGA/90': round((home_xga * 0.7 + home_ga * 0.3) / home_mp, 2) if home_mp > 0 else 0
        })

        away_data.append({
            'Squad': team,
            'MP': away_mp,
            'GF': away_gf,
            'GA': away_ga,
            'Pts': away_pts,
            'xG': round(away_xg, 1),
            'xGA': round(away_xga, 1),
            'wxG': round(away_xg * 0.7 + away_gf * 0.3, 2),
            'wxGA': round(away_xga * 0.7 + away_ga * 0.3, 2),
            'Normalized wxG/90': round((away_xg * 0.7 + away_gf * 0.3) / away_mp, 2) if away_mp > 0 else 0,
            'Normalized wxGA/90': round((away_xga * 0.7 + away_ga * 0.3) / away_mp, 2) if away_mp > 0 else 0
        })

    return pd.DataFrame(home_data), pd.DataFrame(away_data)

# ==============================
# DATA FETCHING FROM UNDERSTAT.COM
# ==============================

def extract_teams_data_from_html(html: str) -> dict:
    """Extract teamsData JSON from understat HTML"""
    import re
    import json

    m = re.search(r"var\s+teamsData\s*=\s*JSON\.parse\('([^']+)'\);", html)
    if not m:
        raise RuntimeError("Understat teamsData not found in HTML.")
    raw = m.group(1)
    raw = raw.encode("utf-8").decode("unicode_escape").replace("\\'", "'")
    return json.loads(raw)

def http_get_understat(url, retries=3, timeout=15):
    """HTTP GET with retries for understat.com"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(1 + i)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")

def fetch_fresh_data(season=2025):
    """Fetch fresh data from understat.com"""
    print(f"Attempting to fetch data from understat.com for season {season}...")

    html = http_get_understat(f"https://understat.com/league/EPL/{season}")
    teams_data = extract_teams_data_from_html(html)

    rows_home, rows_away = [], []
    standings = {}

    for _, team in teams_data.items():
        team_name = team.get("title") or team.get("team_title")

        # Initialize home/away stats
        h = {"MP": 0, "xG": 0.0, "xGA": 0.0, "G": 0, "GA": 0, "Pts": 0, "W": 0, "D": 0, "L": 0}
        a = {"MP": 0, "xG": 0.0, "xGA": 0.0, "G": 0, "GA": 0, "Pts": 0, "W": 0, "D": 0, "L": 0}

        # Process match history
        for game in team.get("history", []):
            side = game.get("h_a")
            xg = float(game.get("xG", 0.0))
            xga = float(game.get("xGA", 0.0))
            goals_scored = int(game.get("scored", game.get("goals", 0)))
            goals_against = int(game.get("missed", game.get("conceded", 0)))

            # Determine result
            if goals_scored > goals_against:
                pts, result = 3, "W"
            elif goals_scored < goals_against:
                pts, result = 0, "L"
            else:
                pts, result = 1, "D"

            if side == "h":
                h["MP"] += 1
                h["xG"] += xg
                h["xGA"] += xga
                h["G"] += goals_scored
                h["GA"] += goals_against
                h["Pts"] += pts
                h[result] += 1
            elif side == "a":
                a["MP"] += 1
                a["xG"] += xg
                a["xGA"] += xga
                a["G"] += goals_scored
                a["GA"] += goals_against
                a["Pts"] += pts
                a[result] += 1

        # Calculate weighted xG (70% xG, 30% actual goals)
        wxG_home = 0.7 * h["xG"] + 0.3 * h["G"]
        wxGA_home = 0.7 * h["xGA"] + 0.3 * h["GA"]
        wxG_away = 0.7 * a["xG"] + 0.3 * a["G"]
        wxGA_away = 0.7 * a["xGA"] + 0.3 * a["GA"]

        # Add to DataFrames
        rows_home.append({
            "Squad": team_name,
            "MP": h["MP"],
            "GF": h["G"],
            "GA": h["GA"],
            "Pts": h["Pts"],
            "xG": h["xG"],
            "xGA": h["xGA"],
            "wxG": wxG_home,
            "wxGA": wxGA_home,
            "Normalized wxG/90": (wxG_home / h["MP"]) if h["MP"] > 0 else 0.0,
            "Normalized wxGA/90": (wxGA_home / h["MP"]) if h["MP"] > 0 else 0.0
        })

        rows_away.append({
            "Squad": team_name,
            "MP": a["MP"],
            "GF": a["G"],
            "GA": a["GA"],
            "Pts": a["Pts"],
            "xG": a["xG"],
            "xGA": a["xGA"],
            "wxG": wxG_away,
            "wxGA": wxGA_away,
            "Normalized wxG/90": (wxG_away / a["MP"]) if a["MP"] > 0 else 0.0,
            "Normalized wxGA/90": (wxGA_away / a["MP"]) if a["MP"] > 0 else 0.0
        })

        # Calculate total standings
        standings[team_name] = {
            'points': h["Pts"] + a["Pts"],
            'mp': h["MP"] + a["MP"],
            'w': h["W"] + a["W"],
            'd': h["D"] + a["D"],
            'l': h["L"] + a["L"],
            'gf': h["G"] + a["G"],
            'ga': h["GA"] + a["GA"]
        }

    home_df = pd.DataFrame(rows_home).sort_values("Squad").reset_index(drop=True)
    away_df = pd.DataFrame(rows_away).sort_values("Squad").reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"✓ Successfully fetched data from understat.com at {timestamp}")
    print(f"✓ Found {len(home_df)} teams")

    return home_df, away_df, standings, timestamp

# ==============================
# PREDICTION ENGINE (from prem_dash_v4.py)
# ==============================

def expected_goals(xG_team, xGA_opp, league_avg_xG):
    return (xG_team / league_avg_xG) * (xGA_opp / league_avg_xG) * league_avg_xG

def poisson_prob_matrix(lambda_A, lambda_B, max_goals=10):
    prob_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            prob_matrix[i, j] = stats.poisson.pmf(i, lambda_A) * stats.poisson.pmf(j, lambda_B)
    return prob_matrix

def adjust_draw_probability(home_win_prob, draw_prob, away_win_prob):
    total_prob = home_win_prob + draw_prob + away_win_prob
    if home_win_prob > away_win_prob:
        favored_win_prob = home_win_prob
        underdog_win_prob = away_win_prob
    else:
        favored_win_prob = away_win_prob
        underdog_win_prob = home_win_prob
    adjusted_draw_prob = max(draw_prob, min(favored_win_prob * 0.75, 0.35))
    normalization_factor = total_prob / (favored_win_prob + adjusted_draw_prob + underdog_win_prob)
    return (home_win_prob * normalization_factor, adjusted_draw_prob * normalization_factor, away_win_prob * normalization_factor)

def find_most_likely_scores(prob_matrix, outcome_type, max_scores=3):
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
    try:
        home_xG = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxG/90'].values[0]
        home_xGA = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxGA/90'].values[0]
        away_xG = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxG/90'].values[0]
        away_xGA = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxGA/90'].values[0]

        lambda_A = expected_goals(home_xG, away_xGA, league_avg_xG)
        lambda_B = expected_goals(away_xG, home_xGA, league_avg_xG)

        prob_matrix = poisson_prob_matrix(lambda_A, lambda_B)
        home_win_prob = np.sum(np.tril(prob_matrix, -1))
        draw_prob = np.sum(np.diag(prob_matrix))
        away_win_prob = np.sum(np.triu(prob_matrix, 1))

        home_win_prob_adj, draw_prob_adj, away_win_prob_adj = adjust_draw_probability(home_win_prob, draw_prob, away_win_prob)

        if home_win_prob_adj > draw_prob_adj and home_win_prob_adj > away_win_prob_adj:
            predicted_outcome = f"{home_team} Win"
            outcome_type = 'home_win'
        elif away_win_prob_adj > home_win_prob_adj and away_win_prob_adj > draw_prob_adj:
            predicted_outcome = f"{away_team} Win"
            outcome_type = 'away_win'
        else:
            predicted_outcome = "Draw"
            outcome_type = 'draw'

        most_likely_scores = find_most_likely_scores(prob_matrix, outcome_type)
        predicted_score = most_likely_scores[0][0] if most_likely_scores else "N/A"

        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_score': predicted_score,
            'home_xG': round(lambda_A, 2),
            'away_xG': round(lambda_B, 2),
            'home_win_prob': round(home_win_prob_adj, 3),
            'draw_prob': round(draw_prob_adj, 3),
            'away_win_prob': round(away_win_prob_adj, 3)
        }
    except Exception as e:
        print(f"Error predicting {home_team} vs {away_team}: {e}")
        return None

# ==============================
# GENERATE PREDICTIONS
# ==============================

def generate_remaining_fixtures(home_df, away_df, league_avg_xG, teams_list=None):
    """Generate predictions for remaining fixtures"""
    if teams_list is None:
        teams = sorted(MOCK_CURRENT_STANDINGS.keys())
    else:
        teams = sorted(teams_list)
    predictions = []

    # Generate round-robin fixtures for remaining 18 matchdays
    matchday_start = 21
    base_date = datetime(2025, 10, 25)

    for md in range(18):
        match_date = base_date + timedelta(days=md * 7)
        teams_copy = teams.copy()
        np.random.seed(42 + md)
        np.random.shuffle(teams_copy)

        for i in range(0, len(teams_copy) - 1, 2):
            home_team = teams_copy[i]
            away_team = teams_copy[i + 1]

            pred = match_outcome_prob(home_df, away_df, home_team, away_team, league_avg_xG)
            if pred:
                predictions.append({
                    'date': match_date,
                    'matchday': matchday_start + md,
                    'home_team': home_team,
                    'away_team': away_team,
                    'predicted_score': pred['predicted_score'],
                    'home_xG': pred['home_xG'],
                    'away_xG': pred['away_xG'],
                    'home_win_prob': pred['home_win_prob'],
                    'draw_prob': pred['draw_prob'],
                    'away_win_prob': pred['away_win_prob'],
                    'home_xPTS': round(pred['home_win_prob'] * 3 + pred['draw_prob'] * 1, 2),
                    'away_xPTS': round(pred['away_win_prob'] * 3 + pred['draw_prob'] * 1, 2)
                })

    return pd.DataFrame(predictions)

def calculate_projections(current_standings, predictions_df):
    """Calculate end-of-season projections"""
    projections = []

    for team in current_standings.keys():
        current = current_standings[team]
        home_fixtures = predictions_df[predictions_df['home_team'] == team]
        away_fixtures = predictions_df[predictions_df['away_team'] == team]

        remaining_xpts = home_fixtures['home_xPTS'].sum() + away_fixtures['away_xPTS'].sum()
        remaining_xgf = home_fixtures['home_xG'].sum() + away_fixtures['away_xG'].sum()
        remaining_xga = home_fixtures['away_xG'].sum() + away_fixtures['home_xG'].sum()

        projections.append({
            'team': team,
            'current_pts': current['points'],
            'current_mp': current['mp'],
            'current_gf': current['gf'],
            'current_ga': current['ga'],
            'current_gd': current['gf'] - current['ga'],
            'remaining_xpts': round(remaining_xpts, 1),
            'projected_pts': round(current['points'] + remaining_xpts, 1),
            'projected_gf': round(current['gf'] + remaining_xgf, 1),
            'projected_ga': round(current['ga'] + remaining_xga, 1),
            'projected_gd': round((current['gf'] - current['ga']) + remaining_xgf - remaining_xga, 1)
        })

    return pd.DataFrame(projections).sort_values('projected_pts', ascending=False).reset_index(drop=True)

# ==============================
# INITIALIZE DATA
# ==============================

print("Initializing Premier League xPTS Tracker...")
home_df, away_df = create_mock_team_stats()
current_standings = MOCK_CURRENT_STANDINGS.copy()

league_avg_xG_home = home_df['Normalized wxG/90'].mean()
league_avg_xG_away = away_df['Normalized wxG/90'].mean()
league_avg_xG = (league_avg_xG_home + league_avg_xG_away) / 2

predictions_df = generate_remaining_fixtures(home_df, away_df, league_avg_xG)
projections_df = calculate_projections(current_standings, predictions_df)

PREMIER_LEAGUE_TEAMS = sorted(current_standings.keys())
last_updated = "Mock Data (2025/26 Season)"

print(f"✓ Loaded {len(PREMIER_LEAGUE_TEAMS)} teams")
print(f"✓ Generated {len(predictions_df)} fixture predictions")
print(f"✓ Projected winner: {projections_df.iloc[0]['team']} ({projections_df.iloc[0]['projected_pts']:.1f} pts)")

# ==============================
# VISUALIZATION FUNCTIONS
# ==============================

def create_league_table(projections_df):
    table_df = projections_df.copy()
    table_df['position'] = range(1, len(table_df) + 1)
    display_df = table_df[['position', 'team', 'current_pts', 'remaining_xpts', 'projected_pts', 'current_gd', 'projected_gd']].copy()
    display_df.columns = ['Pos', 'Team', 'Current Pts', 'Remaining xPTS', 'Projected Pts', 'Current GD', 'Projected GD']
    return display_df

def create_team_progression_graph(team, projections_df, predictions_df, current_standings):
    current = current_standings[team]
    home_fixtures = predictions_df[predictions_df['home_team'] == team].sort_values('date')
    away_fixtures = predictions_df[predictions_df['away_team'] == team].sort_values('date')

    # Create progression
    dates = [datetime(2025, 8, 16)]
    points = [0]

    # Add current progress
    dates.append(datetime.now())
    points.append(current['points'])

    # Add future predictions
    cumulative = current['points']
    for _, match in home_fixtures.iterrows():
        dates.append(match['date'])
        cumulative += match['home_xPTS']
        points.append(cumulative)

    for _, match in away_fixtures.iterrows():
        if match['date'] not in dates:
            dates.append(match['date'])
            cumulative += match['away_xPTS']
            points.append(cumulative)

    # Sort by date
    sorted_data = sorted(zip(dates, points))
    dates, points = zip(*sorted_data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=points, mode='lines+markers', name='Projected Points', line=dict(color='#00ff00', width=2)))
    fig.add_shape(type="line", x0=datetime.now(), x1=datetime.now(), y0=0, y1=max(points), line=dict(color="yellow", width=2, dash="dot"))
    fig.update_layout(title=f'{team} - Season Progression', xaxis_title='Date', yaxis_title='Points', template='plotly_dark', height=400)
    return fig

def get_team_upcoming_matches(team, predictions_df):
    home = predictions_df[predictions_df['home_team'] == team][['date', 'away_team', 'predicted_score', 'home_xPTS', 'home_xG', 'away_xG']].copy()
    away = predictions_df[predictions_df['away_team'] == team][['date', 'home_team', 'predicted_score', 'away_xPTS', 'away_xG', 'home_xG']].copy()

    home['opponent'] = home['away_team']
    home['venue'] = 'Home'
    home['xPTS'] = home['home_xPTS']
    home['xG'] = home['home_xG']
    home['xGA'] = home['away_xG']

    away['opponent'] = away['home_team']
    away['venue'] = 'Away'
    away['xPTS'] = away['away_xPTS']
    away['xG'] = away['away_xG']
    away['xGA'] = away['home_xG']
    away['predicted_score'] = away['predicted_score'].apply(lambda x: f"{x.split('-')[1]}-{x.split('-')[0]}")

    all_matches = pd.concat([
        home[['date', 'opponent', 'venue', 'predicted_score', 'xPTS', 'xG', 'xGA']],
        away[['date', 'opponent', 'venue', 'predicted_score', 'xPTS', 'xG', 'xGA']]
    ]).sort_values('date')

    all_matches['Date'] = all_matches['date'].dt.strftime('%Y-%m-%d')
    return all_matches[['Date', 'opponent', 'venue', 'predicted_score', 'xPTS', 'xG', 'xGA']].rename(columns={'opponent': 'Opponent', 'venue': 'Venue', 'predicted_score': 'Predicted Score'})

# ==============================
# DASH APP
# ==============================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

def create_home_layout():
    league_table_df = create_league_table(projections_df)

    return dbc.Container([
        html.H1("⚽ Premier League xPTS Tracker", className="text-center mb-2 mt-3"),
        html.H5("2025/26 Season Projections", className="text-center text-muted mb-3"),
        html.P(id='last-updated-display', children=f"Data: {last_updated}", className="text-center small text-warning mb-2"),
        dbc.Button("🔄 Refresh Data from understat.com", id="refresh-btn", color="success", className="mb-3", style={'display': 'block', 'margin': '0 auto'}),
        html.Div(id='refresh-status', className="text-center mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Projected Final Standings", className="text-center")),
                    dbc.CardBody([
                        DataTable(
                            id='league-table',
                            data=league_table_df.to_dict('records'),
                            columns=[{'name': col, 'id': col} for col in league_table_df.columns],
                            style_cell={'textAlign': 'center', 'backgroundColor': '#2c3e50', 'color': 'white', 'padding': '12px'},
                            style_header={'backgroundColor': '#1a252f', 'fontWeight': 'bold'},
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
            html.P("🔴 Bottom 3: Relegation", className="text-danger")
        ], className="text-center small")
    ], fluid=True)

def create_team_layout(team):
    team_proj = projections_df[projections_df['team'] == team].iloc[0]
    current = current_standings[team]
    upcoming = get_team_upcoming_matches(team, predictions_df)

    return dbc.Container([
        html.H2(f"{team}", className="text-center mb-3 mt-3"),

        dbc.Row([
            dbc.Col([dbc.Card([dbc.CardBody([html.H5("Current Points"), html.H2(f"{current['points']}", className="text-success"), html.P(f"After {current['mp']} matches", className="small text-muted")])], color="dark")], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([html.H5("Projected Points"), html.H2(f"{team_proj['projected_pts']:.1f}", className="text-warning"), html.P(f"+{team_proj['remaining_xpts']:.1f} xPTS", className="small text-muted")])], color="dark")], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([html.H5("Current GD"), html.H2(f"{current['gf'] - current['ga']:+d}", className="text-info"), html.P(f"{current['gf']}-{current['ga']}", className="small text-muted")])], color="dark")], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([html.H5("Projected GD"), html.H2(f"{team_proj['projected_gd']:+.1f}", className="text-primary"), html.P(f"{team_proj['projected_gf']:.0f}-{team_proj['projected_ga']:.0f}", className="small text-muted")])], color="dark")], width=3)
        ], className="mb-4"),

        dbc.Row([dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=create_team_progression_graph(team, projections_df, predictions_df, current_standings))])], color="dark")])], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H4("Remaining Fixtures", className="text-center mb-3"),
                DataTable(
                    data=upcoming.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in upcoming.columns],
                    style_cell={'textAlign': 'center', 'backgroundColor': '#2c3e50', 'color': 'white', 'padding': '10px'},
                    style_header={'backgroundColor': '#1a252f', 'fontWeight': 'bold'},
                    page_size=20
                )
            ])
        ])
    ], fluid=True)

# Store data
app.layout = dbc.Container([
    dcc.Store(id='home-data-store', data=home_df.to_dict('records')),
    dcc.Store(id='away-data-store', data=away_df.to_dict('records')),
    dcc.Store(id='standings-store', data=current_standings),
    dcc.Store(id='predictions-store', data=predictions_df.to_dict('records')),
    dcc.Store(id='projections-store', data=projections_df.to_dict('records')),
    dcc.Store(id='teams-list-store', data=PREMIER_LEAGUE_TEAMS),

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
    [Output('home-data-store', 'data'),
     Output('away-data-store', 'data'),
     Output('standings-store', 'data'),
     Output('predictions-store', 'data'),
     Output('projections-store', 'data'),
     Output('teams-list-store', 'data'),
     Output('refresh-status', 'children'),
     Output('last-updated-display', 'children')],
    [Input('refresh-btn', 'n_clicks')],
    prevent_initial_call=True
)
def refresh_data(n_clicks):
    if n_clicks:
        try:
            new_home, new_away, new_standings, timestamp = fetch_fresh_data()

            # Recalculate predictions using the team names from the fetched data
            new_league_avg = (new_home['Normalized wxG/90'].mean() + new_away['Normalized wxG/90'].mean()) / 2
            teams_list = list(new_standings.keys())
            new_predictions = generate_remaining_fixtures(new_home, new_away, new_league_avg, teams_list=teams_list)
            new_projections = calculate_projections(new_standings, new_predictions)

            return (
                new_home.to_dict('records'),
                new_away.to_dict('records'),
                new_standings,
                new_predictions.to_dict('records'),
                new_projections.to_dict('records'),
                teams_list,
                dbc.Alert("✓ Successfully refreshed data from understat.com!", color="success", duration=4000),
                f"Last updated: {timestamp}"
            )
        except Exception as e:
            error_msg = str(e)
            if '403' in error_msg or 'Forbidden' in error_msg:
                msg = "⚠️ understat.com blocked the request. Using existing data."
            else:
                msg = f"⚠️ Error: {error_msg}. Using existing data."

            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dbc.Alert(msg, color="warning", duration=6000), dash.no_update

    raise dash.exceptions.PreventUpdate

@app.callback(
    Output('page-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('projections-store', 'data'),
     Input('predictions-store', 'data'),
     Input('standings-store', 'data')]
)
def render_content(tab, proj_data, pred_data, standings_data):
    global projections_df, predictions_df, current_standings

    # Update globals when data changes
    if proj_data:
        projections_df = pd.DataFrame(proj_data)
    if pred_data:
        predictions_df = pd.DataFrame(pred_data)
    if standings_data:
        current_standings = standings_data

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
    print("Access at: http://localhost:8900 (or your forwarded URL)")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=8900)

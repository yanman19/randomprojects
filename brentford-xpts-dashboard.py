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
from plotly.subplots import make_subplots

# ==============================
# DATA FETCHING AND PREPARATION
# ==============================

def fetch_team_stats():
    """Fetch team performance data from fbref.com for Premier League"""
    import requests
    import pandas as pd
    import numpy as np
    import time
    from io import StringIO

    # Premier League - add delay to avoid rate limiting
    time.sleep(3)
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    dfs = pd.read_html(StringIO(response.text))

    df = dfs[1]
    df.columns = [
        "Rk","Squad",
        "Home_MP", "Home_W", "Home_D", "Home_L", "Home_GF", "Home_GA", "Home_GD", "Home_Pts",'Home_Pts/MP', "Home_xG", "Home_xGA", "Home_xGD", "Home_xGD_per_90",
        "Away_MP", "Away_W", "Away_D", "Away_L", "Away_GF", "Away_GA", "Away_GD", "Away_Pts",'Away_Pts/MP', "Away_xG", "Away_xGA", "Away_xGD", "Away_xGD_per_90"]

    df['Country'] = 'England'

    home_df = df[[
        "Rk", "Squad",
        "Home_MP", "Home_W", "Home_D", "Home_L", "Home_GF", "Home_GA",
        "Home_GD", "Home_Pts", "Home_xG", "Home_xGA", "Home_xGD", "Home_xGD_per_90",'Country'
    ]].copy()

    away_df = df[[
        "Rk", "Squad",
        "Away_MP", "Away_W", "Away_D", "Away_L", "Away_GF", "Away_GA",
        "Away_GD", "Away_Pts", "Away_xG", "Away_xGA", "Away_xGD", "Away_xGD_per_90",'Country'
    ]].copy()

    rename_home = {
        "Home_MP": "MP",
        "Home_W": "W",
        "Home_D": "D",
        "Home_L": "L",
        "Home_GF": "GF",
        "Home_GA": "GA",
        "Home_GD": "GD",
        "Home_Pts": "Pts",
        "Home_xG": "xG",
        "Home_xGA": "xGA",
        "Home_xGD": "xGD",
        "Home_xGD_per_90": "xGD_per_90"
    }

    rename_away = {
        "Away_MP": "MP",
        "Away_W": "W",
        "Away_D": "D",
        "Away_L": "L",
        "Away_GF": "GF",
        "Away_GA": "GA",
        "Away_GD": "GD",
        "Away_Pts": "Pts",
        "Away_xG": "xG",
        "Away_xGA": "xGA",
        "Away_xGD": "xGD",
        "Away_xGD_per_90": "xGD_per_90"
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

def fetch_brentford_schedule():
    """Fetch Brentford's full Premier League schedule"""
    try:
        import time
        from io import StringIO

        time.sleep(3)  # Rate limiting
        url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        matches_df = pd.read_html(StringIO(response.text))[0]

        # Filter for Brentford matches
        brentford_matches = matches_df[
            (matches_df['Home'] == 'Brentford') |
            (matches_df['Away'] == 'Brentford')
        ].copy()

        # Convert date
        brentford_matches['Date'] = pd.to_datetime(brentford_matches['Date'])

        # Determine if match has been played
        brentford_matches['Played'] = ~brentford_matches['Score'].isna()

        # Parse scores for played matches
        for idx, row in brentford_matches.iterrows():
            if row['Played']:
                try:
                    # Try different delimiters
                    for delimiter in ['–', '-', ':', ' ']:
                        try:
                            score_parts = row['Score'].split(delimiter)
                            if len(score_parts) == 2:
                                brentford_matches.at[idx, 'HomeGoals'] = int(score_parts[0])
                                brentford_matches.at[idx, 'AwayGoals'] = int(score_parts[1])
                                break
                        except:
                            continue
                except:
                    brentford_matches.at[idx, 'HomeGoals'] = 0
                    brentford_matches.at[idx, 'AwayGoals'] = 0

        # Sort by date
        brentford_matches = brentford_matches.sort_values('Date').reset_index(drop=True)

        return brentford_matches
    except Exception as e:
        print(f"Error fetching Brentford schedule: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def get_brentford_current_points(standings_df):
    """Get Brentford's current points from the standings"""
    try:
        # Filter for Brentford
        brentford_row = standings_df[standings_df['Squad'] == 'Brentford']
        if not brentford_row.empty:
            home_pts = brentford_row['Home_Pts'].values[0]
            away_pts = brentford_row['Away_Pts'].values[0]
            total_pts = home_pts + away_pts

            home_mp = brentford_row['Home_MP'].values[0]
            away_mp = brentford_row['Away_MP'].values[0]
            total_mp = home_mp + away_mp

            home_gf = brentford_row['Home_GF'].values[0]
            away_gf = brentford_row['Away_GF'].values[0]
            total_gf = home_gf + away_gf

            home_ga = brentford_row['Home_GA'].values[0]
            away_ga = brentford_row['Away_GA'].values[0]
            total_ga = home_ga + away_ga

            return {
                'points': int(total_pts),
                'matches_played': int(total_mp),
                'goals_for': int(total_gf),
                'goals_against': int(total_ga),
                'goal_difference': int(total_gf - total_ga)
            }
        else:
            print("Brentford not found in standings")
            return None
    except Exception as e:
        print(f"Error getting Brentford points: {e}")
        return None

# ==============================
# PREDICTION FUNCTIONS
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
    """Adjust draw probability to be the second-highest probability after the favored team's win probability"""
    total_prob = home_win_prob + draw_prob + away_win_prob

    # Identify the stronger team
    if home_win_prob > away_win_prob:
        favored_win_prob = home_win_prob
        underdog_win_prob = away_win_prob
    else:
        favored_win_prob = away_win_prob
        underdog_win_prob = home_win_prob

    # Ensure draw probability is at least the second-highest probability
    adjusted_draw_prob = max(draw_prob, min(favored_win_prob * 0.75, 0.35))

    # Normalize probabilities to sum to 1
    normalization_factor = total_prob / (favored_win_prob + adjusted_draw_prob + underdog_win_prob)

    return (
        home_win_prob * normalization_factor,
        adjusted_draw_prob * normalization_factor,
        away_win_prob * normalization_factor
    )

def find_most_likely_scores(prob_matrix, outcome_type, max_scores=3):
    """Find the most likely scores for a specific outcome type"""
    scores = []

    # Create mask for the relevant part of the matrix based on outcome
    if outcome_type == 'home_win':
        mask = np.tril(np.ones_like(prob_matrix), -1).astype(bool)
    elif outcome_type == 'draw':
        mask = np.eye(prob_matrix.shape[0], dtype=bool)
    elif outcome_type == 'away_win':
        mask = np.triu(np.ones_like(prob_matrix), 1).astype(bool)
    else:
        raise ValueError("outcome_type must be 'home_win', 'draw', or 'away_win'")

    # Apply mask and find most likely scores
    masked_probs = prob_matrix.copy()
    masked_probs[~mask] = 0

    # Get top scores
    for _ in range(min(max_scores, np.sum(mask))):
        if np.max(masked_probs) == 0:
            break

        idx = np.unravel_index(np.argmax(masked_probs), prob_matrix.shape)
        home_goals, away_goals = idx
        probability = masked_probs[idx]

        scores.append((f"{home_goals}-{away_goals}", probability))
        masked_probs[idx] = 0

    return scores

def predict_match(home_df, away_df, home_team, away_team, league_avg_xG):
    """Predict a single match outcome with expected points"""
    try:
        home_xG = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxG/90'].values[0]
        home_xGA = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxGA/90'].values[0]
        away_xG = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxG/90'].values[0]
        away_xGA = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxGA/90'].values[0]

        lambda_home = expected_goals(home_xG, away_xGA, league_avg_xG)
        lambda_away = expected_goals(away_xG, home_xGA, league_avg_xG)

        # Calculate outcome probabilities
        prob_matrix = poisson_prob_matrix(lambda_home, lambda_away)
        home_win_prob = np.sum(np.tril(prob_matrix, -1))
        draw_prob = np.sum(np.diag(prob_matrix))
        away_win_prob = np.sum(np.triu(prob_matrix, 1))

        # Adjust probabilities
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
            'expected_goals_home': round(lambda_home, 2),
            'expected_goals_away': round(lambda_away, 2)
        }
    except Exception as e:
        print(f"Error predicting match {home_team} vs {away_team}: {e}")
        return None

def calculate_brentford_xpts(prediction, is_home):
    """Calculate expected points for Brentford from a prediction"""
    if is_home:
        xPTS = (prediction['home_win_prob'] * 3 +
                prediction['draw_prob'] * 1)
        xG = prediction['expected_goals_home']
        xGA = prediction['expected_goals_away']
    else:
        xPTS = (prediction['away_win_prob'] * 3 +
                prediction['draw_prob'] * 1)
        xG = prediction['expected_goals_away']
        xGA = prediction['expected_goals_home']

    return round(xPTS, 2), xG, xGA

def process_all_brentford_matches(schedule_df, home_df, away_df, league_avg_xG):
    """Process all Brentford matches and calculate projections"""
    results = []

    for idx, match in schedule_df.iterrows():
        is_home = match['Home'] == 'Brentford'
        opponent = match['Away'] if is_home else match['Home']
        played = match['Played']

        match_data = {
            'date': match['Date'],
            'opponent': opponent,
            'venue': 'Home' if is_home else 'Away',
            'played': played
        }

        if played:
            # Get actual results
            home_goals = int(match['HomeGoals'])
            away_goals = int(match['AwayGoals'])

            brentford_goals = home_goals if is_home else away_goals
            opponent_goals = away_goals if is_home else home_goals

            if brentford_goals > opponent_goals:
                actual_pts = 3
                result = 'W'
            elif brentford_goals < opponent_goals:
                actual_pts = 0
                result = 'L'
            else:
                actual_pts = 1
                result = 'D'

            match_data.update({
                'score': f"{home_goals}-{away_goals}",
                'brentford_goals': brentford_goals,
                'opponent_goals': opponent_goals,
                'result': result,
                'actual_pts': actual_pts,
                'predicted_score': None,
                'xPTS': None,
                'xG': None,
                'xGA': None,
                'home_win_prob': None,
                'draw_prob': None,
                'away_win_prob': None
            })
        else:
            # Predict future match
            if is_home:
                prediction = predict_match(home_df, away_df, 'Brentford', opponent, league_avg_xG)
            else:
                prediction = predict_match(home_df, away_df, opponent, 'Brentford', league_avg_xG)

            if prediction:
                xPTS, xG, xGA = calculate_brentford_xpts(prediction, is_home)

                match_data.update({
                    'score': None,
                    'brentford_goals': None,
                    'opponent_goals': None,
                    'result': None,
                    'actual_pts': None,
                    'predicted_score': prediction['predicted_score'],
                    'xPTS': xPTS,
                    'xG': xG,
                    'xGA': xGA,
                    'home_win_prob': prediction['home_win_prob'] if is_home else prediction['away_win_prob'],
                    'draw_prob': prediction['draw_prob'],
                    'away_win_prob': prediction['away_win_prob'] if is_home else prediction['home_win_prob']
                })

        results.append(match_data)

    return pd.DataFrame(results)

# ==============================
# VISUALIZATION FUNCTIONS
# ==============================

def create_xpts_progression_graph(matches_df, current_stats):
    """Create a line graph showing cumulative points progression"""
    # Handle empty dataframe
    if matches_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title='Brentford Points Progression (2024/25 Season)',
            xaxis_title='Date',
            yaxis_title='Cumulative Points',
            template='plotly_dark',
            height=500,
            annotations=[{
                'text': 'No data available',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20}
            }]
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

    # Add vertical line at current date
    current_date = datetime.now()
    fig.add_vline(
        x=current_date,
        line_dash="dot",
        line_color="yellow",
        annotation_text="Today",
        annotation_position="top"
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
    # Handle empty dataframe
    if matches_df.empty:
        return pd.DataFrame([{'Date': 'N/A', 'Opponent': 'No data', 'Venue': 'N/A', 'Status': 'No data available'}])

    display_df = matches_df.copy()

    # Format date
    display_df['Date'] = display_df['date'].dt.strftime('%Y-%m-%d')

    # Create display columns
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
                'xPTS': row['xPTS'],
                'xG': row['xG'],
                'xGA': row['xGA'],
                'Win%': f"{row['home_win_prob']*100:.0f}%" if row['home_win_prob'] else 'N/A',
                'Draw%': f"{row['draw_prob']*100:.0f}%" if row['draw_prob'] else 'N/A',
                'Status': 'Upcoming'
            })

    return pd.DataFrame(table_data)

# ==============================
# INITIALIZE DATA
# ==============================

print("Initializing Brentford xPTS Dashboard...")
try:
    print("Fetching team statistics...")
    home_df, away_df, standings_df = fetch_team_stats()

    print("Calculating league averages...")
    league_avg_xG_home = home_df['Normalized wxG/90'].mean()
    league_avg_xG_away = away_df['Normalized wxG/90'].mean()
    league_avg_xG = (league_avg_xG_home + league_avg_xG_away) / 2

    print("Fetching Brentford schedule...")
    brentford_schedule = fetch_brentford_schedule()

    print("Getting Brentford current stats...")
    brentford_stats = get_brentford_current_points(standings_df)

    print("Processing all Brentford matches...")
    all_matches = process_all_brentford_matches(brentford_schedule, home_df, away_df, league_avg_xG)

    # Calculate projections
    future_matches = all_matches[~all_matches['played']]
    total_xPTS = future_matches['xPTS'].sum()
    total_xG = future_matches['xG'].sum()
    total_xGA = future_matches['xGA'].sum()

    projected_final_pts = brentford_stats['points'] + total_xPTS
    projected_final_gf = brentford_stats['goals_for'] + total_xG
    projected_final_ga = brentford_stats['goals_against'] + total_xGA
    projected_final_gd = projected_final_gf - projected_final_ga

    print(f"Data loaded successfully!")
    print(f"Current Points: {brentford_stats['points']}")
    print(f"Projected Final Points: {projected_final_pts:.1f}")

except Exception as e:
    print(f"Error initializing data: {e}")
    import traceback
    traceback.print_exc()
    # Provide placeholder data
    brentford_stats = {
        'points': 0,
        'matches_played': 0,
        'goals_for': 0,
        'goals_against': 0,
        'goal_difference': 0
    }
    all_matches = pd.DataFrame()
    total_xPTS = 0
    total_xG = 0
    total_xGA = 0
    projected_final_pts = 0
    projected_final_gf = 0
    projected_final_ga = 0
    projected_final_gd = 0

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
    html.H5("2024/25 Premier League Season Projection", className="text-center mb-4 text-muted"),

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
                                'border': '1px solid #555'
                            },
                            style_header={
                                'backgroundColor': '#1a252f',
                                'fontWeight': 'bold',
                                'border': '1px solid #555'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'Result', 'filter_query': '{Result} = W'},
                                    'color': '#00ff00'
                                },
                                {
                                    'if': {'column_id': 'Result', 'filter_query': '{Result} = L'},
                                    'color': '#ff6b6b'
                                },
                                {
                                    'if': {'column_id': 'Result', 'filter_query': '{Result} = D'},
                                    'color': '#ffa500'
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
                                'border': '1px solid #555'
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
    ])

], fluid=True, className="p-4")

# ==============================
# RUN THE APP
# ==============================
if __name__ == '__main__':
    print("Starting Brentford xPTS Dashboard...")
    app.run_server(debug=True, port=8899)

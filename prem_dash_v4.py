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

def fetch_fresh_data():
    """Fetch team performance data from fbref.com for the top 5 European leagues"""
    import requests
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Premier League
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    dfs = pd.read_html(html)

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

    pl_home = home_df
    pl_away = away_df

    home = pl_home
    away = pl_away
    
    home['wxG'] = (home['xG'] * 0.7 + home['GF'] * 0.3).round(2)
    home['wxGA'] = (home['xGA'] * 0.7 + home['GA'] * 0.3).round(2)

    away['wxG'] = (away['xG'] * 0.7 + away['GF'] * 0.3).round(2)
    away['wxGA'] = (away['xGA'] * 0.7 + away['GA'] * 0.3).round(2)

    home['Normalized wxG/90'] = (home['wxG'] / home['MP']).round(2)
    away['Normalized wxG/90'] = (away['wxG'] / away['MP']).round(2)

    home['Normalized wxGA/90'] = (home['wxGA'] / home['MP']).round(2)
    away['Normalized wxGA/90'] = (away['wxGA'] / away['MP']).round(2)

    home = home.sort_values(by='Normalized wxG/90', ascending=False)
    away = away.sort_values(by='Normalized wxG/90', ascending=False)
    
    # Get current timestamp
    last_updated = datetime.now().strftime("%m/%d/%y %H:%M")
    
    return home, away, last_updated

def fetch_pl_match_data():
    """Fetch individual Premier League match data with improved error handling"""
    try:
        url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers).text
        
        # Parse the matches table
        matches_df = pd.read_html(html)[0]
        
        # Clean up the dataframe
        matches_df = matches_df.dropna(subset=['Score'])  # Only keep matches that have been played
        
        # Extract home and away teams
        matches_df['Home'] = matches_df['Home'].str.strip()
        matches_df['Away'] = matches_df['Away'].str.strip()
        
        # Parse score into home_goals and away_goals - make this more robust
        try:
            # Try different possible delimiters for the score
            for delimiter in ['–', '-', ':', ' ']:
                try:
                    matches_df[['HomeGoals', 'AwayGoals']] = matches_df['Score'].str.split(delimiter, expand=True).astype(int)
                    print(f"Score parsed using delimiter: '{delimiter}'")
                    break
                except:
                    continue
            
            # If we still don't have goals columns, create them manually
            if 'HomeGoals' not in matches_df.columns or 'AwayGoals' not in matches_df.columns:
                # Fallback: Extract numbers from score
                matches_df['HomeGoals'] = matches_df['Score'].str.extract('(\d+)').astype(int)
                matches_df['AwayGoals'] = matches_df['Score'].str.extract('.*?(\d+)').astype(int)
        except Exception as e:
            print(f"Error parsing scores: {e}")
            # Create placeholder columns so the rest of the code doesn't fail
            matches_df['HomeGoals'] = 0
            matches_df['AwayGoals'] = 0
        
        # Determine match result
        matches_df['Result'] = np.where(matches_df['HomeGoals'] > matches_df['AwayGoals'], 'Home Win',
                               np.where(matches_df['HomeGoals'] < matches_df['AwayGoals'], 'Away Win', 'Draw'))
        
        # Convert date string to datetime
        matches_df['Date'] = pd.to_datetime(matches_df['Date'])
        
        # Sort by date
        matches_df = matches_df.sort_values('Date')
        
        return matches_df
    except Exception as e:
        print(f"Error fetching match data: {e}")
        import traceback
        traceback.print_exc()
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['Date', 'Home', 'Away', 'Score', 'HomeGoals', 'AwayGoals', 'Result'])

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
    adjusted_draw_prob = max(draw_prob, min(favored_win_prob * 0.75, 0.35))  # Cap draw at 35%
    
    # Normalize probabilities to sum to 1
    normalization_factor = total_prob / (favored_win_prob + adjusted_draw_prob + underdog_win_prob)
    
    return {
        "Home Win Probability": (home_win_prob * normalization_factor).round(2),
        "Draw Probability": (adjusted_draw_prob * normalization_factor).round(2),
        "Away Win Probability": (away_win_prob * normalization_factor).round(2)
    }

def find_most_likely_scores(prob_matrix, outcome_type, max_scores=3):
    """
    Find the most likely scores for a specific outcome type
    
    Parameters:
    prob_matrix: Poisson probability matrix
    outcome_type: 'home_win', 'draw', or 'away_win'
    max_scores: Maximum number of scores to return
    
    Returns:
    List of tuples (score, probability)
    """
    scores = []
    
    # Create mask for the relevant part of the matrix based on outcome
    if outcome_type == 'home_win':
        # Home team scores more (below the diagonal)
        mask = np.tril(np.ones_like(prob_matrix), -1).astype(bool)
    elif outcome_type == 'draw':
        # Equal scores (diagonal)
        mask = np.eye(prob_matrix.shape[0], dtype=bool)
    elif outcome_type == 'away_win':
        # Away team scores more (above the diagonal)
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
        
        # Find index of maximum probability
        idx = np.unravel_index(np.argmax(masked_probs), prob_matrix.shape)
        home_goals, away_goals = idx
        probability = masked_probs[idx]
        
        # Add to results
        scores.append((f"{home_goals}-{away_goals}", probability))
        
        # Zero out this cell so we find the next highest
        masked_probs[idx] = 0
    
    return scores

def match_outcome_prob(home_df, away_df, home_team, away_team, league_avg_xG, max_goals=10):
    """Calculate match outcome probabilities for a single match"""
    home_xG = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxG/90'].values[0]
    home_xGA = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxGA/90'].values[0]
    away_xG = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxG/90'].values[0]
    away_xGA = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxGA/90'].values[0]

    lambda_A = expected_goals(home_xG, away_xGA, league_avg_xG)
    lambda_B = expected_goals(away_xG, home_xGA, league_avg_xG)

    # Calculate outcome probabilities
    prob_matrix = poisson_prob_matrix(lambda_A, lambda_B, max_goals)
    home_win_prob = np.sum(np.tril(prob_matrix, -1))  # Home team wins
    draw_prob = np.sum(np.diag(prob_matrix))  # Draw
    away_win_prob = np.sum(np.triu(prob_matrix, 1))  # Away team wins

    adjusted_probs = adjust_draw_probability(home_win_prob, draw_prob, away_win_prob)
    
    # Determine most likely outcome
    home_win_prob_adj = adjusted_probs["Home Win Probability"]
    draw_prob_adj = adjusted_probs["Draw Probability"]
    away_win_prob_adj = adjusted_probs["Away Win Probability"]
    
    if home_win_prob_adj > draw_prob_adj and home_win_prob_adj > away_win_prob_adj:
        predicted_outcome = f"{home_team} Win"
        outcome_type = 'home_win'
    elif away_win_prob_adj > home_win_prob_adj and away_win_prob_adj > draw_prob_adj:
        predicted_outcome = f"{away_team} Win"
        outcome_type = 'away_win'
    else:
        predicted_outcome = "Draw"
        outcome_type = 'draw'
    
    # Get most likely scores for the predicted outcome
    most_likely_scores = find_most_likely_scores(prob_matrix, outcome_type)
    
    # Also find the overall most likely score (regardless of outcome)
    most_likely_score_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    most_likely_home_goals = most_likely_score_idx[0]
    most_likely_away_goals = most_likely_score_idx[1]
    most_likely_score = f"{most_likely_home_goals}-{most_likely_away_goals}"
    most_likely_score_prob = prob_matrix[most_likely_home_goals, most_likely_away_goals]
    
    # Get most likely score that aligns with the predicted outcome
    most_likely_consistent_score = most_likely_scores[0][0] if most_likely_scores else "N/A"
    most_likely_consistent_score_prob = most_likely_scores[0][1] if most_likely_scores else 0

    # Prepare the results object with all relevant information
    result = {
        "Home Team": home_team,
        "Away Team": away_team,
        "Predicted Outcome": predicted_outcome,
        "Most Likely Score (Overall)": most_likely_score,
        "Score Probability": f"{(most_likely_score_prob * 100).round(1)}%",
        "Most Likely Score (Consistent)": most_likely_consistent_score,
        "Consistent Score Probability": f"{(most_likely_consistent_score_prob * 100).round(1)}%",
        "Expected Goals (Home)": lambda_A.round(2),
        "Expected Goals (Away)": lambda_B.round(2),
        **adjusted_probs
    }
    
    return result

# ==============================
# BACKTESTING FUNCTIONS
# ==============================

def calculate_team_stats(team, matches_df, as_home=True, as_away=True, last_n=None):
    """Calculate a team's performance stats based on prior matches with improved error handling"""
    try:
        # Check if DataFrame has required columns
        required_columns = ['Home', 'Away', 'HomeGoals', 'AwayGoals']
        for col in required_columns:
            if col not in matches_df.columns:
                print(f"Missing required column in calculate_team_stats: {col}")
                # Return empty stats to avoid breaking downstream functions
                return {'MP': 0}, {'MP': 0}
        
        # Filter matches for this team
        home_matches = matches_df[matches_df['Home'] == team].copy() if as_home else pd.DataFrame()
        away_matches = matches_df[matches_df['Away'] == team].copy() if as_away else pd.DataFrame()
        
        # Optionally limit to last N matches
        if last_n is not None and not home_matches.empty:
            home_matches = home_matches.tail(last_n)
        if last_n is not None and not away_matches.empty:
            away_matches = away_matches.tail(last_n)
        
        # Calculate home stats with error handling
        home_stats = {
            'MP': len(home_matches),
            'W': sum(home_matches['HomeGoals'] > home_matches['AwayGoals']) if not home_matches.empty else 0,
            'D': sum(home_matches['HomeGoals'] == home_matches['AwayGoals']) if not home_matches.empty else 0,
            'L': sum(home_matches['HomeGoals'] < home_matches['AwayGoals']) if not home_matches.empty else 0,
            'GF': home_matches['HomeGoals'].sum() if not home_matches.empty else 0,
            'GA': home_matches['AwayGoals'].sum() if not home_matches.empty else 0,
        }
        
        # Calculate away stats with error handling
        away_stats = {
            'MP': len(away_matches),
            'W': sum(away_matches['AwayGoals'] > away_matches['HomeGoals']) if not away_matches.empty else 0,
            'D': sum(away_matches['AwayGoals'] == away_matches['HomeGoals']) if not away_matches.empty else 0,
            'L': sum(away_matches['AwayGoals'] < away_matches['HomeGoals']) if not away_matches.empty else 0,
            'GF': away_matches['AwayGoals'].sum() if not away_matches.empty else 0,
            'GA': away_matches['HomeGoals'].sum() if not away_matches.empty else 0,
        }
        
        # Calculate derived metrics
        if home_stats['MP'] > 0:
            home_stats['GD'] = home_stats['GF'] - home_stats['GA']
            home_stats['Pts'] = home_stats['W'] * 3 + home_stats['D']
            home_stats['GF/90'] = home_stats['GF'] / home_stats['MP']
            home_stats['GA/90'] = home_stats['GA'] / home_stats['MP']
        
        if away_stats['MP'] > 0:
            away_stats['GD'] = away_stats['GF'] - away_stats['GA']
            away_stats['Pts'] = away_stats['W'] * 3 + away_stats['D']
            away_stats['GF/90'] = away_stats['GF'] / away_stats['MP']
            away_stats['GA/90'] = away_stats['GA'] / away_stats['MP']
        
        return home_stats, away_stats
    except Exception as e:
        print(f"Error in calculate_team_stats: {e}")
        import traceback
        traceback.print_exc()
        # Return empty stats dictionaries
        return {'MP': 0}, {'MP': 0}

def predict_match_outcome(home_team, away_team, prior_matches):
    """Predict match outcome using only data from prior matches"""
    try:
        # Calculate team stats from prior matches
        home_team_home_stats, _ = calculate_team_stats(home_team, prior_matches, as_away=False)
        _, away_team_away_stats = calculate_team_stats(away_team, prior_matches, as_home=False)
        
        # Handle cases with insufficient data
        if home_team_home_stats.get('MP', 0) == 0 or away_team_away_stats.get('MP', 0) == 0:
            return {
                'predicted_result': 'Insufficient Data',
                'home_win_prob': None,
                'draw_prob': None,
                'away_win_prob': None,
                'predicted_score': None
            }
        
        # Calculate league averages
        all_teams = list(set(prior_matches['Home']).union(set(prior_matches['Away'])))
        league_total_goals = prior_matches['HomeGoals'].sum() + prior_matches['AwayGoals'].sum()
        league_total_matches = len(prior_matches)
        league_avg_goals = league_total_goals / (2 * league_total_matches) if league_total_matches > 0 else 1.0
        
        # Replace xG with actual goals for simplicity
        # In a full implementation, you would use actual xG data
        home_attack_strength = home_team_home_stats.get('GF/90', 0)
        home_defense_weakness = home_team_home_stats.get('GA/90', 0)
        away_attack_strength = away_team_away_stats.get('GF/90', 0)
        away_defense_weakness = away_team_away_stats.get('GA/90', 0)
        
        # Calculate expected goals
        lambda_home = expected_goals(home_attack_strength, away_defense_weakness, league_avg_goals)
        lambda_away = expected_goals(away_attack_strength, home_defense_weakness, league_avg_goals)
        
        # Calculate outcome probabilities using Poisson
        prob_matrix = poisson_prob_matrix(lambda_home, lambda_away)
        
        home_win_prob = np.sum(np.tril(prob_matrix, -1))
        draw_prob = np.sum(np.diag(prob_matrix))
        away_win_prob = np.sum(np.triu(prob_matrix, 1))
        
        # Adjust to ensure they sum to 1
        total_prob = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total_prob
        draw_prob /= total_prob
        away_win_prob /= total_prob
        
        # Determine most likely outcome
        if home_win_prob > draw_prob and home_win_prob > away_win_prob:
            predicted_result = 'Home Win'
            outcome_type = 'home_win'
        elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
            predicted_result = 'Away Win'
            outcome_type = 'away_win'
        else:
            predicted_result = 'Draw'
            outcome_type = 'draw'

        # Get most likely score that aligns with predicted outcome
        most_likely_scores = find_most_likely_scores(prob_matrix, outcome_type)
        consistent_score = most_likely_scores[0][0] if most_likely_scores else "N/A"
        
        return {
            'predicted_result': predicted_result,
            'home_win_prob': round(home_win_prob, 3),
            'draw_prob': round(draw_prob, 3),
            'away_win_prob': round(away_win_prob, 3),
            'predicted_score': consistent_score
        }
    except Exception as e:
        print(f"Error in predict_match_outcome: {e}")
        import traceback
        traceback.print_exc()
        return {
            'predicted_result': 'Error',
            'home_win_prob': None,
            'draw_prob': None, 
            'away_win_prob': None,
            'predicted_score': None
        }

def get_team_recent_matches(team, matches_df, n=3, venue_filter=None):
    """
    Get the n most recent matches for a team with improved error handling
    
    Parameters:
    team (str): The team name
    matches_df (DataFrame): DataFrame with match data
    n (int): Number of matches to get
    venue_filter (str, optional): Filter for 'Home' or 'Away' matches only
    """
    try:
        # Ensure we have the required columns
        required_columns = ['Date', 'Home', 'Away', 'HomeGoals', 'AwayGoals', 'Result']
        for col in required_columns:
            if col not in matches_df.columns:
                print(f"Missing required column: {col}")
                return []  # Return empty list if data structure isn't as expected
        
        # Make a copy to avoid modifying the original
        matches_df = matches_df.copy()
        
        # Sort by date descending to get most recent first
        matches_df = matches_df.sort_values('Date', ascending=False)
        
        # Filter for matches where the team was playing
        if venue_filter == 'Home':
            team_matches = matches_df[matches_df['Home'] == team]
            print(f"Filtered for {team}'s home matches: {len(team_matches)} matches found")
        elif venue_filter == 'Away':
            team_matches = matches_df[matches_df['Away'] == team]
            print(f"Filtered for {team}'s away matches: {len(team_matches)} matches found")
        else:
            team_matches = matches_df[(matches_df['Home'] == team) | (matches_df['Away'] == team)]
            print(f"Found {len(team_matches)} total matches for {team}")
        
        if team_matches.empty:
            print(f"No {venue_filter.lower() if venue_filter else ''} matches found for team: {team}")
            return []
            
        # Take the n most recent matches
        recent_matches = team_matches.head(n)
        
        # Sort by date (earliest first)
        recent_matches = recent_matches.sort_values('Date')
        
        # Format the data for display
        results = []
        for _, match in recent_matches.iterrows():
            try:
                # Determine if team was home or away
                is_home = match['Home'] == team
                
                # Get opponent
                opponent = match['Away'] if is_home else match['Home']
                
                # Get score
                score = f"{match['HomeGoals']}-{match['AwayGoals']}"
                
                # Get actual result from team's perspective
                if is_home:
                    if match['HomeGoals'] > match['AwayGoals']:
                        actual_result = "Win"
                    elif match['HomeGoals'] < match['AwayGoals']:
                        actual_result = "Loss"
                    else:
                        actual_result = "Draw"
                else:
                    if match['AwayGoals'] > match['HomeGoals']:
                        actual_result = "Win"
                    elif match['AwayGoals'] < match['HomeGoals']:
                        actual_result = "Loss"
                    else:
                        actual_result = "Draw"
                
                # Make a retroactive prediction
                prior_matches = matches_df[matches_df['Date'] < match['Date']]
                if len(prior_matches) < 3:  # Need some history for prediction
                    prediction = "Insufficient data"
                    model_predicted_outcome = "N/A"
                    predicted_score = "N/A"
                else:
                    try:
                        if is_home:
                            pred = predict_match_outcome(team, opponent, prior_matches)
                            # Store the model's predicted outcome
                            model_predicted_outcome = pred['predicted_result']
                            predicted_score = pred['predicted_score']
                        else:
                            pred = predict_match_outcome(opponent, team, prior_matches)
                            # Store the model's predicted outcome (from opposite perspective)
                            if pred['predicted_result'] == 'Home Win':
                                model_predicted_outcome = 'Away Loss'
                            elif pred['predicted_result'] == 'Away Win':
                                model_predicted_outcome = 'Home Loss'
                            else:
                                model_predicted_outcome = 'Draw'
                            
                            # Reverse the score for away team perspective
                            if pred['predicted_score'] and pred['predicted_score'] != "N/A":
                                home_goals, away_goals = pred['predicted_score'].split('-')
                                predicted_score = f"{away_goals}-{home_goals}"
                            else:
                                predicted_score = "N/A"
                            
                        # Translate prediction to team's perspective
                        if pred['predicted_result'] == 'Insufficient Data':
                            prediction = "Insufficient data"
                            model_predicted_outcome = "N/A"
                            predicted_score = "N/A"
                        elif is_home:
                            if pred['predicted_result'] == 'Home Win':
                                prediction = "Win"
                            elif pred['predicted_result'] == 'Away Win':
                                prediction = "Loss"
                            else:
                                prediction = "Draw"
                        else:
                            if pred['predicted_result'] == 'Away Win':
                                prediction = "Win"
                            elif pred['predicted_result'] == 'Home Win':
                                prediction = "Loss"
                            else:
                                prediction = "Draw"
                    except Exception as e:
                        print(f"Error making prediction: {e}")
                        prediction = "Error"
                        model_predicted_outcome = "Error"
                        predicted_score = "Error"
                
                # Check if prediction was correct
                correct = actual_result == prediction if prediction not in ["Insufficient data", "Error"] else None
                
                # Format date
                date_str = match['Date'].strftime('%Y-%m-%d')
                
                # Store result
                results.append({
                    'date': date_str,
                    'opponent': opponent,
                    'score': score,
                    'venue': 'Home' if is_home else 'Away',
                    'actual_result': actual_result,
                    'prediction': prediction,
                    'model_outcome': model_predicted_outcome,
                    'predicted_score': predicted_score,
                    'correct': correct
                })
            except Exception as e:
                print(f"Error processing match: {e}")
                continue  # Skip this match if there's an error
        
        return results
    except Exception as e:
        print(f"Error in get_team_recent_matches: {e}")
        import traceback
        traceback.print_exc()
        return []  # Return empty list on error

def create_recent_matches_table(team, matches, title=None):
    """Create a formatted table showing recent matches for a team"""
    if not matches:
        return html.Div([
            html.H5(f"{title or team} - Recent Matches", className="text-center"),
            html.P("No recent match data available", className="text-center text-muted")
        ])
    
    # Create table header
    header = html.Thead([
        html.Tr([
            html.Th("Date"),
            html.Th("Opponent"),
            html.Th("Venue"),
            html.Th("Score"),
            html.Th("Result"),
            html.Th("Model Predicted"),
            html.Th("Predicted Score"),
            html.Th("From Team View"),
            html.Th("Correct?")
        ])
    ])
    
    # Create table rows
    rows = []
    for match in matches:
        # Style for the "Correct?" column
        if match['correct'] is True:
            correct_style = {'color': 'green'}
            correct_text = "✓"
        elif match['correct'] is False:
            correct_style = {'color': 'red'}
            correct_text = "✗"
        else:
            correct_style = {'color': 'gray'}
            correct_text = "N/A"
        
        # Style for result column
        if match['actual_result'] == 'Win':
            result_style = {'color': 'green'}
        elif match['actual_result'] == 'Loss':
            result_style = {'color': 'red'}
        else:
            result_style = {'color': 'orange'}
            
        # Create row
        rows.append(html.Tr([
            html.Td(match['date']),
            html.Td(match['opponent']),
            html.Td(match['venue']),
            html.Td(match['score']),
            html.Td(match['actual_result'], style=result_style),
            html.Td(match['model_outcome']),
            html.Td(match['predicted_score']),
            html.Td(match['prediction']),
            html.Td(correct_text, style=correct_style)
        ]))
    
    body = html.Tbody(rows)
    
    # Create table
    table = dbc.Table(
        [header, body],
        bordered=True,
        dark=True,
        hover=True,
        responsive=True,
        striped=True,
        style={'margin': '0 auto', 'width': '95%'}
    )
    
    # Create a containing div with header and explanation
    return html.Div([
        html.H5(f"{title or team} - Recent Matches", className="text-center"),
        html.P([
            "All predictions are out-of-sample (made using only data available prior to each match)."
        ], className="small text-muted text-center"),
        table
    ])

def get_team_predictability(matches_df, premier_league_teams, venue='all'):
    """Calculate the predictability of all Premier League teams based on recent matches"""
    if matches_df.empty or not premier_league_teams:
        return []
    
    results = []
    
    for team in premier_league_teams:
        try:
            # Get team's recent matches based on venue filter
            if venue == 'Home':
                recent_matches = get_team_recent_matches(team, matches_df, n=3, venue_filter='Home')
            elif venue == 'Away':
                recent_matches = get_team_recent_matches(team, matches_df, n=3, venue_filter='Away')
            else:
                recent_matches = get_team_recent_matches(team, matches_df, n=3)
            
            # Calculate predictability metrics
            total_matches = len(recent_matches)
            if total_matches == 0:
                continue
                
            correct_predictions = sum(1 for match in recent_matches if match['correct'] is True)
            incorrect_predictions = sum(1 for match in recent_matches if match['correct'] is False)
            undefined_predictions = total_matches - correct_predictions - incorrect_predictions
            
            accuracy = correct_predictions / (correct_predictions + incorrect_predictions) if (correct_predictions + incorrect_predictions) > 0 else None
            
            results.append({
                'Team': team,
                'Matches': total_matches,
                'Correct': correct_predictions,
                'Incorrect': incorrect_predictions,
                'No Prediction': undefined_predictions,
                'Accuracy': accuracy,
                'Predictability': 'High' if accuracy is not None and accuracy >= 0.7 else 
                               'Medium' if accuracy is not None and accuracy >= 0.4 else 
                               'Low' if accuracy is not None else 'Unknown'
            })
        except Exception as e:
            print(f"Error calculating predictability for {team}: {e}")
            
    # Sort by accuracy (highest first)
    results.sort(key=lambda x: (x['Accuracy'] if x['Accuracy'] is not None else -1), reverse=True)
    
    return results

def initialize_backtesting():
    """Initialize backtesting data on app startup"""
    try:
        print("Initializing backtesting data...")
        
        # Fetch match data
        matches_df = fetch_pl_match_data()
        print(f"Retrieved {len(matches_df)} Premier League matches")
        
        # Return match history only (we don't need full backtesting for this simplified approach)
        return {
            'match_history': matches_df.to_dict('records') if not matches_df.empty else []
        }
    except Exception as e:
        print(f"Error initializing backtesting: {e}")
        import traceback
        traceback.print_exc()
        return {
            'match_history': []
        }

def dict_to_table(results_dict):
    """Convert a dictionary to a Dash Bootstrap Components table"""
    # Items to show in this order
    keys_to_show = [
        "Home Team", "Away Team", "Predicted Outcome", 
        "Most Likely Score (Consistent)", "Consistent Score Probability",
        "Expected Goals (Home)", "Expected Goals (Away)",
        "Home Win Probability", "Draw Probability", "Away Win Probability"
    ]
    
    # Optional keys that might not be in the dict
    optional_keys = ["Most Likely Score (Overall)", "Score Probability"]
    
    # Start with the mandatory keys
    rows = []
    for k in keys_to_show:
        if k in results_dict:
            rows.append(html.Tr([html.Td(k), html.Td(results_dict[k])]))
    
    # Add optional keys if present
    for k in optional_keys:
        if k in results_dict:
            rows.append(html.Tr([html.Td(k), html.Td(results_dict[k])]))
    
    return dbc.Table(
        [html.Tbody(rows)],
        bordered=True,
        dark=True,
        hover=True,
        responsive=True,
        striped=True,
        style={'margin': '0 auto', 'width': '50%'}
    )

def recalc_dataframes(home_data, away_data):
    """Recalculate dataframes with normalized values"""
    # Convert stored data to DataFrames
    home_df = pd.DataFrame(home_data) if isinstance(home_data, list) else home
    away_df = pd.DataFrame(away_data) if isinstance(away_data, list) else away
    
    # Calculate league average xG
    league_avg_xG_home = home_df['Normalized wxG/90'].mean()
    league_avg_xG_away = away_df['Normalized wxG/90'].mean()
    league_avg_xG_val = (league_avg_xG_home + league_avg_xG_away) / 2
    
    return home_df, away_df, league_avg_xG_val

def create_team_predictability_table(teams_data, title):
    """Create a table showing team predictability"""
    if not teams_data:
        return html.Div([
            html.H5(title, className="text-center"),
            html.P("No predictability data available", className="text-center text-muted")
        ])
    
    # Create table header
    header = html.Thead([
        html.Tr([
            html.Th("Team"),
            html.Th("Matches"),
            html.Th("Correct"),
            html.Th("Incorrect"),
            html.Th("No Prediction"),
            html.Th("Accuracy"),
            html.Th("Predictability")
        ])
    ])
    
    # Create table rows
    rows = []
    for team_data in teams_data:
        # Style for predictability
        pred_style = {}
        if team_data['Predictability'] == 'High':
            pred_style = {'color': 'green'}
        elif team_data['Predictability'] == 'Medium':
            pred_style = {'color': 'orange'}
        elif team_data['Predictability'] == 'Low':
            pred_style = {'color': 'red'}
        
        # Format accuracy
        accuracy_text = f"{team_data['Accuracy']:.1%}" if team_data['Accuracy'] is not None else "N/A"
        
        # Create row
        rows.append(html.Tr([
            html.Td(team_data['Team']),
            html.Td(team_data['Matches']),
            html.Td(team_data['Correct']),
            html.Td(team_data['Incorrect']),
            html.Td(team_data['No Prediction']),
            html.Td(accuracy_text),
            html.Td(team_data['Predictability'], style=pred_style)
        ]))
    
    body = html.Tbody(rows)
    
    # Create table
    table = dbc.Table(
        [header, body],
        bordered=True,
        dark=True,
        hover=True,
        responsive=True,
        striped=True,
        style={'margin': '0 auto', 'width': '95%'}
    )
    
    # Create a containing div with header
    return html.Div([
        html.H3(title, className="text-center mb-3"),
        table
    ], className="mt-4")

# ==============================
# INITIALIZE DATA
# ==============================

print("Initializing data...")
# Initial data fetch
try:
    home, away, last_updated = fetch_fresh_data()
    teams = sorted(home['Squad'].unique())
    print(f"Fetched data for {len(teams)} teams across 5 leagues")
except Exception as e:
    print(f"Error fetching initial data: {e}")
    import traceback
    traceback.print_exc()
    # Provide placeholder data to prevent app from crashing
    home = pd.DataFrame()
    away = pd.DataFrame()
    teams = []
    last_updated = datetime.now().strftime("%m/%d/%y %H:%M")

# Initialize backtesting data
backtesting_data = initialize_backtesting()

# ==============================
# INITIALIZE DASH APP
# ==============================

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# ==============================
# APP LAYOUT DEFINITIONS
# ==============================

# Match layout with recent performance tables
match_layout = dbc.Container([
    dbc.Row([dbc.Col(html.Div("Home Team"), width=12)], justify='center'),
    dbc.Row([dbc.Col(dcc.Dropdown(
        id='home-team', 
        options=[{'label': t, 'value': t} for t in teams],
        value=teams[0] if teams else None,
        clearable=True,
        searchable=True
    ), width=6)], justify='center'),
    dbc.Row([dbc.Col(html.Div("Away Team"), width=12)], justify='center'),
    dbc.Row([dbc.Col(dcc.Dropdown(
        id='away-team', 
        options=[{'label': t, 'value': t} for t in teams],
        value=teams[1] if len(teams) > 1 else None,
        clearable=True,
        searchable=True
    ), width=6)], justify='center'),
    dbc.Row([dbc.Col(html.Button("Calculate Match Outcome", id='match-btn', n_clicks=0, className="btn btn-primary"), width=12)], justify='center', style={'marginTop': '1rem'}),
    dbc.Row([dbc.Col(html.Div(id='match-output'), width=12)], justify='center', style={'marginTop': '1rem'}),
    
    # Note about predicted scores
    dbc.Row([
        dbc.Col(
            html.Div([
                html.P(
                    "Note: The 'Most Likely Score (Consistent)' shows the most probable score that aligns with the predicted outcome. "
                    "This ensures the score prediction matches the outcome prediction.",
                    className="small text-muted font-italic text-center"
                )
            ]),
            width=12
        )
    ], justify='center', style={'marginTop': '0.5rem', 'marginBottom': '1rem'}),
    
    # Add sections for team venue-specific and recent performance
    html.Div(id='home-team-recent-home', style={'marginTop': '2rem'}),
    html.Div(id='away-team-recent-away', style={'marginTop': '2rem'}),
    
    # Add section for recent form (regardless of venue)
    html.H4("Recent Form (Last 3 Matches Any Venue)", className="text-center mt-4"),
    html.Div(id='home-team-recent-form', style={'marginTop': '1rem'}),
    html.Div(id='away-team-recent-form', style={'marginTop': '1rem'})
], fluid=True, style={'textAlign': 'center'})

# Predictability Analysis layout
predictability_layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab([
            html.Div(id='home-predictability-content')
        ], label="Home Predictability"),
        dbc.Tab([
            html.Div(id='away-predictability-content')
        ], label="Away Predictability"),
        dbc.Tab([
            html.Div(id='overall-predictability-content')
        ], label="Overall Predictability")
    ])
], fluid=True)

# Main layout
app.layout = dbc.Container([
    html.H1("YANUS PL", style={'textAlign': 'center', 'marginBottom': '0.5rem'}),
    html.H6(id="last-updated-text", children=f"Last updated: {last_updated}", style={'textAlign': 'center', 'marginBottom': '1rem'}),
    dbc.Button("Refresh Data", id="refresh-button", color="success", className="mb-3", style={'display': 'block', 'margin': '0 auto'}),
    
    html.Hr(),
    
    # Main tabs
    dcc.Tabs(id='tabs', value='match', children=[
        dcc.Tab(label='Match Prediction', value='match'),
        dcc.Tab(label='Team Predictability', value='predictability')
    ]),
    html.Div(id='tab-content'),
    
    # Store components for our data
    dcc.Store(id='home-data-store', data=home.to_dict('records')),
    dcc.Store(id='away-data-store', data=away.to_dict('records')),
    dcc.Store(id='teams-store', data=teams),
    dcc.Store(id='backtesting-data-store', data=backtesting_data)
], fluid=True, style={'textAlign': 'center', 'marginTop': '1rem'})

# ==============================
# CALLBACKS
# ==============================

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab):
    if tab == 'match':
        return match_layout
    elif tab == 'predictability':
        return predictability_layout

@app.callback(
    [Output('home-data-store', 'data'),
     Output('away-data-store', 'data'),
     Output('teams-store', 'data'),
     Output('last-updated-text', 'children')],
    [Input('refresh-button', 'n_clicks')],
    prevent_initial_call=True
)
def refresh_data(n_clicks):
    """Callback to refresh data when the refresh button is clicked"""
    if n_clicks:
        # Fetch fresh data
        try:
            new_home, new_away, new_timestamp = fetch_fresh_data()
            
            # Convert DataFrame to dict for dcc.Store
            home_dict = new_home.to_dict('records')
            away_dict = new_away.to_dict('records')
            
            # Get updated team list
            updated_teams = sorted(new_home['Squad'].unique())
            
            return home_dict, away_dict, updated_teams, f"Last updated: {new_timestamp}"
        except Exception as e:
            print(f"Error refreshing data: {e}")
            import traceback
            traceback.print_exc()
            return dash.no_update, dash.no_update, dash.no_update, f"Error refreshing data: {e}"
    
    # Prevent update if button wasn't clicked
    raise dash.exceptions.PreventUpdate

@app.callback(
    [Output('home-team', 'options'),
     Output('away-team', 'options')],
    [Input('teams-store', 'data')],
    prevent_initial_call=True
)
def update_team_dropdowns(teams_data):
    """Update team dropdowns when teams list changes"""
    if teams_data:
        options = [{'label': t, 'value': t} for t in teams_data]
        return options, options
    raise dash.exceptions.PreventUpdate

@app.callback(
    [Output('match-output', 'children'),
     Output('home-team-recent-home', 'children'),
     Output('away-team-recent-away', 'children'),
     Output('home-team-recent-form', 'children'),
     Output('away-team-recent-form', 'children')],
    Input('match-btn', 'n_clicks'),
    [State('home-team', 'value'),
     State('away-team', 'value'),
     State('home-data-store', 'data'),
     State('away-data-store', 'data'),
     State('backtesting-data-store', 'data')]
)
def update_match_with_recent(n_clicks, home_team, away_team, home_data, away_data, backtesting_data):
    if n_clicks > 0 and home_team and away_team:
        try:
            # Calculate prediction
            home_df, away_df, league_avg_xG_val = recalc_dataframes(home_data, away_data)
            res = match_outcome_prob(home_df, away_df, home_team, away_team, league_avg_xG_val)
            
            # Get recent match history if available
            home_recent_home = None
            away_recent_away = None
            home_recent_form = None
            away_recent_form = None
            
            if backtesting_data and 'match_history' in backtesting_data and backtesting_data['match_history']:
                try:
                    # Convert match history to DataFrame
                    match_history = pd.DataFrame(backtesting_data['match_history'])
                    match_history['Date'] = pd.to_datetime(match_history['Date'])
                    
                    # Get venue-specific recent matches
                    home_recent_home_matches = get_team_recent_matches(home_team, match_history, n=3, venue_filter='Home')
                    away_recent_away_matches = get_team_recent_matches(away_team, match_history, n=3, venue_filter='Away')
                    
                    # Get overall recent form (any venue)
                    home_recent_form_matches = get_team_recent_matches(home_team, match_history, n=3, venue_filter=None)
                    away_recent_form_matches = get_team_recent_matches(away_team, match_history, n=3, venue_filter=None)
                    
                    # Create tables
                    home_recent_home = create_recent_matches_table(home_team, home_recent_home_matches, f"{home_team} - Last 3 Home Matches")
                    away_recent_away = create_recent_matches_table(away_team, away_recent_away_matches, f"{away_team} - Last 3 Away Matches")
                    
                    home_recent_form = create_recent_matches_table(home_team, home_recent_form_matches, f"{home_team} - Recent Form")
                    away_recent_form = create_recent_matches_table(away_team, away_recent_form_matches, f"{away_team} - Recent Form")
                except Exception as e:
                    print(f"Error creating recent match tables: {e}")
                    import traceback
                    traceback.print_exc()
                    error_msg = html.Div(f"Error retrieving recent matches: {str(e)}")
                    home_recent_home = error_msg
                    away_recent_away = error_msg
                    home_recent_form = error_msg
                    away_recent_form = error_msg
            
            # Return prediction and recent match tables
            return dict_to_table(res), home_recent_home, away_recent_away, home_recent_form, away_recent_form
        except Exception as e:
            error_msg = html.Div(f"Error calculating match outcome: {str(e)}")
            return error_msg, None, None, None, None
    return "", None, None, None, None

@app.callback(
    [Output('home-predictability-content', 'children'),
     Output('away-predictability-content', 'children'),
     Output('overall-predictability-content', 'children')],
    [Input('tabs', 'value'),
     Input('backtesting-data-store', 'data'),
     Input('teams-store', 'data')]
)
def update_predictability_content(active_tab, backtesting_data, teams_data):
    if active_tab != 'predictability' or not backtesting_data or not 'match_history' in backtesting_data:
        return [html.Div()] * 3  # Return empty divs if not on predictability tab
    
    try:
        # Convert match history to DataFrame
        match_history = pd.DataFrame(backtesting_data['match_history'])
        match_history['Date'] = pd.to_datetime(match_history['Date'])
        
        # Get unique Premier League teams
        pl_teams = list(set(match_history['Home']).union(set(match_history['Away'])))
        
        # Calculate predictability metrics
        home_predictability = get_team_predictability(match_history, pl_teams, venue='Home')
        away_predictability = get_team_predictability(match_history, pl_teams, venue='Away')
        overall_predictability = get_team_predictability(match_history, pl_teams, venue='all')
        
        # Create tables
        home_table = create_team_predictability_table(home_predictability, "Home Match Predictability")
        away_table = create_team_predictability_table(away_predictability, "Away Match Predictability")
        overall_table = create_team_predictability_table(overall_predictability, "Overall Predictability")
        
        return home_table, away_table, overall_table
    except Exception as e:
        print(f"Error generating predictability tables: {e}")
        import traceback
        traceback.print_exc()
        error_msg = html.Div(f"Error generating predictability analysis: {str(e)}")
        return error_msg, error_msg, error_msg

# ==============================
# RUN THE APP
# ==============================
if __name__ == '__main__':
    print("Starting Dash app...")
    app.run_server(debug=True, port=8898)

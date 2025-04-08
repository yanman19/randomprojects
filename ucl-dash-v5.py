#ucl dash
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import pandas as pd
import numpy as np
import scipy.stats as stats
import requests
import datetime



# ----------------------------
# Data & Model Functions Setup
# ----------------------------

def load_data():
    url = "https://fbref.com/en/comps/8/Champions-League-Stats"
    headers = {"User-Agent": "Mozilla/5.0"}
    html_text = requests.get(url, headers=headers).text
    dfs = pd.read_html(html_text)

    df = dfs[2]
    df.columns = [
        "Rk", "Squad",
        "Home_MP", "Home_W", "Home_D", "Home_L", "Home_GF", "Home_GA", "Home_GD", "Home_Pts", 'Home_Pts/MP',
        "Home_xG", "Home_xGA", "Home_xGD", "Home_xGD_per_90",
        "Away_MP", "Away_W", "Away_D", "Away_L", "Away_GF", "Away_GA", "Away_GD", "Away_Pts", 'Away_Pts/MP',
        "Away_xG", "Away_xGA", "Away_xGD", "Away_xGD_per_90"
    ]
    df['Country'] = 'UCL'
    df['Squad'] = df['Squad'].astype(str).str.split(' ', 1).str[1]

    home_df = df[[
        "Squad", "Home_MP", "Home_W", "Home_D", "Home_L", "Home_GF", "Home_GA",
        "Home_GD", "Home_Pts", "Home_xG", "Home_xGA", "Home_xGD", "Home_xGD_per_90",
        'Country'
    ]].copy()

    away_df = df[[
        "Squad", "Away_MP", "Away_W", "Away_D", "Away_L", "Away_GF", "Away_GA",
        "Away_GD", "Away_Pts", "Away_xG", "Away_xGA", "Away_xGD", "Away_xGD_per_90",
        'Country'
    ]].copy()

    rename_home = {
        "Home_MP": "MP", "Home_W": "W", "Home_D": "D", "Home_L": "L", "Home_GF": "GF",
        "Home_GA": "GA", "Home_GD": "GD", "Home_Pts": "Pts", "Home_xG": "xG",
        "Home_xGA": "xGA", "Home_xGD": "xGD", "Home_xGD_per_90": "xGD_per_90"
    }
    rename_away = {
        "Away_MP": "MP", "Away_W": "W", "Away_D": "D", "Away_L": "L", "Away_GF": "GF",
        "Away_GA": "GA", "Away_GD": "GD", "Away_Pts": "Pts", "Away_xG": "xG",
        "Away_xGA": "xGA", "Away_xGD": "xGD", "Away_xGD_per_90": "xGD_per_90"
    }

    home_df.rename(columns=rename_home, inplace=True)
    away_df.rename(columns=rename_away, inplace=True)

    home = home_df.copy()
    away = away_df.copy()

    # Calculate wxG, wxGA, and Normalized Metrics (rounded)
    for df_ in [home, away]:
        df_['wxG'] = (df_['xG'] * 0.7 + df_['GF'] * 0.3).round(2)
        df_['wxGA'] = (df_['xGA'] * 0.7 + df_['GA'] * 0.3).round(2)
        df_['Normalized wxG/90'] = (df_['wxG'] / df_['MP']).round(2)
        df_['Normalized wxGA/90'] = (df_['wxGA'] / df_['MP']).round(2)

    # Sort and add rank columns
    home = home.sort_values(by='Normalized wxG/90', ascending=False)
    away = away.sort_values(by='Normalized wxG/90', ascending=False)
    home['Rank'] = range(1, len(home) + 1)
    away['Rank'] = range(1, len(away) + 1)
    home = home[['Rank'] + [col for col in home.columns if col != 'Rank']]
    away = away[['Rank'] + [col for col in away.columns if col != 'Rank']]
    
    # Calculate league average metrics
    league_avg_xG_home = home['Normalized wxG/90'].mean()
    league_avg_xG_away = away['Normalized wxG/90'].mean()
    league_avg_xG = (league_avg_xG_home + league_avg_xG_away) / 2
    
    return home, away, league_avg_xG

home, away, league_avg_xG = load_data()
teams = sorted(home['Squad'].astype(str).unique().tolist())

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
    return {
        "Home Win Probability": round(home_win_prob * normalization_factor, 2),
        "Draw Probability": round(adjusted_draw_prob * normalization_factor, 2),
        "Away Win Probability": round(away_win_prob * normalization_factor, 2)
    }

def match_outcome_prob(home_df, away_df, home_team, away_team, league_avg_xG, max_goals=10):
    home_xG = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxG/90'].values[0]
    home_xGA = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxGA/90'].values[0]
    away_xG = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxG/90'].values[0]
    away_xGA = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxGA/90'].values[0]
    lambda_A = expected_goals(home_xG, away_xGA, league_avg_xG)
    lambda_B = expected_goals(away_xG, home_xGA, league_avg_xG)
    prob_matrix = poisson_prob_matrix(lambda_A, lambda_B, max_goals)
    home_win_prob = np.sum(np.tril(prob_matrix, -1))
    draw_prob = np.sum(np.diag(prob_matrix))
    away_win_prob = np.sum(np.triu(prob_matrix, 1))
    adjusted_probs = adjust_draw_probability(home_win_prob, draw_prob, away_win_prob)
    return {
        "Home Team": home_team,
        "Away Team": away_team,
        "Expected Goals (Home)": round(lambda_A, 2),
        "Expected Goals (Away)": round(lambda_B, 2),
        **adjusted_probs
    }, lambda_A, lambda_B, prob_matrix

def two_leg_outcome_prob(home_df, away_df, team1, team2, league_avg_xG, max_goals=10):
    first_leg = match_outcome_prob(home_df, away_df, team1, team2, league_avg_xG, max_goals)[0]
    second_leg = match_outcome_prob(home_df, away_df, team2, team1, league_avg_xG, max_goals)[0]
    team1_expected_goals = first_leg["Expected Goals (Home)"] + second_leg["Expected Goals (Away)"]
    team2_expected_goals = first_leg["Expected Goals (Away)"] + second_leg["Expected Goals (Home)"]
    prob_team1_wins = 0
    prob_team2_wins = 0
    prob_draw = 0
    for g1 in range(2 * max_goals + 1):
        for g2 in range(2 * max_goals + 1):
            p1 = stats.poisson.pmf(g1, team1_expected_goals)
            p2 = stats.poisson.pmf(g2, team2_expected_goals)
            joint = p1 * p2
            if g1 > g2:
                prob_team1_wins += joint
            elif g2 > g1:
                prob_team2_wins += joint
            else:
                prob_draw += joint
    prob_team1_wins += prob_draw / 2
    prob_team2_wins += prob_draw / 2
    return {
        "Team1": team1,
        "Team2": team2,
        "Expected Goals (Team1)": round(team1_expected_goals, 2),
        "Expected Goals (Team2)": round(team2_expected_goals, 2),
        "Team1 Win Probability": round(prob_team1_wins, 2),
        "Team2 Win Probability": round(prob_team2_wins, 2)
    }

def champion_function(home_df, away_df, main_team, other_teams, league_avg_xG, max_goals=10):
    overall_prob = 1.0
    for opponent in other_teams:
        match_prob = two_leg_outcome_prob(home_df, away_df, main_team, opponent, league_avg_xG, max_goals)
        overall_prob *= match_prob["Team1 Win Probability"]
    return {
        "Main Team": main_team,
        "Probability of Beating All Opponents": round(overall_prob * 100, 2)  # Already in %
    }

def win_by_margin(home_df, away_df, home_team, away_team, margin, team_side, league_avg_xG, max_goals=15):
    """
    Calculate probability of selected team winning by more than specified margin goals
    
    Parameters:
    - home_team: The home team name
    - away_team: The away team name
    - margin: Minimum margin to win by (must be > 0)
    - team_side: Either 'home' or 'away' to specify which team should win by the margin
    
    Returns:
    - Probability as a float between 0 and 1
    """
    match_results, lambda_home, lambda_away, prob_matrix = match_outcome_prob(
        home_df, away_df, home_team, away_team, league_avg_xG, max_goals
    )
    
    # Calculate probability of winning by margin
    total_prob = 0
    if team_side.lower() == 'home':
        # Sum probabilities where home team (A) scores at least margin more goals than away team (B)
        for i in range(max_goals):
            for j in range(max_goals):
                if i - j > margin:
                    total_prob += prob_matrix[i, j]
    else:  # away team
        # Sum probabilities where away team (B) scores at least margin more goals than home team (A)
        for i in range(max_goals):
            for j in range(max_goals):
                if j - i > margin:
                    total_prob += prob_matrix[i, j]
    
    return {
        "Home Team": home_team,
        "Away Team": away_team,
        "Team to Win by Margin": home_team if team_side.lower() == 'home' else away_team,
        "Margin": margin,
        "Expected Goals (Home)": round(lambda_home, 2),
        "Expected Goals (Away)": round(lambda_away, 2),
        "Probability of Winning by > {} Goal(s)".format(margin): round(total_prob, 4),
        "Percentage Chance": f"{round(total_prob * 100, 2)}%"
    }

# Helper to format probabilities as percentages in a single-row DataTable
def dict_to_percentage_table(result_dict, prob_keys=None):
    df = pd.DataFrame([result_dict])
    if prob_keys is None:
        prob_keys = []
    for k in prob_keys:
        if k in df.columns:
            df[k] = df[k].apply(lambda x: f"{x*100:.2f}%" if x <= 1 else f"{x:.2f}%")
    return dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.to_dict('records'),
        style_table={'width': '100%', 'margin': '0 auto'},
        style_cell={'textAlign': 'center', 'backgroundColor': '#303030', 'color': 'white', 'userSelect': 'none'},
        style_header={'backgroundColor': '#444444', 'color': 'white', 'fontWeight': 'bold', 'userSelect': 'none'}
    )

# ----------------------------
# Dash App Setup (Dark Theme)
# ----------------------------

external_stylesheets = [dbc.themes.DARKLY]

TAB_STYLE = {
    'backgroundColor': '#333333',
    'color': 'white',
    'textAlign': 'center'
}

TAB_SELECTED_STYLE = {
    'backgroundColor': '#444444',
    'color': 'white',
    'textAlign': 'center'
}

# Set Last Refreshed timestamp
last_refreshed = datetime.datetime.now().strftime("%m/%d/%y %H:%M")

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dbc.Container([
    html.H1("YANUS UCL", style={'textAlign': 'center', 'marginTop': 20, 'userSelect': 'none'}),
    dbc.Button("Refresh", id='refresh-button', color='primary', className='mx-auto d-block', style={'marginTop': '10px', 'marginBottom': '10px'}),
    html.P(id='last-refreshed-time', children=f"Last Refreshed: {last_refreshed}", style={'textAlign': 'center', 'userSelect': 'none'}),
    dcc.Tabs(
        id="tabs",
        value='tab-match',
        children=[
            dcc.Tab(label='One Match', value='tab-match', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label='Two Legs', value='tab-two-leg', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label='Champion Likelihood', value='tab-champion', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label='Winning Margin', value='tab-margin', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label='Tables', value='tab-df', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label='Info', value='tab-info', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        ],
        style={
            'marginTop': '20px',
            'justifyContent': 'center',
            'backgroundColor': '#333333'
        }
    ),
    html.Div(id='tabs-content', style={'marginTop': '20px'}),
    dcc.Store(id='store-data'),
], fluid=True, style={'textAlign': 'center'})

@app.callback(
    [Output('store-data', 'data'),
     Output('last-refreshed-time', 'children')],
    [Input('refresh-button', 'n_clicks')],
    prevent_initial_call=True
)
def refresh_data(n_clicks):
    if n_clicks:
        # Reload the data
        home_df, away_df, avg_xG = load_data()
        teams_list = sorted(home_df['Squad'].astype(str).unique().tolist())
        
        # Update the timestamp
        new_time = datetime.datetime.now().strftime("%m/%d/%y %H:%M")
        
        # Store the data in the dcc.Store
        return {'home': home_df.to_dict('records'), 
                'away': away_df.to_dict('records'), 
                'league_avg_xG': avg_xG,
                'teams': teams_list}, f"Last Refreshed: {new_time}"
    
    return dash.no_update, dash.no_update

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
     Input('store-data', 'data')]
)
def render_content(tab, store_data):
    global home, away, league_avg_xG, teams
    
    # If we have refreshed data, update our global variables
    if store_data:
        home = pd.DataFrame.from_records(store_data['home'])
        away = pd.DataFrame.from_records(store_data['away'])
        league_avg_xG = store_data['league_avg_xG']
        teams = store_data['teams']
    
    if tab == 'tab-match':
        return html.Div([
            html.H3("Single Match Prediction", style={'userSelect': 'none'}),
            html.Div([
                html.Label("Home Team", style={'userSelect': 'none'}),
                dcc.Dropdown(
                    id='match-home-team',
                    options=[{'label': team, 'value': team} for team in teams],
                    value=teams[0] if teams else None,
                    style={'color': '#000'}
                ),
            ], style={'margin': '10px auto', 'width': '50%'}),
            html.Div([
                html.Label("Away Team", style={'userSelect': 'none'}),
                dcc.Dropdown(
                    id='match-away-team',
                    options=[{'label': team, 'value': team} for team in teams],
                    value=teams[1] if len(teams) > 1 else None,
                    style={'color': '#000'}
                ),
            ], style={'margin': '10px auto', 'width': '50%'}),
            html.Br(),
            html.Button("Predict", id='match-button', n_clicks=0, style={'margin': '10px'}),
            html.Div(id='match-output', style={'marginTop': '20px'})
        ])
    elif tab == 'tab-two-leg':
        return html.Div([
            html.H3("Two-Leg Tie Prediction", style={'userSelect': 'none'}),
            html.Div([
                html.Label("Team 1 (First Leg Home)", style={'userSelect': 'none'}),
                dcc.Dropdown(
                    id='two-leg-team1',
                    options=[{'label': team, 'value': team} for team in teams],
                    value=teams[0] if teams else None,
                    style={'color': '#000'}
                ),
            ], style={'margin': '10px auto', 'width': '50%'}),
            html.Div([
                html.Label("Team 2 (First Leg Away)", style={'userSelect': 'none'}),
                dcc.Dropdown(
                    id='two-leg-team2',
                    options=[{'label': team, 'value': team} for team in teams],
                    value=teams[1] if len(teams) > 1 else None,
                    style={'color': '#000'}
                ),
            ], style={'margin': '10px auto', 'width': '50%'}),
            html.Br(),
            html.Button("Predict", id='two-leg-button', n_clicks=0, style={'margin': '10px'}),
            html.Div(id='two-leg-output', style={'marginTop': '20px'})
        ])
    elif tab == 'tab-champion':
        return html.Div([
            html.H3("Champion Likelihood", style={'userSelect': 'none'}),
            html.Div([
                html.Label("Main Team", style={'userSelect': 'none'}),
                dcc.Dropdown(
                    id='champion-main-team',
                    options=[{'label': team, 'value': team} for team in teams],
                    value=teams[0] if teams else None,
                    style={'color': '#000'}
                ),
            ], style={'margin': '10px auto', 'width': '50%'}),
            html.Div([
                html.Label("Opponents (Select multiple)", style={'userSelect': 'none'}),
                dcc.Dropdown(
                    id='champion-opponents',
                    options=[{'label': team, 'value': team} for team in teams],
                    multi=True,
                    value=teams[1:5] if len(teams) > 5 else teams[1:] if len(teams) > 1 else [],
                    style={'color': '#000'}
                ),
            ], style={'margin': '10px auto', 'width': '50%'}),
            html.Br(),
            html.Button("Predict", id='champion-button', n_clicks=0, style={'margin': '10px'}),
            html.Div(id='champion-output', style={'marginTop': '20px'})
        ])
    elif tab == 'tab-margin':
        return html.Div([
            html.H3("Winning Margin Prediction", style={'userSelect': 'none'}),
            html.Div([
                html.Label("Home Team", style={'userSelect': 'none'}),
                dcc.Dropdown(
                    id='margin-home-team',
                    options=[{'label': team, 'value': team} for team in teams],
                    value=teams[0] if teams else None,
                    style={'color': '#000'}
                ),
            ], style={'margin': '10px auto', 'width': '50%'}),
            html.Div([
                html.Label("Away Team", style={'userSelect': 'none'}),
                dcc.Dropdown(
                    id='margin-away-team',
                    options=[{'label': team, 'value': team} for team in teams],
                    value=teams[1] if len(teams) > 1 else None,
                    style={'color': '#000'}
                ),
            ], style={'margin': '10px auto', 'width': '50%'}),
            html.Div([
                html.Label("Team to win by margin:", style={'userSelect': 'none'}),
                dcc.RadioItems(
                    id='margin-team-side',
                    options=[
                        {'label': 'Home Team', 'value': 'home'},
                        {'label': 'Away Team', 'value': 'away'}
                    ],
                    value='home',
                    style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'marginTop': '10px'},
                    inputStyle={'marginRight': '5px'}
                ),
            ], style={'margin': '10px auto', 'width': '50%'}),
            html.Div([
                html.Label("To win by more than X goals:", style={'userSelect': 'none'}),
                dcc.Input(
                    id='margin-value',
                    type='number',
                    min=0,
                    max=5,
                    step=1,
                    value=1,
                    style={'color': '#000', 'marginLeft': '10px', 'width': '80px'}
                ),
            ], style={'margin': '10px auto', 'width': '50%', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),
            html.Br(),
            html.Button("Calculate Probability", id='margin-button', n_clicks=0, style={'margin': '10px'}),
            html.Div(id='margin-output', style={'marginTop': '20px'})
        ])
    elif tab == 'tab-df':
        return html.Div([
            dcc.Tabs(id='df-tabs', value='tab-home-df', children=[
                dcc.Tab(label='Home Data', value='tab-home-df', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='Away Data', value='tab-away-df', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE)
            ], style={'marginTop': '20px', 'justifyContent': 'center'}),
            html.Div(id='df-tabs-content', style={'marginTop': '20px'})
        ])
    elif tab == 'tab-info':
        return html.Div([
            html.H3("Methodology", style={'userSelect': 'none'}),
            html.Ul([
                html.Li("Splits Home and away form and solve for a weighted xG and xGA using 70% xG and 30% actual goals.", style={'userSelect': 'none'}),
                html.Li("Uses Poisson distributions to solve for matrix of probabilities each team has against each other.", style={'userSelect': 'none'}),
                html.Li("Champion prediction uses 2 legged results as i believe this is a fair representation to model the final, which is neither home nor away (neutral).", style={'userSelect': 'none'})
            ], style={'textAlign': 'left', 'maxWidth': '800px', 'margin': '0 auto', 'userSelect': 'none'}),
            html.H3("Limitations", style={'marginTop': '30px', 'userSelect': 'none'}),
            html.Ul([
                html.Li("Sample size is very small as it is only league phase of this year and already played games in play-ins.", style={'userSelect': 'none'}),
                html.Li("Does not account for where the teams come from since this logic deems all teams are equal since they are 'in the champions league.'", style={'userSelect': 'none'}),
                html.Li("Assumes only performance data from this year in the champions league.", style={'userSelect': 'none'})
            ], style={'textAlign': 'left', 'maxWidth': '800px', 'margin': '0 auto', 'userSelect': 'none'}),
            html.H3("Next Steps", style={'marginTop': '30px', 'userSelect': 'none'}),
            html.Ul([
                html.Li("Combine with big 5 league model.", style={'userSelect': 'none'}),
                html.Li("Create a weighted wxG where i can toggle how much a team should be using its UCL or league wxG. (ucl wxG will not be adjusted for any conversion rate)", style={'userSelect': 'none'}),
                html.Li("For teams who are not in top 5, assume their UCL wxG since there is no other way to get their data.", style={'userSelect': 'none'}),
                html.Li("Do exact same thing for Europa League.", style={'userSelect': 'none'})
            ], style={'textAlign': 'left', 'maxWidth': '800px', 'margin': '0 auto', 'userSelect': 'none'}),
        ])

@app.callback(Output('df-tabs-content', 'children'),
              Input('df-tabs', 'value'))
def render_df_content(tab):
    if tab == 'tab-home-df':
        return dash_table.DataTable(
            id='home-table',
            columns=[{"name": i, "id": i} for i in home.columns],
            data=home.round(2).to_dict('records'),
            filter_action="native",
            sort_action="native",
            page_size=len(home),
            style_table={'overflowX': 'auto', 'width': '90%', 'margin': '0 auto'},
            style_cell={'textAlign': 'center', 'backgroundColor': '#303030', 'color': 'white', 'userSelect': 'none'},
            style_header={'backgroundColor': '#444444', 'color': 'white', 'fontWeight': 'bold', 'userSelect': 'none'}
        )
    elif tab == 'tab-away-df':
        return dash_table.DataTable(
            id='away-table',
            columns=[{"name": i, "id": i} for i in away.columns],
            data=away.round(2).to_dict('records'),
            filter_action="native",
            sort_action="native",
            page_size=len(away),
            style_table={'overflowX': 'auto', 'width': '90%', 'margin': '0 auto'},
            style_cell={'textAlign': 'center', 'backgroundColor': '#303030', 'color': 'white', 'userSelect': 'none'},
            style_header={'backgroundColor': '#444444', 'color': 'white', 'fontWeight': 'bold', 'userSelect': 'none'}
        )

@app.callback(Output('match-output', 'children'),
              Input('match-button', 'n_clicks'),
              State('match-home-team', 'value'),
              State('match-away-team', 'value'))
def update_match(n_clicks, home_team, away_team):
    if n_clicks and home_team and away_team:
        result = match_outcome_prob(home, away, home_team, away_team, league_avg_xG)[0]
        return dict_to_percentage_table(result, prob_keys=["Home Win Probability", "Draw Probability", "Away Win Probability"])
    return ""

@app.callback(Output('two-leg-output', 'children'),
              Input('two-leg-button', 'n_clicks'),
              State('two-leg-team1', 'value'),
              State('two-leg-team2', 'value'))
def update_two_leg(n_clicks, team1, team2):
    if n_clicks and team1 and team2:
        result = two_leg_outcome_prob(home, away, team1, team2, league_avg_xG)
        return dict_to_percentage_table(result, prob_keys=["Team1 Win Probability", "Team2 Win Probability"])
    return ""

@app.callback(Output('champion-output', 'children'),
              Input('champion-button', 'n_clicks'),
              State('champion-main-team', 'value'),
              State('champion-opponents', 'value'))
def update_champion(n_clicks, main_team, opponents):
    if n_clicks and main_team and opponents:
        result = champion_function(home, away, main_team, opponents, league_avg_xG)
        return dict_to_percentage_table(result)
    return ""

@app.callback(Output('margin-output', 'children'),
              Input('margin-button', 'n_clicks'),
              State('margin-home-team', 'value'),
              State('margin-away-team', 'value'),
              State('margin-team-side', 'value'),
              State('margin-value', 'value'))
def update_margin(n_clicks, home_team, away_team, team_side, margin_value):
    if n_clicks and home_team and away_team and team_side and margin_value is not None:
        result = win_by_margin(home, away, home_team, away_team, margin_value, team_side, league_avg_xG)
        
        # Get the probability matrix from match_outcome_prob
        _, lambda_home, lambda_away, prob_matrix = match_outcome_prob(
            home, away, home_team, away_team, league_avg_xG, max_goals=10
        )
        
        # Format the main result
        main_result = dict_to_percentage_table(result)
        
        # Create a pretty display of the probability matrix
        matrix_df = pd.DataFrame(prob_matrix)
        matrix_df.index.name = f"{home_team} Goals"
        matrix_df.columns.name = f"{away_team} Goals"
        
        # Convert to percentage format for better readability
        matrix_percentage = matrix_df.applymap(lambda x: f"{x*100:.2f}%" if x >= 0.0001 else "<0.01%")
        
        # Highlight the cells that contribute to the winning margin calculation
        if team_side.lower() == 'home':
            # Highlight cells where home team (rows) scores margin more than away team (columns)
            highlighted_cells = [
                {
                    'if': {
                        'row_index': i,
                        'column_id': str(j)
                    },
                    'backgroundColor': '#006400',  # Dark green
                    'color': 'white'
                }
                for i in range(10)  # Adjust based on max_goals
                for j in range(10)
                if i - j > margin_value
            ]
        else:  # away team
            # Highlight cells where away team (columns) scores margin more than home team (rows)
            highlighted_cells = [
                {
                    'if': {
                        'row_index': i,
                        'column_id': str(j)
                    },
                    'backgroundColor': '#006400',  # Dark green
                    'color': 'white'
                }
                for i in range(10)  # Adjust based on max_goals
                for j in range(10)
                if j - i > margin_value
            ]
        
        # Create data table for the matrix
        matrix_table = dash_table.DataTable(
            id='probability-matrix',
            columns=[{"name": str(i), "id": str(i)} for i in range(10)],  # Adjust based on max_goals
            data=matrix_percentage.reset_index().to_dict('records'),
            style_table={'width': '95%', 'margin': '20px auto', 'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'backgroundColor': '#303030', 'color': 'white', 'minWidth': '60px'},
            style_header={
                'backgroundColor': '#444444',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=highlighted_cells,
            tooltip_header={
                str(i): f'{away_team} scores {i} goals' for i in range(10)
            },
            tooltip_data=[
                {
                    str(j): {'value': f'{home_team} scores {i} goals, {away_team} scores {j} goals: {prob_matrix[i,j]*100:.4f}%'}
                    for j in range(10)
                }
                for i in range(10)
            ],
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
            }
        )
        
        # Add explanatory text
        explanation = html.Div([
            html.H4(f"Poisson Probability Matrix", style={'marginTop': '20px', 'userSelect': 'none'}),
            html.P(f"This matrix shows the probability of each possible score between {home_team} (rows) and {away_team} (columns).", 
                   style={'userSelect': 'none'}),
            html.P(f"Highlighted cells show outcomes where {home_team if team_side.lower() == 'home' else away_team} wins by more than {margin_value} goal(s).", 
                   style={'userSelect': 'none'}),
            html.P(f"Expected goals: {home_team} = {lambda_home:.2f}, {away_team} = {lambda_away:.2f}", 
                   style={'userSelect': 'none', 'fontWeight': 'bold'})
        ])
        
        # Display additional stats
        home_win_prob = np.sum(np.tril(prob_matrix, -1)) * 100
        away_win_prob = np.sum(np.triu(prob_matrix, 1)) * 100
        draw_prob = np.sum(np.diag(prob_matrix)) * 100
        
        stats_div = html.Div([
            html.H4("Match Outcome Probabilities", style={'marginTop': '20px', 'userSelect': 'none'}),
            html.Ul([
                html.Li(f"{home_team} Win: {home_win_prob:.2f}%", style={'userSelect': 'none'}),
                html.Li(f"Draw: {draw_prob:.2f}%", style={'userSelect': 'none'}),
                html.Li(f"{away_team} Win: {away_win_prob:.2f}%", style={'userSelect': 'none'}),
            ], style={'textAlign': 'left', 'display': 'inline-block'})
        ])
        
        # Return all elements
        return html.Div([
            main_result,
            explanation,
            matrix_table,
            stats_div
        ])
    
    return ""

if __name__ == '__main__':
    app.run_server(debug=True, port=8089)
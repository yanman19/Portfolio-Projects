#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd

url = "https://fbref.com/en/comps/9/Premier-League-Stats"
headers = {"User-Agent": "Mozilla/5.0"}
html = requests.get(url, headers=headers).text
dfs = pd.read_html(html)

df = dfs[1]
df.columns = [
    "Rk","Squad",
    "Home_MP", "Home_W", "Home_D", "Home_L", "Home_GF", "Home_GA", "Home_GD", "Home_Pts",'Home_Pts/MP', "Home_xG", "Home_xGA", "Home_xGD", "Home_xGD_per_90",
    "Away_MP", "Away_W", "Away_D", "Away_L", "Away_GF", "Away_GA", "Away_GD", "Away_Pts",'Away_Pts/MP', "Away_xG", "Away_xGA", "Away_xGD", "Away_xGD_per_90"]


# In[2]:


dfs


# In[3]:


df.to_clipboard()


# In[4]:


import requests
import pandas as pd

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




# In[5]:


## la liga

url = "https://fbref.com/en/comps/12/La-Liga-Stats"
headers = {"User-Agent": "Mozilla/5.0"}
html = requests.get(url, headers=headers).text
dfs = pd.read_html(html)

df = dfs[1]
df.columns = [
    "Rk","Squad",
    "Home_MP", "Home_W", "Home_D", "Home_L", "Home_GF", "Home_GA", "Home_GD", "Home_Pts",'Home_Pts/MP', "Home_xG", "Home_xGA", "Home_xGD", "Home_xGD_per_90",
    "Away_MP", "Away_W", "Away_D", "Away_L", "Away_GF", "Away_GA", "Away_GD", "Away_Pts",'Away_Pts/MP', "Away_xG", "Away_xGA", "Away_xGD", "Away_xGD_per_90"]



df['Country'] = 'Spain'

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

laliga_home = home_df
laliga_away = away_df



# In[6]:


## Serie A

url = "https://fbref.com/en/comps/11/Serie-A-Stats"
headers = {"User-Agent": "Mozilla/5.0"}
html = requests.get(url, headers=headers).text
dfs = pd.read_html(html)

df = dfs[1]
df.columns = [
    "Rk","Squad",
    "Home_MP", "Home_W", "Home_D", "Home_L", "Home_GF", "Home_GA", "Home_GD", "Home_Pts",'Home_Pts/MP', "Home_xG", "Home_xGA", "Home_xGD", "Home_xGD_per_90",
    "Away_MP", "Away_W", "Away_D", "Away_L", "Away_GF", "Away_GA", "Away_GD", "Away_Pts",'Away_Pts/MP', "Away_xG", "Away_xGA", "Away_xGD", "Away_xGD_per_90"]

df['Country'] = 'Italy'

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

seriea_home = home_df
seriea_away = away_df


# In[7]:


## Bundesliga

url = "https://fbref.com/en/comps/20/Bundesliga-Stats"
headers = {"User-Agent": "Mozilla/5.0"}
html = requests.get(url, headers=headers).text
dfs = pd.read_html(html)

df = dfs[1]
df.columns = [
    "Rk","Squad",
    "Home_MP", "Home_W", "Home_D", "Home_L", "Home_GF", "Home_GA", "Home_GD", "Home_Pts",'Home_Pts/MP', "Home_xG", "Home_xGA", "Home_xGD", "Home_xGD_per_90",
    "Away_MP", "Away_W", "Away_D", "Away_L", "Away_GF", "Away_GA", "Away_GD", "Away_Pts",'Away_Pts/MP', "Away_xG", "Away_xGA", "Away_xGD", "Away_xGD_per_90"]



df['Country'] = 'Germany'

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

bundesliga_home = home_df
bundesliga_away = away_df


# In[8]:


## ligue 1

url = "https://fbref.com/en/comps/13/Ligue-1-Stats"
headers = {"User-Agent": "Mozilla/5.0"}
html = requests.get(url, headers=headers).text
dfs = pd.read_html(html)

df = dfs[1]
df.columns = [
    "Rk","Squad",
    "Home_MP", "Home_W", "Home_D", "Home_L", "Home_GF", "Home_GA", "Home_GD", "Home_Pts",'Home_Pts/MP', "Home_xG", "Home_xGA", "Home_xGD", "Home_xGD_per_90",
    "Away_MP", "Away_W", "Away_D", "Away_L", "Away_GF", "Away_GA", "Away_GD", "Away_Pts",'Away_Pts/MP', "Away_xG", "Away_xGA", "Away_xGD", "Away_xGD_per_90"]



df['Country'] = 'France'

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

ligue1_home = home_df
ligue1_away = away_df


# In[10]:


homes = [pl_home,laliga_home,seriea_home,bundesliga_home,ligue1_home]
aways = [pl_away,laliga_away,seriea_away,bundesliga_away,ligue1_away]


home = pd.concat(homes,ignore_index=True)
away = pd.concat(aways,ignore_index=True)


# In[11]:


import pandas as pd



# League scores and conversion ratios
league_scores = {
    "England": 106.624,
    "Italy": 93.043,
    "Spain": 88.596,
    "Germany": 83.331,
    "France": 69.379,
    "Portugal": 61.366,
    "Netherlands":65.15
}

converted_league_scores = {league: score / league_scores["England"] for league, score in league_scores.items()}

# Map conversion ratios to the DataFrame
home['Conversion Ratio'] = home['Country'].map(converted_league_scores).round(2)
away['Conversion Ratio'] = away['Country'].map(converted_league_scores).round(2)

home['wxG'] = home['xG'] * 0.7 + home['GF'] * 0.3
home['wxGA'] = home['xGA'] * 0.7 + home['GA'] * 0.3 

away['wxG'] = away['xG'] * 0.7 + away['GF'] * 0.3
away['wxGA'] = away['xGA'] * 0.7 + away['GA'] * 0.3 

home['Normalized wxG/90'] = ((home['Conversion Ratio'] * home['wxG']) / home['MP']).round(2)
away['Normalized wxG/90'] = ((away['Conversion Ratio'] * away['wxG']) / away['MP']).round(2)

home['Normalized wxGA/90'] = ((home['Conversion Ratio'] * home['wxGA']) / home['MP']).round(2)
away['Normalized wxGA/90'] = ((away['Conversion Ratio'] * away['wxGA']) / away['MP']).round(2)


home = home.sort_values(by='Normalized wxG/90',ascending=False)
away = away.sort_values(by='Normalized wxG/90',ascending=False)



# In[33]:


import numpy as np
import pandas as pd
import scipy.stats as stats

def expected_goals(xG_team, xGA_opp, league_avg_xG):
    return (xG_team / league_avg_xG) * (xGA_opp / league_avg_xG) * league_avg_xG

def poisson_prob_matrix(lambda_A, lambda_B, max_goals=10):
    prob_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            prob_matrix[i, j] = stats.poisson.pmf(i, lambda_A) * stats.poisson.pmf(j, lambda_B)
    return prob_matrix

def adjust_draw_probability(home_win_prob, draw_prob, away_win_prob):
    """
    Adjust draw probability to be the second-highest probability after the favored team's win probability.
    """
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

def match_outcome_prob(home_df, away_df, home_team, away_team, league_avg_xG, max_goals=10):
    home_xG = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxG/90'].values[0]
    home_xGA = home_df.loc[home_df['Squad'] == home_team, 'Normalized wxGA/90'].values[0]
    away_xG = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxG/90'].values[0]
    away_xGA = away_df.loc[away_df['Squad'] == away_team, 'Normalized wxGA/90'].values[0]

    lambda_A = expected_goals(home_xG, away_xGA, league_avg_xG)
    lambda_B = expected_goals(away_xG, home_xGA, league_avg_xG)

    prob_matrix = poisson_prob_matrix(lambda_A, lambda_B, max_goals)

    home_win_prob = np.sum(np.tril(prob_matrix, -1))  # Home team wins
    draw_prob = np.sum(np.diag(prob_matrix))  # Draw
    away_win_prob = np.sum(np.triu(prob_matrix, 1))  # Away team wins

    adjusted_probs = adjust_draw_probability(home_win_prob, draw_prob, away_win_prob)

    return {
        "Home Team": home_team,
        "Away Team": away_team,
        "Expected Goals (Home)": lambda_A.round(2),
        "Expected Goals (Away)": lambda_B.round(2),
        **adjusted_probs
    }
# Example usage with separate home and away DataFrames


league_avg_xG_home = home['Normalized wxG/90'].mean()
league_avg_xG_away = away['Normalized wxG/90'].mean()
league_avg_xG = (league_avg_xG_home + league_avg_xG_away) / 2  # Overall league average xG per game

probabilities = match_outcome_prob(home, away, 'Juventus', "Inter", league_avg_xG)
print(probabilities)


# In[13]:


import scipy.stats as stats

def two_leg_outcome_prob(home_df, away_df, team1, team2, league_avg_xG, max_goals=10):
    # First leg: team1 at home, team2 away
    first_leg_probs = match_outcome_prob(home_df, away_df, team1, team2, league_avg_xG, max_goals)
    
    # Second leg: team2 at home, team1 away
    second_leg_probs = match_outcome_prob(home_df, away_df, team2, team1, league_avg_xG, max_goals)
    
    # Expected goals per team over two legs
    team1_expected_goals = first_leg_probs["Expected Goals (Home)"] + second_leg_probs["Expected Goals (Away)"]
    team2_expected_goals = first_leg_probs["Expected Goals (Away)"] + second_leg_probs["Expected Goals (Home)"]
    
    # Compute probabilities using Poisson distribution
    prob_team1_wins = 0
    prob_team2_wins = 0
    prob_draw = 0
    
    for goals_team1 in range(2 * max_goals + 1):
        for goals_team2 in range(2 * max_goals + 1):
            prob_team1 = stats.poisson.pmf(goals_team1, team1_expected_goals)
            prob_team2 = stats.poisson.pmf(goals_team2, team2_expected_goals)
            joint_prob = prob_team1 * prob_team2
            
            if goals_team1 > goals_team2:
                prob_team1_wins += joint_prob
            elif goals_team2 > goals_team1:
                prob_team2_wins += joint_prob
            else:
                prob_draw += joint_prob
    
    # Resolve draws: Assume equal probability of winning if tied (can be adjusted for real-world penalty stats)
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

# Example usage
two_leg_probs = two_leg_outcome_prob(home, away, "Crystal Palace", "Everton", league_avg_xG)
print(two_leg_probs)


# In[31]:


def champion_function(home_df, away_df, main_team, other_teams, league_avg_xG, max_goals=10):
    overall_prob = 1.0
    
    for opponent in other_teams:
        match_prob = two_leg_outcome_prob(home_df, away_df, main_team, opponent, league_avg_xG, max_goals)
        overall_prob *= match_prob["Team1 Win Probability"] 
    
    return {
        "Main Team": main_team,
        "Probability of Beating All Opponents": round(overall_prob, 2) * 100
    }

champion_prob = champion_function(home, away, "Manchester Utd", ["Real Sociedad", "Eint Frankfurt",'Lazio','Athletic Club'], league_avg_xG)
print(champion_prob)


# In[27]:


home.head(50)


# In[ ]:


#use past 10 home games and past 10 away games

#make website where I can put any two teams in.

#allow me to toggle uefa coefficents

#run backtest 

#systematically pull in upcoming fixtures and display percentages

#perhaps incorp a tie no bet


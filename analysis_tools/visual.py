"""Module containing functions to generate football data anlytics visuals.

Functions
---------


"""


import json

import pandas as pd
import requests
from highlight_text import fig_text
from mplsoccer import Pitch


def plot_match_shotmap(ax):

    match_id =3609980
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')
    data = json.loads(response.content)

    matchId = data['general']['matchId']
    homeTeam = data['general']['homeTeam']
    awayTeam = data['general']['awayTeam']

    shot_data = data['content']['shotmap']['shots']
    df_shot = pd.DataFrame(shot_data)
    df_shot['matchId'] = matchId

    team_dict_name = {homeTeam['id']: homeTeam['name'], awayTeam['id']: awayTeam['name']}
    df_shot['teamName'] = df_shot['teamId'].map(team_dict_name)

    team_dict = {homeTeam['id']: 'Home', awayTeam['id']: 'Away'}
    df_shot['Venue'] = df_shot['teamId'].map(team_dict)

    h_data = df_shot[df_shot['Venue'] == 'Home']
    a_data = df_shot[df_shot['Venue'] == 'Away']

    Home_shots = h_data[h_data['eventType'] == 'Goal']
    Home_goals = h_data[h_data['eventType'] != 'Goal']
    Away_shots = a_data[a_data['eventType'] == 'Goal']
    Away_goals = a_data[a_data["eventType"] != 'Goal']

    #fig = plt.figure(figsize=(6, 4), dpi=900)
    #ax = plt.subplot(111)
    #fig.set_facecolor("#2E2929")

    pitch = Pitch(
        pitch_color='#2E2929',
        pitch_type="uefa",
        half=False,
        goal_type='box',
        line_color='#1A1D1A'
    )
    pitch.draw(ax=ax)

    ax.scatter(Home_shots.x,
               Home_shots.y,
               c="#A04043",
               s=Home_shots.expectedGoals * 90
               )

    ax.scatter(Home_goals.x,
               Home_goals.y,
               c="#EFE9F4",
               s=Home_goals.expectedGoals * 90,
               marker='*'
               )

    ax.scatter(105 - Away_shots.x,
               70 - Away_shots.y,
               c="#C00808",
               s=Away_shots.expectedGoals * 90
               )

    ax.scatter(105 - Away_goals.x,
               70 - Away_goals.y,
               c="#EFE9F4",
               s=Away_goals.expectedGoals * 90,
               marker='*'
               )

    homeTeam_Name = homeTeam['name']
    awayTeam_Name = awayTeam['name']
    title = f"{homeTeam_Name} vs {awayTeam_Name} Shot Map"
    fig_text(0.5, 0.97, title, ha='center', va='center', color="#EFE9F4")

    return ax

#%%

#%%

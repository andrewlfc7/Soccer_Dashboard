import json

import matplotlib.pyplot as plt
import pandas as pd
import requests
from mplsoccer import Pitch, VerticalPitch


def plot_match_shotmap(ax, match_id:int):
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

    h_data = df_shot[df_shot['Venue']=='Home']
    a_data = df_shot[df_shot['Venue']=='Away']

    # Set team color for both home and away team
    home_color = h_data.teamColor.iloc[0]
    away_color = a_data.teamColor.iloc[0]

    Home_shots = h_data[h_data['eventType'] != 'Goal']
    Home_goals = h_data[h_data['eventType'] == 'Goal']
    Away_shots = a_data[a_data['eventType'] != 'Goal']
    Away_goals = a_data[a_data["eventType"] == 'Goal']

    #fig = plt.figure(figsize=(6, 4), dpi=900)
    #ax = plt.subplot(111)
    #fig.set_facecolor("#201D1D")

    pitch = Pitch(
        linewidth=2.5,
        pitch_color='#201D1D',
        pitch_type="uefa",
        half=False,
        goal_type='box',
        line_color='black'
    )
    pitch.draw(ax=ax)



    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0] - 0.5, pos_y[-1] + 0.5], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0] - 0.5, pos_x[-1] + 0.5], [y, y], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)


    ax.scatter(105 - Home_shots.x,
               70 - Home_shots.y,
               c=home_color,
               s=Home_shots.expectedGoals * 90 * 6,
               marker = 'o',
               alpha = .8,
               label = 'Shots'
               )

    ax.scatter(105 - Home_goals.x,
               70 - Home_goals.y,
               c="#018b95",
               s=Home_goals.expectedGoals * 90 * 6,
               marker='o',
               alpha = .8,
               label = 'Goal'
               )

    ax.scatter( Away_shots.x,
                Away_shots.y,
                c=away_color,
                s=Away_shots.expectedGoals * 90 * 6,
                alpha = .8,
                label = 'Shot',
                marker = 'o'
                )

    ax.scatter( Away_goals.x,
                Away_goals.y,
                c="#018b95",
                s=Away_goals.expectedGoals * 90 * 6,
                marker='o',
                alpha= .8,


                )

    for eg in [.10,.25,.50]:
        ax.scatter([], [], c='k', alpha=0.3, s=eg * 90 * 6,
                   label=str(eg)+'xG')

    legend = ax.legend(scatterpoints=1, markerscale=.4, labelcolor='white',columnspacing=.02, labelspacing=.02, ncol=6,
                       loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    legend.get_frame().set_alpha(0.6)


    return ax






def plot_match_xgflow(ax, match_id:int):
    response = requests.get(
        f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')

    data = json.loads(response.content)
    data

    team_logos = data['header']['teams']
    team_logos = pd.DataFrame(team_logos)
    team_logos

    homeTeam = data['general']['homeTeam']
    awayTeam = data['general']['awayTeam']
    #homeTeam = pd.DataFrame(homeTeam,index=[0])
    #awayTeam = pd.DataFrame(awayTeam,index=[0])


    shot_data = data['content']['shotmap']['shots']
    df_shot = pd.DataFrame(shot_data)
    df_shot['min'] = df_shot['min'].astype(int)
    df_shot['xG'] = df_shot['expectedGoals'].astype(float)
    xg_flow = df_shot[['teamId', 'situation', 'eventType', 'expectedGoals', 'playerName', 'min', 'teamColor', 'isOwnGoal']]
    xg_flow
    team_dict_name = {homeTeam['id']: homeTeam['name'], awayTeam['id']: awayTeam['name']}
    xg_flow['teamName'] = xg_flow['teamId'].map(team_dict_name)
    xg_flow
    # Create a dictionary mapping team IDs to team names
    team_dict = {homeTeam['id']: 'Home', awayTeam['id']: 'Away'}

    xg_flow['Venue'] = xg_flow['teamId'].map(team_dict)
    xg_flow
    h_data = xg_flow[xg_flow['Venue'] == 'Home']
    h_data
    a_data = xg_flow[xg_flow['Venue'] == 'Away']

    home_color = h_data.teamColor.iloc[0]
    away_color = a_data.teamColor.iloc[0]


    def nums_cumulative_sum(nums_list):
        return [sum(nums_list[:i + 1]) for i in range(len(nums_list))]


    a_cumulative = nums_cumulative_sum(a_data['expectedGoals'])
    h_cumulative = nums_cumulative_sum(h_data['expectedGoals'])

    #this is used to find the total xG. It just creates a new variable from the last item in the cumulative list
    alast = round(a_cumulative[-1], 2)
    hlast = round(h_cumulative[-1], 2)

    h_data['cum_xg']= h_cumulative
    a_data['cum_xg'] = a_cumulative

    plt.xticks([0,15,30,45,60,75,90])

    plt.step(0 + h_data['min'], h_cumulative, color=home_color, linestyle='dashdot', label=homeTeam['name'])

    plt.step(0 + a_data['min'], a_cumulative, color=away_color, linestyle='solid', label=awayTeam['name'])

    plt.xticks([], [])
    plt.yticks([], [])



    plt.axvline(45, c='#018b95')
    #plt.xlabel('Minutes')
    #plt.ylabel('Cumulative Expected Goals')
    #plt.legend()
    #plt.xlabel('Minute',fontsize=16)
    #plt.ylabel('xG',fontsize=16)

    return ax

#%%

#%%

def plot_player_shotmap(ax, match_id: int, player_name):
    response = requests.get(
        f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')
    data = json.loads(response.content)

    shot_data = data['content']['shotmap']['shots']
    df_shot = pd.DataFrame(shot_data)

    df_shot = df_shot[df_shot['playerName'] == player_name]
    shots = df_shot[df_shot['eventType'] != 'Goal']

    df_goal = df_shot[df_shot['eventType'] == 'Goal']

    pitch = pitch = VerticalPitch(
        pitch_color='#201D1D',
        half=True,
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='uefa',
        goal_type='box',
        pad_top=10,
        line_color='black')
    pitch.draw(ax=ax)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    # Remember that we need to invert the axis!!
    for x in pos_x[1:-1]:
        ax.plot([pos_y[0], pos_y[-1]], [x, x], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([y, y], [pos_x[0], pos_x[-1]], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)

    ax.scatter(shots.y,
               shots.x,
               c='#7371FC',
               s=shots.expectedGoals * 90 * 4,
               alpha=.8,
               label = 'Shot'
               )

    ax.scatter(df_goal.y,
               df_goal.x,
               c="#5EB39E",
               s=df_goal.expectedGoals * 90 * 4,
               marker='o',
               alpha=.8,
               label = 'Goal'
               )

    for eg in [.10,.25,.50]:
        ax.scatter([], [], c='k', alpha=0.3, s=eg * 90 * 6,
                   label=str(eg)+'xG')

    legend = ax.legend(scatterpoints=1, markerscale=.4, columnspacing=.02, labelcolor='white',labelspacing=.02, ncol=6,
                       loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    legend.get_frame().set_alpha(0.6)


    return ax


#%%

#%%

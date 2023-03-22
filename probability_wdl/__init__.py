#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Modules Imported
## ----------------------------
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer.pitch import Pitch
## ----------------------------

def check():
    print(random.__file__)



def playerXG(shooters, team_name='', nonpen=False):
    """
    Creates a bar graph figure for the xG distribution by player

    Parameters
    ----------
    shooters : A pandas series of the players that took shots and their total xG
        - Created by taking the dataframe and home_df.groupby(['player'])['xG'].sum()

    team_name : string
        - Name of team for distribution

    nonpen : boolean
        - Boolean index for whether or not you're looking at nonpenalty xg
        - Defaults to false

    Returns
    -------
    distribution : figure
        - The bar graph distribution of players that took shoots grouped by total xG

    """

    x = shooters.index
    y = shooters

    x_pos = [i for i, _ in enumerate(x)]

    fig, ax = plt.subplots(figsize=(10,6))
    fig.subplots_adjust(bottom=0.25)

    ax.bar(x_pos, y, color='#1f77b4')
    plt.ylabel('xG', fontSize = 16)

    if nonpen:
        title = f'{team_name} Non-Penalty xG Distribution'
    else:
        title = f'{team_name} xG Distribution'
    plt.title(title, fontsize=20)

    plt.xticks(x_pos, x, fontsize=14)
    plt.yticks(fontsize=14)

    # rotate axis labels
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show()


def shotMapTeam(shots, mask_goal, home_name, away_name, home=True):
    """
    Creates a team shot map (similar to caley style) for a single team on half pitch

    Parameters
    ----------
    shots : A pandas dataframe
        - Subset of team dataframe
        - contains ['id', 'minute', 'X', 'Y', 'player', 'shotType', 'situation', 'result', 'xG', 'pos_x', 'pos_y']
        - pos_x and pos_y are scaled to the 120x80 statsbomb sized pitch

    mask_goal : Pandas index mask
        - Mask for all goals
        - Lets us separate points where the shot is scored

    team_name : string
        - Name of home/away team

    home : boolean
        - Defaults to true
        - True if looking at home team's shots.
        - Responsible for changing draw color options and title text

    Returns
    -------
    shotmap : figure
        - Pitch shot map for a single team

    """

    max_marker_size = 500

    ## Plotting options for single team or both teams
    orientation = 'vertical'
    view = 'half'

    if home:
        color = '#1f77b4'
        team_name = home_name
        opp_name = away_name
    else:
        color = '#ff7f0e'
        team_name = away_name
        opp_name = home_name


    fig, ax = plt.subplots(figsize=(10,6))
    fig.set_facecolor('#3d4849')
    ax.patch.set_facecolor('#3d4849')

    pitch = Pitch(pitch_type='statsbomb', orientation=orientation,
                  pitch_color='#22312b', line_color='#c7d5cc', figsize=(8,5.5),
                  constrained_layout=False, tight_layout=True, view=view,
                  pad_top=2)

    pitch.scatter(shots[~mask_goal].pos_x, shots[~mask_goal].pos_y,
                  s=shots[~mask_goal].xG*max_marker_size, edgecolors='white', c=color,
                  marker='s', zorder=1, label='Shot', ax=ax, alpha=0.8)

    pitch.scatter(shots[mask_goal].pos_x, shots[mask_goal].pos_y,
                  edgecolors='black', c='white', s=shots[mask_goal].xG*max_marker_size,
                  zorder=2, marker='s', alpha=0.95, label='Goal', ax=ax)

    pitch.draw(ax=ax)

    # Flip so goal is on bottom
    plt.gca().invert_yaxis()

    # Set the title
    ax.set_title(f'{team_name} Shots \n vs {opp_name}', fontsize=20, color='w')


def shotMapMatch(shots_home, shots_away, mask_goal_h, mask_goal_a, home_name, away_name, home_score, away_score):
    """
    Creates a combined shot map (similar to caley style) for a both teams on a full, horizontal pitch

    Parameters
    ----------
    shots_XXXX : A pandas dataframe
        - Subset of team dataframe for home and away
        - contains ['id', 'minute', 'X', 'Y', 'player', 'shotType', 'situation', 'result', 'xG', 'pos_x', 'pos_y']
        - pos_x and pos_y are scaled to the 120x80 statsbomb sized pitch

    mask_goal_X : Pandas index mask
        - Mask for all goals separated home and away
        - Lets us separate points where the shot is scored

    team_name : string
        - Name of home/away team

    score : int
        - Actual team score
        - used for display options

    Returns
    -------
    shotmap : figure
        - Pitch shot map for a both teams

    """

    max_marker_size = 500

    home_xg = round(shots_home['xG'].sum(),1)
    away_xg = round(shots_away['xG'].sum(),1)

    fig, ax = plt.subplots(figsize=(10,6))
    fig.set_facecolor('#3d4849')
    ax.patch.set_facecolor('#3d4849')

    pitch = Pitch(pitch_type='statsbomb', orientation='horizontal',
                  pitch_color='#22312b', line_color='#c7d5cc', figsize=(10,6),
                  constrained_layout=False, tight_layout=True, view='full',
                  pad_top=2)

    # Home
    pitch.scatter(120-shots_home[~mask_goal_h].pos_x, 80-shots_home[~mask_goal_h].pos_y,
                  s=shots_home[~mask_goal_h].xG*max_marker_size, edgecolors='white', c='#1f77b4',
                  marker='s', zorder=1, label=home_name, ax=ax, alpha=0.8)

    pitch.scatter(120-shots_home[mask_goal_h].pos_x, 80-shots_home[mask_goal_h].pos_y,
                  edgecolors='black', c='white', s=shots_home[mask_goal_h].xG*max_marker_size,
                  zorder=2, marker='s', alpha=0.95, ax=ax)

    # Away
    pitch.scatter(shots_away[~mask_goal_a].pos_x, shots_away[~mask_goal_a].pos_y,
                  s=shots_away[~mask_goal_a].xG*max_marker_size, edgecolors='white', c='#ff7f0e',
                  marker='s', zorder=1, label=away_name, ax=ax, alpha=0.8)

    pitch.scatter(shots_away[mask_goal_a].pos_x, shots_away[mask_goal_a].pos_y,
                  edgecolors='black', c='white', s=shots_away[mask_goal_a].xG*max_marker_size,
                  zorder=2, marker='s', alpha=0.95, label='Goal', ax=ax)

    pitch.draw(ax=ax)

    # Flip so goal is on bottom
    plt.gca().invert_yaxis()

    # Set the title
    ax.set_title(f'{home_name} (H) vs {away_name} (A) Shot Map', fontsize=20, color='w')
    legend = plt.legend(loc='center', markerscale=1, framealpha=1.0, bbox_to_anchor=[0.5, 0.88])
    # make the markers the same size in the legend
    for handle in legend.legendHandles:
        handle.set_sizes([100])

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
    textstr = f'Score: (H) {home_score} - {away_score} (A)\nxG: (H) {home_xg} - {away_xg} (A)'
    ax.text(0.5, 0.125, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props, horizontalalignment='center')

    plt.show()


def gameFlow(df_home, df_away, nonpen=False):
    """
    Creates a cumulative xG flow plot for the match

    Parameters
    ----------
    df_XXXX : A pandas dataframe
        - Team dataframe for home and away
        - Must contain ['minute', 'cumulative_xg', 'h_team', 'a_team']
        - Data used for plotting

    nonpen : boolean
        - T/F value for nonpenalty xg
        - Function uses this to change the calculations

    Returns
    -------
    cumulative_xg : figure
        - Figure showing the cumulative xg for the match as a step graph

    """

    fig, ax = plt.subplots(figsize=(8,5))
    fig.set_facecolor('#3d4849')
    ax.patch.set_facecolor('#3d4849')

    ## Halftime line
    ax.axvline(x=45, color='w', linestyle='--', alpha=0.2)

    ## Set labels
    plt.xticks([0, 15, 30, 45, 60, 75, 90])
    plt.xlabel('Minute')
    plt.ylabel('xG')

    ## Data
    h_x = [0] + list(df_home['minute'])           # grab the xg values, make sure to start at 0
    if df_home['minute'].iloc[-1] < 90:
        h_x.append(90)
    else:
        h_x.append(df_home['minute'].iloc[-1])    # append the last value again at 90 to extend the line
    if nonpen:
        h_y = [0] + list(df_home['npxg_cumulative'])
        h_y.append(df_home['npxg_cumulative'].iloc[-1])
    else:
        h_y = [0] + list(df_home['cumulative_xg'])
        h_y.append(df_home['cumulative_xg'].iloc[-1])

    a_x = [0] + list(df_away['minute'])
    if df_away['minute'].iloc[-1] < 90:
        a_x.append(90)
    else:
        a_x.append(df_away['minute'].iloc[-1])
    if nonpen:
        a_y = [0] + list(df_away['npxg_cumulative'])
        a_y.append(df_away['npxg_cumulative'].iloc[-1])
    else:
        a_y = [0] + list(df_away['cumulative_xg'])
        a_y.append(df_away['cumulative_xg'].iloc[-1])

    ax.step(h_x, h_y, where='post', label=df_home['h_team'][0]+' (H)')
    ax.plot(h_x, h_y, 'C0o', alpha=0.5)

    ax.step(a_x, a_y, where='post', label=df_away['a_team'][0]+' (A)')
    ax.plot(a_x, a_y, 'C1o', alpha=0.5)

    # Make it pretty
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    if nonpen:
        title = 'Cumulative Non-Penalty xG Distribution'
    else:
        title = 'Cumulative xG Distribution'
    ax.set_title(title, color='white', fontsize=20)
    plt.legend(loc='best', fontsize=14, framealpha=1.0)

    plt.show()


def simShots(shots):
    """
    Loops through a list of shot xG from a game and simulates the individual shots,
        returning total number of goals.

    Parameters
    ----------
    shots : A list of xG values representing shots in a game.
        - List of decimals
        - Each shot value needs to be between 0 - 1

    Returns
    -------
    goals : int
        - integer for the total number of shots made in this simulation from the input xG list

    """

    goals = 0

    # Shoot! Update scoreboard if it goes in.
    for count,ele in enumerate(shots):
        if random.random() <= ele:
            goals += 1

    return goals


def simMatch(h_xg, a_xg, batch=True):
    """
    Simulates the outcome of a match based on input xG

    Parameters
    ----------
    h_xg : A list of xG values representing the shots for the home team in a game.
        - List of decimals
        - Each shot value needs to be between 0 - 1
    a_xg : A list of xG values representing the shots for the away team in a game.
        - List of decimals
        - Each shot value needs to be between 0 - 1
    batch : optional
        A boolean option for whether or not this is in batch mode.
        - Batch mode suppresses output so it is smoother for many simulations.
        - Default True
        - If false, outputs the results from a simulated game in text form

    Returns
    -------
    result : str
        - string for the outcome: home win, away win, or draw

    """

    h_goals = simShots(h_xg)
    a_goals = simShots(a_xg)

    # Print outcomes if not in batch
    if not batch:
        if h_goals > a_goals:
            print("Home Wins! {} - {}".format(h_goals, a_goals))
        elif a_goals > h_goals:
            print("Away Wins! {} - {}".format(h_goals, a_goals))
        else:
            print("The teams share of the points! {} - {}".format(h_goals, a_goals))

    # Return outcomes in batch
    elif batch:
        if h_goals > a_goals:
            return 'home'
        elif a_goals > h_goals:
            return 'away'
        else:
            return 'draw'


def calculateMatch(h_xg, a_xg, nSims=100000):
    """
    Calculates the probabilities for a match outcomes based on input xG

    Parameters
    ----------
    h_xg : A list of xG values representing the shots for the home team in a game.
        - List of decimals
        - Each shot value needs to be between 0 - 1
    a_xg : A list of xG values representing the shots for the away team in a game.
        - List of decimals
        - Each shot value needs to be between 0 - 1
    nSims : optional
        An integer number of simulations to run for the game
        - Defaults to 10,000 simulations

    Returns
    -------
    result : str
        - string printing the outcome probabilities
        - home win, away win, or draw
        - Based on simulations gives probability of different outcomes

    """

    home = 0
    away = 0
    draw = 0

    # simulations
    # nSims = 10000
    for i in range(nSims):
        outcome = simMatch(h_xg, a_xg)

        if outcome == 'home':
            home += 1
        elif outcome == 'away':
            away += 1
        elif outcome =='draw':
            draw += 1

    home_per = home/nSims*100
    away_per = away/nSims*100
    draw_per = draw/nSims*100

    print("Over {} simulated games:\n\nHome wins: {:.2f}%\nAway wins: {:.2f}%\nDraw:      {:.2f}%".format(nSims, home_per, away_per, draw_per))



def simGrid(h_xg, a_xg, num_sims=100000):
    """
    Calculates the probabilities for a match outcomes based on input xG

    Parameters
    ----------
    h_xg : A list of xG values representing the shots for the home team in a game.
        - List of decimals
        - Each shot value needs to be between 0 - 1
    a_xg : A list of xG values representing the shots for the away team in a game.
        - List of decimals
        - Each shot value needs to be between 0 - 1
    nSims : optional
        An integer number of simulations to run for the game
        - Defaults to 10,000 simulations

    Returns
    -------
    grid : 2-D list
        - probabilities of different outcomes
        - 6x6 array, binning results into number of goals 0 - 5+
        - Based on simulations gives probability of different outcomes

    """

    # Arrays for storing number of goals
    home_goals = np.zeros(6)
    away_goals = np.zeros(6)

    goals = 0

    # Simulate the games and determine the number of goals a team gets
    # increment the array position corresponding to number of goals
    # Cut off 5+ goals into a single outcome
    for i in range(num_sims):
        goals = simShots(h_xg)
        if goals > 5:
            goals = 5
        home_goals[goals] += 1
        goals = simShots(a_xg)
        if goals > 5:
            goals = 5
        away_goals[goals] += 1

    # Debug: print the lists of simulated goals by each team
    # Remember, the number is the number of simulations in which the team scored those goals
    # For example, home_goals[2] is the number of simulations in which the home team scored 2 goals
    # print(home_goals)
    # print(away_goals)

    # determine the likelihood for team to score number by dividing total number of simulations
    home_ll = [i/num_sims for i in home_goals]
    away_ll = [i/num_sims for i in away_goals]

    # Grid for representing the possible outcomes
    grid = [[0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]]

    # Multiply the likelihoods of certain home-away goals to get probability of result
    for i in range(len(home_ll)):
        for j in range(len(away_ll)):
            grid[i][j] = round(home_ll[i]*away_ll[j],5)

    return grid


def gameProbabilities(grid, nSims=100000):
    """
    Calculates the probabilities for a match outcomes based on input xG already put in grid form

    Parameters
    ----------
    grid : The grid of probabilities for the match outcomes input with xG
        - If the grid is already produced
    nSims : optional
        An integer number of simulations to run for the game
        - Defaults to 10,000 simulations

    Returns
    -------
    result : str
        - string printing the outcome probabilities
        - home win, away win, or draw
        - Based on simulations gives probability of different outcomes

    """

    # Use numpy
    outcomes = np.array(grid)

    draw_prob = np.trace(outcomes)
    away_prob = np.sum(outcomes[np.triu_indices(6, k = 1)])
    home_prob = np.sum(outcomes[np.tril_indices(6, k = -1)])

    print("Over {} simulated games:\n\nHome wins: {:.2f}%\nAway wins: {:.2f}%\nDraw:      {:.2f}%\n".format(nSims, home_prob*100, away_prob*100, draw_prob*100))


def mostLikelyGameOutcome(grid, home_score, away_score, home_team_name='Home Team', away_team_name='Away Team'):
    """
    Outputs the most likely match outcome

    Parameters
    ----------
    grid : The grid of probabilities for the match outcomes input with xG
        - If the grid is already produced
    team_names : optional
        - home and away team names
        - can be specified
        - defaults to home team and away team
    Returns
    -------
    result : str
        - string printing the most likely game outcome

    """

    # Use numpy array
    outcomes = np.array(grid)

    # Possible outcomes
    axis_labels = ['0 ', '1 ', '2 ', '3 ', '4 ', '5+']

    # Max out indicies
    home_score_index = home_score
    if home_score_index > 5:
        home_score_index = 5
    away_score_index = away_score
    if away_score_index > 5:
        away_score_index = 5

    # Most likely outcome:
    max_value=np.argmax(outcomes)
    max_index=np.unravel_index(np.argmax(outcomes),outcomes.shape)

    # Actual goals outcome
    actual_value=outcomes[home_score_index][away_score_index]
    ratio = round(outcomes.flatten()[max_value]/actual_value,2)

    print('\nMost likely outcome:')
    print(f'{home_team_name}   ', axis_labels[max_index[0]], '- ', axis_labels[max_index[1]] + f'   {away_team_name}')
    if ratio == 1:
        print('\nThis was the most likely outcome!')
    else:
        print(f'\nThis was {ratio} more likely than the actual score:\n{home_team_name}   ', home_score, ' - ', away_score, f'   {away_team_name}.')



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", cbar_fontsize=20, cbar_labelsize=14,
            axis_labelsize=15, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=cbar_fontsize)
    cbar.ax.tick_params(labelsize=cbar_labelsize)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=axis_labelsize)
    ax.set_yticklabels(row_labels, fontsize=axis_labelsize)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts





def drawGrid(grid=None, h_xg=None, a_xg=None, grid_textsize=15, titlefontsize=20,
             user_figsize=(8,8), axis_fontsize=14, nonpen=False,
             home_team_name='Home Team', away_team_name='Away Team'):
    """
    Draw heatmap for the match outcomes

    Parameters
    ----------
    grid: The grid of probabilities for the match outcomes input with xG
        - If the grid already produced
    h_xg : A list of xG values representing the shots for the home team in a game.
        - List of decimals
        - Each shot value needs to be between 0 - 1
    a_xg : A list of xG values representing the shots for the away team in a game.
        - List of decimals
        - Each shot value needs to be between 0 - 1
    grid_textsize : optional
        - size of text in grid squares
        - default size=15
    title_fontsize : optional
        - size of text for title
        - default size=20
    user_figsize : optional
        - size of figure
        - defaults to 8,8
    axis_fontsize : optional
        - size of axis labels
        - defaults to 14
    team_name : optional
        - home and away team names
        - can be specified
        - defaults to home team and away team

    Returns
    -------
    heatmap : Draws a probability heatmap of the match outcomes

    """

    if grid is not None:
        outcomes = np.array(grid)
    else:
        outcomes = np.array(simGrid(h_xg, a_xg))

    # Find a way to put this on a heatmap
    axis_labels = ['0 ', '1 ', '2 ', '3 ', '4 ', '5+']

    fig, ax = plt.subplots(figsize=user_figsize)

    im, cbar = heatmap(outcomes, axis_labels, axis_labels, ax=ax,
                       cmap="Oranges", cbarlabel="Probability of final score", vmin=0)
    texts = annotate_heatmap(im, valfmt="{x:.3f}", fontsize=grid_textsize)

    if nonpen:
        title = 'Probabilities of Game Outcomes\n(Non-Penalty xG)'
        ax.set_title(title, fontsize=titlefontsize, pad=30)
    else:
        title = 'Probabilities of Game Outcomes'
        ax.set_title(title, fontsize=titlefontsize, pad=20)
    plt.xlabel(away_team_name, fontsize=axis_fontsize)
    ax.xaxis.set_label_position('top')
    plt.ylabel(home_team_name, fontsize=axis_fontsize)
    fig.tight_layout()
    plt.show()


def goalDistribution(grid, home_name, away_name, nonpen=False):
    """
    Creates a histogram showing the goal scoring distribution for each team side-by-side

    Parameters
    ----------
    grid: The grid of probabilities for the match outcomes input with xG
        - If the grid already produced

    team_name : string
        - Name of team for distribution

    nonpen : boolean
        - T/F value for nonpenalty xg
        - Function uses this to change the title

    Returns
    -------
    distribution : figure
        - The histogram probability distribution of simulated goals scored for each team

    """

    ## Goal probability distribution
    outcomes = np.array(grid)
    home_goal_dist = outcomes.sum(axis=1)
    away_goal_dist = outcomes.sum(axis=0)

    fig, ax = plt.subplots(figsize=(10,6))
    fig.subplots_adjust(bottom=0.25)

    axis_labels = ['0 ', '1 ', '2 ', '3 ', '4 ', '5+']
    x = np.arange(len(axis_labels))

    width = 0.35  # the width of the bars

    ax.bar(x-width/2, home_goal_dist, color='#1f77b4', width=width, label=home_name)
    ax.bar(x+width/2, away_goal_dist, color='#ff7f0e', width=width, label=away_name)

    if nonpen:
        title = f'{home_name} vs. {away_name}\nNon-Penalty Simulated Goal Distribution'
    else:
        title = f'{home_name} vs. {away_name}\nSimulated Goal Distribution'
    plt.title(title, fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(axis_labels)
    plt.xticks(fontsize=14)
    plt.ylabel('Probability', fontSize = 16)
    plt.yticks(fontsize=14)
    plt.legend(loc='best', fontsize=14)

    plt.show()




#########
# Debug #
#########
# Lists of shots

# h_xg = [0.21,0.66,0.1,0.14,0.01]
# a_xg = [0.04,0.06,0.01,0.04,0.06,0.12,0.01,0.06]

# simShots(h_xg)

# simMatch(h_xg, a_xg, False)
# simMatch(h_xg, a_xg)

# calculateMatch(h_xg, a_xg, 100)
# calculateMatch(h_xg, a_xg, 10000)

# grid = simGrid()
# print(grid)

if __name__ == "__main__":
    print("You ran this module directly (and did not 'import' it).")
    input("\n\nPress the enter key to exit.")


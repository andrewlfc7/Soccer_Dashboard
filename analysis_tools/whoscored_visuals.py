import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.colors import to_rgba
from mplsoccer import Pitch, VerticalPitch


def compute_contested_zones(matchID, team_name, data):
    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.1,
        line_color='black',
        pad_top=10,
        corner_arcs=True
    )
    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y
    data = data.copy()
    df_match = data[data['match_id'] == matchID]
    # -- Adjust opposition figures
    df_match.loc[:, 'x'] = [100 - x if y != team_name else x for x, y in zip(df_match['x'], df_match['team_name'])]
    df_match.loc[:, 'y'] = [100 - x if y != team_name else x for x, y in zip(df_match['y'], df_match['team_name'])]
    df_match = df_match.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    df_match = df_match.assign(bins_y=lambda x: pd.cut(x.y, bins=list(pos_y) + [105]))
    df_match_groupped = df_match.groupby(['bins_x', 'bins_y', 'team_name', 'match_id'])['isTouch'].sum().reset_index(
        name='touches')
    df_team = df_match_groupped[df_match_groupped['team_name'] == team_name]
    df_oppo = df_match_groupped[df_match_groupped['team_name'] != team_name].rename(
        columns={'team_name': 'opp_name', 'touches': 'opp_touches'})
    df_plot = pd.merge(df_team, df_oppo, on=['bins_x', 'bins_y'])
    df_plot = df_plot.assign(ratio=lambda x: x.touches / (x.touches + x.opp_touches))
    df_plot['left_x'] = df_plot['bins_x'].apply(lambda x: x.left).astype(float)
    df_plot['right_x'] = df_plot['bins_x'].apply(lambda x: x.right).astype(float)
    df_plot['left_y'] = df_plot['bins_y'].apply(lambda x: x.left).astype(float)
    df_plot['right_y'] = df_plot['bins_y'].apply(lambda x: x.right).astype(float)
    return df_plot


def plot_zone_dominance(ax, match_id, team_name, homecolor, awaycolor, zonecolor, data):
    data_plot = data.copy()
    data_plot = compute_contested_zones(match_id, team_name, data=data_plot)
    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        pad_top=10,
        corner_arcs=True
    )
    pitch.draw(ax=ax)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
                condition_bounds = (data_plot['left_x'] >= lower_x) & (data_plot['right_x'] <= upper_x) & (
                        data_plot['left_y'] >= lower_y) & (data_plot['right_y'] <= upper_y)
                data_point = data_plot[condition_bounds]['ratio'].iloc[0]
                if data_point > .55:
                    color = homecolor
                elif data_point < .45:
                    color = awaycolor
                else:
                    color = zonecolor
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=0.75,
                    ec='None'
                )
            except:
                continue

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    return ax


def plot_pass_map_with_xT(ax, teamId:int, data):
    """Parameters:
    ax (Matplotlib Axes object): Axes object to plot the map on
    teamId (int): The Opta ID of the team to plot the map for
    data (dict): A dictionary containing Opta event data"""

    data = data.copy()
    data =data[data['teamId'] == teamId]



    def get_passes_df(events_dict):
        df = pd.DataFrame(events_dict)
        # create receiver column based on the next event
        # this will be correct only for successfull passes
        df["pass_recipient"] = df["playerName"].shift(-1)
        # filter only passes

        passes_ids = df.index[df['event_type'] == 'Pass']
        df_passes = df.loc[
            passes_ids, ["id", "minute", "x", "y", "endX", "endY", "teamId", "playerId", "playerName", "event_type",
                         "outcomeType", "pass_recipient",'isTouch','xThreat_gen']]

        return df_passes


    passes_df = get_passes_df(data)
    passes_df
    passes_df = passes_df[passes_df['outcomeType'] == 'Successful']
    #find the first subsititution and filter the successful dataframe to be less than that minute
    subs = data[data['event_type'] == 'SubstitutionOff']
    subs
    subs = subs['minute']
    firstSub = subs.min()

    passes_df = passes_df[passes_df['minute'] < firstSub]

    average_locs_and_count_df = (passes_df.groupby('playerName')
                                 .agg({'x': ['mean'], 'y': ['mean', 'count']}))
    average_locs_and_count_df.columns = ['x', 'y', 'count']
    average_locs_and_count_df
    #average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position']],
    #                                                            on='playerId', how='left')
    #average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')





    pass_between = passes_df.groupby(['playerName', 'pass_recipient']).agg(total_xt=('xThreat_gen', 'sum'),
                                                                           pass_count=('id', 'count')).reset_index()

    #pass_between = passes_df.groupby(['playerName', 'pass_recipient'])['xT'].sum().reset_index()
    #pass_between.rename({'xT': 'total_xt', 'id': 'pass_count'}, axis='columns', inplace=True)




    pass_between = pass_between.merge(average_locs_and_count_df, left_on='playerName', right_index=True)
    pass_between = pass_between.merge(average_locs_and_count_df, left_on='pass_recipient', right_index=True,
                                      suffixes=['', '_end'])


    pass_between = pass_between[pass_between['pass_count'] >= 3]

    # Group passes by playerId and sum their xt values
    player_xt_df = passes_df.groupby('playerName')['xThreat_gen'].sum().reset_index()

    # Merge with the original player locations dataframe
    average_locs_and_count_df = pd.merge(average_locs_and_count_df, player_xt_df, on='playerName')


    MAX_MARKER_SIZE = 1200
    average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count']
                                                / average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE)

    MAX_LINE_WIDTH = 20
    pass_between['width'] = (pass_between.pass_count / pass_between.pass_count.max() *
                             MAX_LINE_WIDTH)

    MIN_TRANSPARENCY = 0.2
    color = np.array(to_rgba('#1D535B'))
    color = np.tile(color, (len(average_locs_and_count_df.xThreat_gen), 1))
    c_transparency = average_locs_and_count_df.xThreat_gen.min() / average_locs_and_count_df.xThreat_gen.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency

    #adjusting that only the surname of a player is presented.
    average_locs_and_count_df["playerName"] = average_locs_and_count_df["playerName"].apply(lambda x: str(x).split()[-1])
    #df_pass["pass_recipient_name"] = df_pass["pass_recipient_name"].apply(lambda x: str(x).split()[-1])



    MIN_TRANSPARENCY = 0.3
    color = np.array(to_rgba('#009991'))
    color = np.tile(color, (len(pass_between), 1))
    c_transparency = pass_between.pass_count / pass_between.pass_count.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency



    pitch = pitch = Pitch(
        pitch_color= '#201D1D',
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='opta',
        goal_type='box',
        pad_top=10,
        line_color='black')
    pitch.draw(ax=ax)

    norm = colors.Normalize(vmin=pass_between.total_xt.min(), vmax=pass_between.total_xt.max())

    # plot the arrows
    arrows = pitch.lines(pass_between.x,pass_between.y,pass_between.x_end,pass_between.y_end, linewidth=pass_between.width,
                         color=plt.cm.coolwarm(norm(pass_between.total_xt.values)),
                         ax=ax, zorder=1, alpha=1)


    # Visualize the nodes using the new xt values
    nodes = pitch.scatter(average_locs_and_count_df.x, average_locs_and_count_df.y,
                          s = average_locs_and_count_df.marker_size, c = average_locs_and_count_df.xThreat_gen,
                          cmap = 'coolwarm', linewidth = 2.5, alpha = 1, zorder = 1, ax=ax)



    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0] - 0.5, pos_y[-1] + 0.5], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0] - 0.5, pos_x[-1] + 0.5], [y, y], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)

    for index, row in average_locs_and_count_df.iterrows():
        pitch.annotate(row.playerName, xy=(row.x, row.y), c='#132743', va='center', ha='center', size=6, ax=ax)



    return ax




def plot_pass_map_minute_xT(ax, teamId:int, minute_start:int, minute_end:int, data):
    """Parameters:
    ax (Matplotlib Axes object): Axes object to plot the map on
    teamId (int): The Opta ID of the team to plot the map for
    data (dict): A dictionary containing Opta event data"""

    data = data.copy()
    data =data[data['teamId'] == teamId]



    def get_passes_df(events_dict):
        df = pd.DataFrame(events_dict)
        # create receiver column based on the next event
        # this will be correct only for successfull passes
        df["pass_recipient"] = df["playerName"].shift(-1)
        # filter only passes

        passes_ids = df.index[df['event_type'] == 'Pass']
        df_passes = df.loc[
            passes_ids, ["id", "minute", "x", "y", "endX", "endY", "teamId", "playerId", "playerName", "event_type",
                         "outcomeType", "pass_recipient",'isTouch','xThreat_gen']]

        return df_passes


    passes_df = get_passes_df(data)
    passes_df
    passes_df = passes_df[passes_df['outcomeType'] == 'Successful']

    minute_mask = (passes_df['minute'] >= minute_start) & (passes_df['minute'] <= minute_end)
    # df_successful = df_successful.loc[(period_mask & minute_mask)]
    passes_df = passes_df.loc[minute_mask]



    average_locs_and_count_df = (passes_df.groupby('playerName')
                                 .agg({'x': ['mean'], 'y': ['mean', 'count']}))
    average_locs_and_count_df.columns = ['x', 'y', 'count']
    average_locs_and_count_df
    #average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position']],
    #                                                            on='playerId', how='left')
    #average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')





    pass_between = passes_df.groupby(['playerName', 'pass_recipient']).agg(total_xt=('xThreat_gen', 'sum'),
                                                                           pass_count=('id', 'count')).reset_index()

    #pass_between = passes_df.groupby(['playerName', 'pass_recipient'])['xT'].sum().reset_index()
    #pass_between.rename({'xT': 'total_xt', 'id': 'pass_count'}, axis='columns', inplace=True)




    pass_between = pass_between.merge(average_locs_and_count_df, left_on='playerName', right_index=True)
    pass_between = pass_between.merge(average_locs_and_count_df, left_on='pass_recipient', right_index=True,
                                      suffixes=['', '_end'])


    #pass_between = pass_between[pass_between['pass_count'] >= 3]

    # Group passes by playerId and sum their xt values
    player_xt_df = passes_df.groupby('playerName')['xThreat_gen'].sum().reset_index()

    # Merge with the original player locations dataframe
    average_locs_and_count_df = pd.merge(average_locs_and_count_df, player_xt_df, on='playerName')


    MAX_MARKER_SIZE = 1200
    average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count']
                                                / average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE)

    MAX_LINE_WIDTH = 20
    pass_between['width'] = (pass_between.pass_count / pass_between.pass_count.max() *
                             MAX_LINE_WIDTH)

    MIN_TRANSPARENCY = 0.2
    color = np.array(to_rgba('#1D535B'))
    color = np.tile(color, (len(average_locs_and_count_df.xThreat_gen), 1))
    c_transparency = average_locs_and_count_df.xThreat_gen.min() / average_locs_and_count_df.xThreat_gen.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency

    #adjusting that only the surname of a player is presented.
    average_locs_and_count_df["playerName"] = average_locs_and_count_df["playerName"].apply(lambda x: str(x).split()[-1])
    #df_pass["pass_recipient_name"] = df_pass["pass_recipient_name"].apply(lambda x: str(x).split()[-1])



    MIN_TRANSPARENCY = 0.3
    color = np.array(to_rgba('#009991'))
    color = np.tile(color, (len(pass_between), 1))
    c_transparency = pass_between.pass_count / pass_between.pass_count.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency



    pitch = pitch = Pitch(
        pitch_color= '#201D1D',
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='opta',
        goal_type='box',
        pad_top=10,
        line_color='black')
    pitch.draw(ax=ax)

    norm = colors.Normalize(vmin=pass_between.total_xt.min(), vmax=pass_between.total_xt.max())

    # plot the arrows
    arrows = pitch.lines(pass_between.x,pass_between.y,pass_between.x_end,pass_between.y_end, linewidth=pass_between.width,
                         color=plt.cm.coolwarm(norm(pass_between.total_xt.values)),
                         ax=ax, zorder=1, alpha=1)


    # Visualize the nodes using the new xt values
    nodes = pitch.scatter(average_locs_and_count_df.x, average_locs_and_count_df.y,
                          s = average_locs_and_count_df.marker_size, c = average_locs_and_count_df.xThreat_gen,
                          cmap = 'coolwarm', linewidth = 2.5, alpha = 1, zorder = 1, ax=ax)



    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0] - 0.5, pos_y[-1] + 0.5], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0] - 0.5, pos_x[-1] + 0.5], [y, y], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)

    for index, row in average_locs_and_count_df.iterrows():
        pitch.annotate(row.playerName, xy=(row.x, row.y), c='#132743', va='center', ha='center', size=6, ax=ax)



    return ax




def plot_pitch_opta(ax):
    fig = plt.figure(figsize=(10, 8), dpi=900)
    ax = plt.subplot(111)
    fig.set_facecolor("#2E2929")
    pitch = pitch = Pitch(linewidth=2.1,
                          corner_arcs=True,
                          pitch_type='opta',
                          pitch_color='#2E2929',
                          goal_type='box',
                          pad_top=10,
                          line_color='#141414')
    pitch.draw(ax=ax)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    return ax


# %%


def plot_team_offensive_actions(ax, match_id, team_id, data):
    data = data[data['is_open_play'] == True]

    data_offensive = data.copy()

    data_offensive = data_offensive[(data_offensive['match_id'] == match_id) & (data_offensive['teamId'] == team_id)]

    data_offensive = data_offensive[data_offensive['x'] >= 50].reset_index(drop=True)

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=True
    )

    pitch.draw(ax=ax)

    fig = plt.figure(figsize=(8, 4))

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_offensive = data_offensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_offensive = data_offensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_offensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='#4A9FA2',
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1

    unique_actions = data_offensive['event_type'].unique()
    actions = list(unique_actions)
    markers = ['o', 'X', 'v', 's', '^']
    for a, m in zip(actions, markers):
        if a == 'TakeOn':
            a_label = 'Take On'
        else:
            a_label = a
        num_data_points = len(data_offensive[data_offensive['event_type'] == a])
        marker_size = 40 + num_data_points * 0.5
        ax.scatter(data_offensive[data_offensive['event_type'] == a].x,
                   data_offensive[data_offensive['event_type'] == a].y, s=marker_size, alpha=0.85, lw=0.85,
                   fc='#3c6e71', ec='#2F2B2B', zorder=3, marker=m, label=a_label)

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax


# %%


def plot_team_defensive_actions_opp_half(ax, match_id: int, team_id: int, data):
    data =data.copy()
    data = data[data['is_open_play'] == True]



    data_recoveries = data[data['won_possession'] == True]

    data_recoveries = data_recoveries[
        (data_recoveries['match_id'] == match_id) & (data_recoveries['teamId'] == team_id)]

    data_defensive = data_recoveries.copy()

    data_defensive = data_defensive[data_defensive['x'] >= 50].reset_index(drop=True)

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=True
    )

    pitch.draw(ax=ax)

    fig = plt.figure(figsize=(8, 4))

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_defensive = data_defensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_defensive = data_defensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_defensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='#4A9FA2',
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1

    unique_actions = data_defensive['event_type'].unique()
    actions = list(unique_actions)
    markers = ['o', 'X', 'v', 's', '^']
    for a, m in zip(actions, markers):
        if a == 'BallRecovery':
            a_label = 'Ball recovery'
        else:
            a_label = a
        num_data_points = len(data_defensive[data_defensive['event_type'] == a])
        marker_size = 40 + num_data_points * 0.5
        ax.scatter(data_defensive[data_defensive['event_type'] == a].x,
                   data_defensive[data_defensive['event_type'] == a].y, s=marker_size, alpha=0.85, lw=0.85,
                   fc='#3c6e71', ec='#2F2B2B', zorder=3, marker=m, label=a_label)

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax


def plot_player_hull_team(player_df, ax, poly_edgecolor='#A0A0A0', poly_facecolor='#379A95', poly_alpha=0.3,
                     scatter_edgecolor='#0C0F0A', scatter_facecolor='#3c6e71', avg_marker_size=600):
    pitch = Pitch()
    # Draw convex hull polygon
    hull = pitch.convexhull(player_df.x, player_df.y)
    poly = pitch.polygon(hull, ax=ax, edgecolor=poly_edgecolor, facecolor=poly_facecolor, alpha=poly_alpha)

    # Draw scatter plot
    scatter = pitch.scatter(player_df.x, player_df.y, ax=ax, edgecolor=scatter_edgecolor, facecolor=scatter_facecolor)

    # Draw average location marker
    pitch.scatter(player_df.x_avg, player_df.y_avg, ax=ax, edgecolor=scatter_edgecolor, facecolor=poly_facecolor,
                  s=avg_marker_size, marker='o', alpha=.30)

    # Add player initials as labels
    for i, row in player_df.iterrows():
        ax.text(row['x_avg'], row['y_avg'], row['initials'], fontsize=8, color='Black', ha='center', va='center')

    return ax


# %%

# %%

def plot_player_passmap_opta(ax, player_name, data):
    data = data.copy()

    data = data[data['is_open_play']==True]
    data = data[data['playerName'] == player_name]
    data = data[data['event_type'] == 'Pass']
    # fig = plt.figure(figsize = (4,4), dpi = 900)
    # ax = plt.subplot(111)
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',

    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    pitch.arrows(data.x, data.y, data.endX, data.endY, color="#2a9d8f", width=2, headwidth=4, headlength=4, alpha=.8 ,label = 'Open Play Pass' , ax=ax)


    legend = ax.legend(ncol=2, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')



    return ax


# %%

def plot_convex_hull_opta_player(ax, player_name, data):
    data = data.copy()
    data = data[(data['playerName'] == player_name) & (data['isTouch'] == True)]

    player_avg_pos = data.groupby('playerName')[['x', 'y']].mean()
    player_avg_pos['initials'] = player_avg_pos.index.str.split().str[0].str[0] + \
                                 player_avg_pos.index.str.split().str[-1].str[0]

    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',

    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    hull = pitch.convexhull(data.x, data.y)
    poly = pitch.polygon(hull, ax=ax, edgecolor='#2a9d8f', facecolor='#2a9d8f', alpha=0.3)

    ax.scatter(data.x, data.y, s=16, alpha=1, edgecolors='#1A1D1A', color='#2a9d8f',label = 'Open Play Touch')

    pitch.scatter(player_avg_pos.x, player_avg_pos.y, ax=ax, edgecolor='#2a9d8f', facecolor='#2a9d8f', s=280,
                  marker='o', alpha=.30, label = 'Average Postion')

    for i, row in player_avg_pos.iterrows():
        ax.text(row['x'], row['y'], row['initials'], fontsize=8, color='Black', ha='center', va='center')
    legend = ax.legend(ncol=3, loc='upper center', fontsize=6, handlelength=2.5, handleheight=2.5, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


    return ax


# %%
def plot_players_defensive_actions_opta(ax, playerName, data):
    data = data.copy()
    data = data[data['is_open_play'] == True]
    # data = def_action(data)

    data = data[(data['playerName'] == playerName) & (data['outcomeType'] == 'Successful')]

    data_defensive = data.copy()

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_defensive = data_defensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_defensive = data_defensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_defensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='#4A9FA2',
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1

    unique_actions = data_defensive['event_type'].unique()
    actions = list(unique_actions)
    markers = ['o', 'X', 'v', 's', '^']
    for a, m in zip(actions, markers):
        if a == 'BallRecovery':
            a_label = 'Ball recovery'
        else:
            a_label = a
        num_data_points = len(data_defensive[data_defensive['event_type'] == a])
        marker_size = 40 + num_data_points * 0.5
        ax.scatter(data_defensive[data_defensive['event_type'] == a].x,
                   data_defensive[data_defensive['event_type'] == a].y, s=marker_size, alpha=0.85, lw=0.85,
                   fc='#3c6e71', ec='#2F2B2B', zorder=3, marker=m, label=a_label)

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax


def plot_players_offensive_actions_opta(ax, playerName, data):
    data = data.copy()
    data = data[data['is_open_play'] == True]
    data = data[(data['playerName'] == playerName) & (data['outcomeType'] == 'Successful')]

    data_offensive = data.copy()

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_offensive = data_offensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_offensive = data_offensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_offensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='#4A9FA2',
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1

    unique_actions = data_offensive['event_type'].unique()
    actions = list(unique_actions)
    markers = ['o', 'X', 'v', 's', '^']
    for a, m in zip(actions, markers):
        if a == 'BallRecovery':
            a_label = 'Ball recovery'
        else:
            a_label = a
        num_data_points = len(data_offensive[data_offensive['event_type'] == a])
        marker_size = 40 + num_data_points * 0.5
        ax.scatter(data_offensive[data_offensive['event_type'] == a].x,
                   data_offensive[data_offensive['event_type'] == a].y, s=marker_size, alpha=0.85, lw=0.85,
                   fc='#3c6e71', ec='#2F2B2B', zorder=3, marker=m, label=a_label)

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax


def plot_carry_opta(ax, player_name, data):
    data = data.copy()
    data = data[data['playerName'] == player_name]
    data = data[data['event_type'] == 'Carry']
    # fig = plt.figure(figsize = (4,4), dpi = 900)
    # ax = plt.subplot(111)
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',

    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    pitch.arrows(data.x, data.y, data.endX, data.endY, color="#2a9d8f", width=1.5, headwidth=3, headlength=4, ax=ax)

    return ax




def plot_player_passes_rec_opta(ax, player_name, data):
    data = data.copy()
    data = data[data['pass_recipient'] == player_name]
    # fig = plt.figure(figsize = (4,4), dpi = 900)
    # ax = plt.subplot(111)
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',

    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    pitch.arrows(data.x, data.y, data.endX, data.endY, width=2,
                 headwidth=4, headlength=4, color='#2a9d8f', alpha=.8, label= 'Pass Receive', ax=ax)

    # ax.scatter(data.x, data.y, s=16, alpha=1,  edgecolors='#1A1D1A' , color='#2a9d8f')

    legend = ax.legend(ncol=1, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')


    return ax


def plot_carry_into_box_team_opta(ax, team_id, data):
    data = data.copy()
    data = data[data['teamId'] == team_id]
    data = data[data['event_type'] == 'Carry']
    # fig = plt.figure(figsize = (4,4), dpi = 900)
    # ax = plt.subplot(111)
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',

    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    # pitch.arrows(data.x, data.y, data.endX, data.endY, color="#2a9d8f", width=1.5, headwidth=3, headlength=4, ax=ax)

    # Plot arrows for carries into the box in blue
    pitch.arrows(data[data['is_carry_into_box'] == True].x, data[data['is_carry_into_box'] == True].y,
                 data[data['is_carry_into_box'] == True].endX, data[data['is_carry_into_box'] == True].endY,
                 color='#4A7B9D', width=1.5, headwidth=3, headlength=4, alpha=.7, ax=ax, label='Carries into box')

    # Plot arrows for carries outside the box in green
    pitch.arrows(data[data['is_carry_into_box'] != True].x, data[data['is_carry_into_box'] != True].y,
                 data[data['is_carry_into_box'] != True].endX, data[data['is_carry_into_box'] != True].endY,
                 color='#2a9d8f', width=1.5, headwidth=3, headlength=4, alpha=.7, ax=ax, label='Carries')

    # Add legend
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    return ax


def plot_carry_into_box_player_opta(ax, player_name, data):
    data = data.copy()
    data = data[data['playerName'] == player_name]
    data = data[data['event_type'] == 'Carry']
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',
    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    # Plot arrows for carries into the box in blue
    pitch.arrows(data[data['is_carry_into_box'] == True].x, data[data['is_carry_into_box'] == True].y,
                 data[data['is_carry_into_box'] == True].endX, data[data['is_carry_into_box'] == True].endY,
                 color='#4A7B9D', width=1.5, headwidth=3, headlength=4, alpha=.7, ax=ax, label='Carries into box')

    # Plot arrows for carries outside the box in green
    pitch.arrows(data[data['is_carry_into_box'] != True].x, data[data['is_carry_into_box'] != True].y,
                 data[data['is_carry_into_box'] != True].endX, data[data['is_carry_into_box'] != True].endY,
                 color='#2a9d8f', width=1.5, headwidth=3, headlength=4, alpha=.7, ax=ax, label='Carries')

    # Add legend
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')



    return ax


def plot_team_touch_heatmap_opta(ax, match_id, team_id, data):
    data = data[data['is_open_play'] == True]


    data_offensive = data.copy()

    data_offensive = data_offensive[(data_offensive['match_id'] == match_id) & (data_offensive['teamId'] == team_id)]

    # data_offensive = data_offensive[(data_offensive['event_type'] == 'Pass') & (data_offensive['isTouch'] == true)]

    data_offensive = data_offensive[data_offensive['x'] > 50].reset_index(drop=True)

    data_offensive = data_offensive[(data_offensive['event_type'] == 'Pass') & (data_offensive['isTouch'] == True)]

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=True
    )

    pitch.draw(ax=ax)

    #fig = plt.figure(figsize=(8, 4))

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_offensive = data_offensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_offensive = data_offensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_offensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='#4A9FA2',
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1

        #ax.scatter(data_offensive.x, data_offensive.y, s=20, alpha=0.6, lw=0.85,
        #          fc='#3c6e71', ec='#2F2B2B', zorder=3, marker='o', label = "Touch")

        #pitch.arrows(data_offensive.x, data_offensive.y, data_offensive.endX, data_offensive.endY, color="#2a9d8f", width=1.5, headwidth=3, headlength=4, ax=ax)


    ax.scatter(data_offensive.x, data_offensive.y, s=20, alpha=0.6, lw=0.85,
               fc='#3c6e71', ec='#2F2B2B', zorder=3, marker='o', label="Open Play Touches")

    legend = ax.legend(loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))


    #legend = ax.legend(['Open Play Touch'], loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))

    #legend = ax.legend( loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax


def plot_team_touch_heatmap_full_opta(ax, match_id, team_id, data):
    data = data[data['is_open_play'] == True]


    data_touch = data.copy()

    data_touch = data_touch[(data_touch['match_id'] == match_id) & (data_touch['teamId'] == team_id)]

    # data_touch = data_touch[(data_touch['event_type'] == 'Pass') & (data_touch['isTouch'] == true)]

    #data_touch = data_touch[data_touch['x'] > 50].reset_index(drop=True)

    data_touch = data_touch[(data_touch['event_type'] == 'Pass') & (data_touch['isTouch'] == True)]

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    #fig = plt.figure(figsize=(8, 4))

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_touch = data_touch.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_touch = data_touch.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_touch.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='#4A9FA2',
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1

        #ax.scatter(data_offensive.x, data_offensive.y, s=20, alpha=0.6, lw=0.85,
        #          fc='#3c6e71', ec='#2F2B2B', zorder=3, marker='o', label = "Touch")

        #pitch.arrows(data_offensive.x, data_offensive.y, data_offensive.endX, data_offensive.endY, color="#2a9d8f", width=1.5, headwidth=3, headlength=4, ax=ax)


    ax.scatter(data_touch.x, data_touch.y, s=20, alpha=0.6, lw=0.85,
               fc='#3c6e71', ec='#2F2B2B', zorder=3, marker='o', label="Open Play Touches")

    legend = ax.legend( loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))


    #legend = ax.legend(['Open Play Touch'], loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))

    #legend = ax.legend( loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax



def plot_xT_flow_chart(ax ,home_color, away_color, data):

    data = data.copy()

    data = data[data['xThreat'] >= 0]

    #df_minute = data.groupby([pd.Grouper(key='minute'), 'teamId','Venue'])['xT'].mean().reset_index()

    df_minute = data.groupby(['minute', 'Venue','teamId'])['xThreat'].mean().reset_index()


    #df_minute['xT_rolling_avg'] = df_minute.groupby('teamId')['xThreat'].apply(lambda x: x.ewm(span=5).mean()).reset_index(0, drop=True)

    df_minute['xT_rolling_avg'] = df_minute.groupby('teamId')['xThreat'].rolling(window=14, min_periods=0).mean().reset_index(0, drop=True)



    df_home = df_minute[df_minute['Venue'] == 'Home']
    df_away = df_minute[df_minute['Venue'] == 'Away']


    ax.plot(0+ df_home['minute'], df_home['xT_rolling_avg'], color=home_color,  lw = 1)
    ax.plot(0+ df_away['minute'], -df_away['xT_rolling_avg'], color=away_color,  lw = 1)

    ax.fill_between(df_home['minute'], df_home['xT_rolling_avg'], 0, where=df_home['xT_rolling_avg'] > 0, color=home_color, alpha=0.2)
    ax.fill_between(df_away['minute'], -df_away['xT_rolling_avg'], 0, where=df_away['xT_rolling_avg'] > 0, color=away_color, alpha=0.2)

    plt.axvline(45, c='#018b95',linestyle="--",ymin=0, ymax=5)


    ax.set_xlabel('Minute')
    ax.set_ylabel('xT')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticklabels([])
    ax.set_facecolor("#201D1D")
    ax.grid(False)


    return ax




def plot_team_touch_heatmap_opta_halfpitch(ax, match_id, team_id, data):
    data = data[data['is_open_play'] == True]


    data_offensive = data.copy()

    data_offensive = data_offensive[(data_offensive['match_id'] == match_id) & (data_offensive['teamId'] == team_id)]

    # data_offensive = data_offensive[(data_offensive['event_type'] == 'Pass') & (data_offensive['isTouch'] == true)]

    data_offensive = data_offensive[data_offensive['x'] > 50].reset_index(drop=True)

    data_offensive = data_offensive[(data_offensive['event_type'] == 'Pass') & (data_offensive['isTouch'] == True)]

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=True
    )

    pitch.draw(ax=ax)

    #fig = plt.figure(figsize=(8, 4))

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_offensive = data_offensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_offensive = data_offensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_offensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='#4A9FA2',
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1

        #ax.scatter(data_offensive.x, data_offensive.y, s=20, alpha=0.6, lw=0.85,
        #          fc='#3c6e71', ec='#2F2B2B', zorder=3, marker='o', label = "Touch")

        #pitch.arrows(data_offensive.x, data_offensive.y, data_offensive.endX, data_offensive.endY, color="#2a9d8f", width=1.5, headwidth=3, headlength=4, ax=ax)


    ax.scatter(data_offensive.x, data_offensive.y, s=20, alpha=0.6, lw=0.85,
               fc='#3c6e71', ec='#2F2B2B', zorder=3, marker='o', label="Open Play Touches")

    legend = ax.legend( loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))


    #legend = ax.legend(['Open Play Touch'], loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))

    #legend = ax.legend( loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax



def plot_passes_into_box_team_opta(ax, teamid, data):
    data = data.copy()
    data = data[data['teamId'] == teamid]
    data = data[data['is_open_play']==True]
    data = data[(data['event_type'] == 'Pass') | (data['outcomeType'] == 'Successful')]


    pitch = VerticalPitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',
        half=True
    )
    pitch.draw(ax=ax)


    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    # Remember that we need to invert the axis!!
    for x in pos_x[1:-1]:
        ax.plot([pos_y[0], pos_y[-1]], [x,x], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([y,y], [pos_x[0], pos_x[-1]], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)



    # Plot arrows for passes into the box in blue
    pitch.arrows(data[data['is_pass_into_box'] == True].x, data[data['is_pass_into_box'] == True].y,
                 data[data['is_pass_into_box'] == True].endX, data[data['is_pass_into_box'] == True].endY,
                 color = '#2a9d8f', width=1.5, headwidth=3, headlength=4, alpha= .8, ax=ax, label='Passes into box')




    # Add legend
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    return ax


# %%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

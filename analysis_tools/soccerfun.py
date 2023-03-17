import numpy as np
import pandas as pd


def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=60, min_carry_duration=1, max_carry_duration=10):
    """ Add carry events to whoscored-style events dataframe

    Function to read a whoscored-style events dataframe (single or multiple matches) and return an event dataframe
    that contains carry information.

    Args:
        events_df (pandas.DataFrame): whoscored-style dataframe of event data. Events can be from multiple matches.
        min_carry_length (float, optional): minimum distance required for event to qualify as carry. 5m by default.
        max_carry_length (float, optional): largest distance in which event can qualify as carry. 60m by default.
        min_carry_duration (float, optional): minimum duration required for event to quality as carry. 2s by default.
        max_carry_duration (float, optional): longest duration in which event can qualify as carry. 10s by default.

    Returns:
        pandas.DataFrame: whoscored-style dataframe of events including carries
    """

    # Initialise output dataframe
    events_out = pd.DataFrame()

    # Carry conditions (convert from metres to opta)
    min_carry_length = 3.0
    max_carry_length = 60.0
    min_carry_duration = 1.0
    max_carry_duration = 10.0

    for match_id in events_df['match_id'].unique():

        match_events = events_df[events_df['match_id'] == match_id].reset_index()
        match_carries = pd.DataFrame()

        for idx, match_event in match_events.iterrows():

            if idx < len(match_events) - 1:
                prev_evt_team = match_event['teamId']
                next_evt_idx = idx + 1
                init_next_evt = match_events.loc[next_evt_idx]
                take_ons = 0
                incorrect_next_evt = True

                while incorrect_next_evt:

                    next_evt = match_events.loc[next_evt_idx]

                    if next_evt['eventType'] == 'TakeOn' and next_evt['outcomeType'] == 'Successful':
                        take_ons += 1
                        incorrect_next_evt = True

                    elif ((next_evt['eventType'] == 'TakeOn' and next_evt['outcomeType'] == 'Unsuccessful')
                          or (next_evt['teamId'] != prev_evt_team and next_evt['eventType'] == 'Challenge' and next_evt[
                                'outcomeType'] == 'Unsuccessful')
                          or (next_evt['eventType'] == 'Foul')):
                        incorrect_next_evt = True

                    else:
                        incorrect_next_evt = False

                    next_evt_idx += 1

                # Apply some conditioning to determine whether carry criteria is satisfied

                same_team = prev_evt_team == next_evt['teamId']
                not_ball_touch = match_event['eventType'] != 'BallTouch'
                dx = 105*(match_event['endX'] - next_evt['x'])/100
                dy = 68*(match_event['endY'] - next_evt['y'])/100
                far_enough = dx ** 2 + dy ** 2 >= min_carry_length ** 2
                not_too_far = dx ** 2 + dy ** 2 <= max_carry_length ** 2
                dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
                min_time = dt >= min_carry_duration
                same_phase = dt < max_carry_duration
                same_period = match_event['period'] == next_evt['period']

                valid_carry = same_team & not_ball_touch & far_enough & not_too_far & min_time & same_phase &same_period

                if valid_carry:
                    carry = pd.DataFrame()
                    prev = match_event
                    nex = next_evt

                    carry.loc[0, 'eventId'] = prev['eventId'] + 0.5
                    carry['minute'] = np.floor(((init_next_evt['minute'] * 60 + init_next_evt['second']) + (
                            prev['minute'] * 60 + prev['second'])) / (2 * 60))
                    carry['second'] = (((init_next_evt['minute'] * 60 + init_next_evt['second']) +
                                        (prev['minute'] * 60 + prev['second'])) / 2) - (carry['minute'] * 60)
                    carry['teamId'] = nex['teamId']
                    carry['x'] = prev['endX']
                    carry['y'] = prev['endY']
                    carry['expandedMinute'] = np.floor(
                        ((init_next_evt['expandedMinute'] * 60 + init_next_evt['second']) +
                         (prev['expandedMinute'] * 60 + prev['second'])) / (2 * 60))
                    carry['period'] = nex['period']
                    carry['type'] = carry.apply(lambda x: {'value': 99, 'displayName': 'Carry'}, axis=1)
                    carry['outcomeType'] = 'Successful'
                    carry['qualifiers'] = carry.apply(
                        lambda x: {'type': {'value': 999, 'displayName': 'takeOns'}, 'value': str(take_ons)}, axis=1)
                    carry['satisfiedEventsTypes'] = carry.apply(lambda x: [], axis=1)
                    carry['isTouch'] = True
                    carry['playerId'] = nex['playerId']
                    carry['endX'] = nex['x']
                    carry['endY'] = nex['y']
                    carry['blockedX'] = np.nan
                    carry['blockedY'] = np.nan
                    carry['goalMouthZ'] = np.nan
                    carry['goalMouthY'] = np.nan
                    carry['isShot'] = np.nan
                    carry['relatedEventId'] = nex['eventId']
                    carry['relatedPlayerId'] = np.nan
                    carry['isGoal'] = np.nan
                    carry['cardType'] = np.nan
                    carry['isOwnGoal'] = np.nan
                    carry['match_id'] = nex['match_id']
                    carry['eventType'] = 'Carry'
                    carry['cumulative_mins'] = (prev['cumulative_mins'] + init_next_evt['cumulative_mins']) / 2

                    match_carries = pd.concat([match_carries, carry], ignore_index=True, sort=False)

        match_events_and_carries = pd.concat([match_carries, match_events], ignore_index=True, sort=False)
        match_events_and_carries = match_events_and_carries.sort_values(
            ['match_id', 'period', 'cumulative_mins']).reset_index(drop=True)

        # Rebuild events dataframe
        events_out = pd.concat([events_out, match_events_and_carries])

    return events_out



#%%

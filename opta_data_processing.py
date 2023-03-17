import json
import pandas as pd
import numpy as np
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter import ttk


import json
from ast import literal_eval



#import analysis_tool.whoscored_data_engineering as wcde
import analysis_tools.whoscored_custom_events as ws_custom_events




path = "Data/unprocess data/"
filename = input("Enter file name: ")
file_with_path = path + filename
f = open(file_with_path, encoding='utf-8')
data = json.load(f)








event_data = data['matchCentreData']


player_id = event_data['playerIdNameDictionary']
player_id

match_id = data['matchId']

home_data = data['matchCentreData']['home']

home_team_name =  event_data['home']['name']



away_team_name =  event_data['away']['name']


home_teamID = event_data['home']['teamId']


away_teamID = event_data['away']['teamId']


events = event_data['events']


event_data = data['matchCentreData']


player_id = event_data['playerIdNameDictionary']
player_id

match_id = data['matchId']

home_data = data['matchCentreData']['home']

home_team_name = event_data['home']['name']


away_team_name = event_data['away']['name']


home_teamID = event_data['home']['teamId']


away_teamID = event_data['away']['teamId']


events = event_data['events']



events = pd.DataFrame(events)
events

events['match_id'] = match_id

match_string = home_team_name +  " "  + "-" + " " + away_team_name

events['match_string'] = match_string

# Create a dictionary mapping team IDs to team names
team_name = {
    home_teamID: home_team_name,
    away_teamID: away_team_name
}
events['team_name'] = events['teamId'].map(team_name)

events


# Create a dictionary mapping team IDs to team names
team_dict = {home_teamID: 'Home', away_teamID: 'Away'}

# Add a column to the event data to indicate whether the corresponding data is for the home team or away team
events['Venue'] = events['teamId'].map(team_dict)


# Load the event ID json file
with open('data/event_id.json') as f:
    event_id = {value: key for key, value in json.load(f).items()}

# Replace the values in the 'satisfiedEventsTypes' column with the corresponding event type labels
events['satisfiedEventsTypes'] = events['satisfiedEventsTypes'].apply(lambda x: [event_id.get(i) for i in x])

# Extract the outcomeType values using the apply() function and a lambda function
events['outcomeType'] = events['outcomeType'].apply(lambda x: x['displayName'])

events['type'] = events['type'].apply(lambda x: x['displayName'])


events['period'] = events['period'].apply(lambda x: x['displayName'])
events

events['qualifiers'] = events['qualifiers'].apply(lambda x: [{item['type']['displayName']: item.get('value', True)} for item in x])


events = events.rename(columns={'type': 'event_type'})

events['period'] = events['period'].map({'FirstHalf': 1, 'SecondHalf': 2})


events = wcde.cumulative_match_mins(events)

events = ws_custom_events.insert_ball_carries(events)

events = ws_custom_events.get_xthreat(events)


for i, row in events.iterrows():
    player_id_val = row['playerId']
    if not pd.isnull(player_id_val):
        player_name = player_id.get(str(int(player_id_val)), 'Unknown')
        events.at[i, 'playerName'] = player_name
    else:
        events.at[i, 'playerName'] = 'Unknown'


filename = f'Data/{home_team_name}_vs_{away_team_name}.csv'
events.to_csv(filename, index=False)










#%%

########################################################################################################################
#
#
# Goblin Net: Neural Networks that Predict the Outcome of competitive Team Fortress 2 Matches
#
# get_data
#
# Module for getting data from logs.tf logs and preparing it to be fed into the goblin
#
# Authors / Contributors:
# Chandler Calkins
#
#
########################################################################################################################

# for getting data from web pages
from urllib.request import urlopen
# used for getting around needing to verify logs.tf's ssl certificate to get data from the website
import ssl
# used for getting json data from logs.tf and turning into a python dictionary
import json
# used for handling date data from match logs
from datetime import datetime
# used for changing the time zone of the retrieved match times
import pytz
# used for putting data into arrays to be fed into the goblin
import numpy as np

def refresh_data(log_ids):
	players, maps, dates, weekdays = refresh_inputs(log_ids)
	
	print(players)
	print(maps)
	print(dates)
	print(weekdays)

def refresh_inputs(log_ids):
	input_players = np.empty((0, 12), str)
	maps = np.empty((0, 1), str)
	dates = np.empty((0, 3), int)
	weekdays = np.empty((0, 1), str)
	for log_id in log_ids:
		# Read json data of log

		# get string of log id
		log_id_str = str(log_id)
		# get parts of necessary urls
		log_url_p1 = "https://logs.tf/"
		log_url_p2 = "json/"
		# concatenate to form json url and original log page url
		log_json_url = log_url_p1 + log_url_p2 + log_id_str
		log_url = log_url_p1 + log_id_str
		# create context that doesn't require ssl certificate verification when requesting from website
		context = ssl._create_unverified_context()
		# request data from json file of log
		response = urlopen(log_json_url, context=context)
		# turn data from json file into dictionary
		data = json.loads(response.read())

		# Make sure all keys needed from dictionary are present

		# make sure there is version data
		if "version" not in data:
			print(f"Log version missing from log {log_id}")
			quit()
		# make sure there is player data
		if "players" not in data:
			print(f"Player data missing from log {log_id}")
			quit()
		# make sure there is name data
		if "names" not in data:
			print(f"Names are missing from log {log_id}")
			quit()
		# make sure there is extra info data
		if "info" not in data:
			print(f"Info field missing from log {log_id}")
			quit()
		# make sure the map was recorded
		if "map" not in data["info"]:
			print(f"Map missing from log {log_id}")
			quit()
		# make sure there was a recorded date
		if "date" not in data["info"]:
			print(f"Date missing from log {log_id}")
			quit()

		# Collect input data

		# make sure log version is correct
		if data["version"] != 3:
			print(f"Log version is not 3 in log {log_id}")
			quit()

		# get player data
		player_data = data["players"]
		# make sure there are 12 players
		if len(player_data) != 12:
			print(f"Player count is not 12 in log {log_id}")
			quit()

		# lists of players playing each class on each team that will be concatenated later
		red_scouts = []
		red_soldiers = []
		red_demo = []
		red_med = []
		blu_scouts = []
		blu_soldiers = []
		blu_demo = []
		blu_med = []
		# loop through each player and put them in the correct list for their team and class
		for sid3 in player_data:
			player = player_data[sid3]
			key_team = "team"
			key_stats = "class_stats"
			key_type = "type"
			if key_team not in player:
				print(f"Team info missing for player {sid3} in log {log_id}")
				quit()
			if key_stats not in player or len(player[key_stats]) < 1:
				print(f"Class stats missing for player {sid3} in log {log_id}")
				quit()
			if key_type not in player[key_stats][0]:
				print(f"Class type missing for player {sid3} in log {log_id}")
				quit()
			
			if player[key_team] == "Red":
				if player[key_stats][0][key_type] == "scout":
					red_scouts.append(sid3)
				elif player[key_stats][0][key_type] == "soldier":
					red_soldiers.append(sid3)
				elif player[key_stats][0][key_type] == "demoman":
					red_demo.append(sid3)
				elif player[key_stats][0][key_type] == "medic":
					red_med.append(sid3)
				else:
					print(f"Primary class if non-sixes meta for player {sid3} in log {log_id}")
					quit()
			elif player[key_team] == "Blue":
				if player[key_stats][0][key_type] == "scout":
					blu_scouts.append(sid3)
				elif player[key_stats][0][key_type] == "soldier":
					blu_soldiers.append(sid3)
				elif player[key_stats][0][key_type] == "demoman":
					blu_demo.append(sid3)
				elif player[key_stats][0][key_type] == "medic":
					blu_med.append(sid3)
				else:
					print(f"Primary class if non-sixes meta for player {sid3} in log {log_id}")
					quit()
			else:
				print(f"Unknown team for player {sid3} in log {log_id}")
				quit()

		if len(red_scouts) != 2:
			print(f"Not 2 scouts on red team in log {log_id}")
			quit()
		elif len(red_soldiers) != 2:
			print(f"Not 2 soldiers on red team in log {log_id}")
			quit()
		elif len(red_demo) != 1:
			print(f"Not 1 demo on red team in log {log_id}")
			quit()
		elif len(red_med) != 1:
			print(f"Not 1 med on red team in log {log_id}")
			quit()
		elif len(blu_scouts) != 2:
			print(f"Not 2 scouts on blu team in log {log_id}")
			quit()
		elif len(blu_soldiers) != 2:
			print(f"Not 2 soldiers on blu team in log {log_id}")
			quit()
		elif len(blu_demo) != 1:
			print(f"Not 1 demo on blu team in log {log_id}")
			quit()
		elif len(blu_med) != 1:
			print(f"Not 1 med on blu team in log {log_id}")
			quit()

		# turn steam id 3s into an array
		player_sid3s = np.array(red_scouts + red_soldiers + red_demo + red_med + blu_scouts + blu_soldiers + blu_demo + blu_med)
		# get all player names
		player_names = data["names"]
		# make sure there are 12 player names
		if len(player_names) != 12:
			print(f"There aren't 12 player names in log {log_id}")
			quit()

		# print out each player name and steam id 3
		for sid3 in player_sid3s:
			# make sure all of the steam id 3s have a player name linked to them
			if sid3 not in player_names:
				print(f"Player {sid3} doesn't have a name in log {log_id}")
				quit()

		# get map name
		map_name = np.array(data["info"]["map"], ndmin=1)

		# get match date in us eastern timezone (since that's the standard timezone for tf2 in na)
		match_datetime = datetime.fromtimestamp(data["info"]["date"], tz=pytz.timezone("US/Eastern"))
		# get match year, month, day, and day of the week
		match_date = np.array([match_datetime.year, match_datetime.month, match_datetime.day])
		match_weekday = np.array(match_datetime.strftime("%A"), ndmin=1)

		input_players = np.append(input_players, player_sid3s)
		maps = np.append(maps, map_name)
		dates = np.append(dates, match_date)
		weekdays = np.append(weekdays, match_weekday)
	
	return input_players, maps, dates, weekdays

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
	# arrays of input data to collect from each log
	input_players = np.empty((0, 12), str)
	maps = np.empty((0, 1), str)
	dates = np.empty((0, 3), int)
	weekdays = np.empty((0, 1), str)

	# arrays of output data to collect from each log
	scores = np.empty((0, 2), int)
	stats = np.empty((0, 61), int)

	# collect data from each log
	for log_id in log_ids:
		# Read json data of log

		# get string of log id
		log_id_str = str(log_id)
		# get parts of necessary urls
		log_url_p1 = "https://logs.tf/"
		log_url_p2 = "json/"
		# concatenate to form json url and original log page url
		log_json_url = log_url_p1 + log_url_p2 + log_id_str
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
			continue
		# make sure there team data
		if "teams" not in data:
			print(f"Team data missing from log {log_id}")
			continue
		# make sure red and blu teams aren't missing
		if "Red" not in data["teams"]:
			print(f"Red team missing from log {log_id}")
			continue
		if "Blue" not in data["teams"]:
			print(f"Blu team missing from log {log_id}")
			continue
		# make sure scores aren't missing from red and blu teams
		if "score" not in data["teams"]["Red"]:
			print(f"Score missing from red team in log {log_id}")
			continue
		if "score" not in data["teams"]["Blue"]:
			print(f"Score missing from blu team in log {log_id}")
			continue
		# make sure there is player data
		if "players" not in data:
			print(f"Player data missing from log {log_id}")
			continue
		# make sure there is name data
		if "names" not in data:
			print(f"Names are missing from log {log_id}")
			continue
		# make sure there is extra info data
		if "info" not in data:
			print(f"Info field missing from log {log_id}")
			continue
		# make sure there is match length data
		if "total_length" not in data["info"]:
			print(f"Match length missing from log {log_id}")
			continue
		# make sure the map was recorded
		if "map" not in data["info"]:
			print(f"Map missing from log {log_id}")
			continue
		# make sure there was a recorded date
		if "date" not in data["info"]:
			print(f"Date missing from log {log_id}")
			continue

		# make sure log version is correct
		if data["version"] != 3:
			print(f"Log version is not 3 in log {log_id}")
			continue

		# collect the scores of each team
		score = np.array([data["teams"]["Red"]["score"], data["teams"]["Blue"]["score"]])
		# collect the length of the match
		match_length = [data["info"]["total_length"]]

		# get player data
		player_data = data["players"]
		# make sure there are 12 players
		if len(player_data) != 12:
			print(f"Player count is not 12 in log {log_id}")
			continue

		# lists of players playing each class on each team that will be concatenated later
		red_scouts = []
		red_soldiers = []
		red_demo = []
		red_med = []
		blu_scouts = []
		blu_soldiers = []
		blu_demo = []
		blu_med = []
		# lists of stats for each player on each team
		red_scouts_stats = []
		red_soldiers_stats = []
		red_demo_stats = []
		red_med_stats = []
		blu_scouts_stats = []
		blu_soldiers_stats = []
		blu_demo_stats = []
		blu_med_stats = []

		# flag to tell function to drop this log if there was en error encountered inside the loop
		error = False

		# loop through each player and put them in the correct list for their team and class
		for sid3 in player_data:
			player = player_data[sid3]
			# keys names for each statistic
			key_team = "team"
			key_stats = "class_stats"
			key_type = "type"
			key_kills = "kills"
			key_assists = "assists"
			key_deaths = "deaths"
			key_dmg = "dmg"
			key_dt = "dt"
			# make sure the team value exists for this player
			if key_team not in player:
				print(f"Team info missing for player {sid3} in log {log_id}")
				error = True
				break
			# make sure the class_stats value exists for this player
			if key_stats not in player or len(player[key_stats]) < 1:
				print(f"Class stats missing for player {sid3} in log {log_id}")
				error = True
				break
			# make sure the type of class the player played exists for this player
			if key_type not in player[key_stats][0]:
				print(f"Class type missing for player {sid3} in log {log_id}")
				error = True
				break
			# make sure the kills were recorded
			if key_kills not in player:
				print(f"Kills missing from player {sid3} in log {log_id}")
				error = True
				break
			# make sure the assists were recorded
			if key_assists not in player:
				print(f"Assists missing from player {sid3} in log {log_id}")
				error = True
				break
			# make sure the deaths were recorded
			if key_deaths not in player:
				print(f"Deaths missing from player {sid3} in log {log_id}")
				error = True
				break
			# make sure the damage was recorded
			if key_dmg not in player:
				print(f"Damage missing from player {sid3} in log {log_id}")
				error = True
				break
			# make sure the damage taken was recorded
			if key_dt not in player:
				print(f"Damage taken missing from player {sid3} in log {log_id}")
				error = True
				break

			# if the player was on the red team
			if player[key_team] == "Red":
				# determine which class the player played
				if player[key_stats][0][key_type] == "scout":
					# place them in the correct team / class list for sorting
					red_scouts.append(sid3)
					# add their stats to the correct team / class list for sorting
					red_scouts_stats.append(player[key_kills])
					red_scouts_stats.append(player[key_assists])
					red_scouts_stats.append(player[key_deaths])
					red_scouts_stats.append(player[key_dmg])
					red_scouts_stats.append(player[key_dt])
				elif player[key_stats][0][key_type] == "soldier":
					red_soldiers.append(sid3)
					red_soldiers_stats.append(player[key_kills])
					red_soldiers_stats.append(player[key_assists])
					red_soldiers_stats.append(player[key_deaths])
					red_soldiers_stats.append(player[key_dmg])
					red_soldiers_stats.append(player[key_dt])
				elif player[key_stats][0][key_type] == "demoman":
					red_demo.append(sid3)
					red_demo_stats.append(player[key_kills])
					red_demo_stats.append(player[key_assists])
					red_demo_stats.append(player[key_deaths])
					red_demo_stats.append(player[key_dmg])
					red_demo_stats.append(player[key_dt])
				elif player[key_stats][0][key_type] == "medic":
					# key names for each medic statistic
					key_heals = "heal"
					key_ubers = "ubers"
					key_drops = "drops"
					# make sure keys for medic stats exist for each class
					if key_heals not in player:
						print(f"Heals missing from from medic {sid3} in log {log_id}")
						error = True
						break
					if key_ubers not in player:
						print(f"Ubers missing from medic {sid3} in log {log_id}")
						error = True
						break
					if key_drops not in player:
						print(f"Drops missing from medic {sid3} in log {log_id}")
						error = True
						break
					# place them in the correct team / class list for sorting
					red_med.append(sid3)
					# add their stats to the correct team / class list for sorting
					red_med_stats.append(player[key_kills])
					red_med_stats.append(player[key_assists])
					red_med_stats.append(player[key_deaths])
					red_med_stats.append(player[key_dmg])
					red_med_stats.append(player[key_dt])
				# if the class isn't a meta sixes class
				else:
					print(f"Primary class is non-sixes meta for player {sid3} in log {log_id}")
					error = True
					break
			# if the player was on the blu team
			elif player[key_team] == "Blue":
				# determine which class the player played
				if player[key_stats][0][key_type] == "scout":
					# place them in the correct team / class list for sorting
					blu_scouts.append(sid3)
					# add their stats to the correct team / class list for sorting
					blu_scouts_stats.append(player[key_kills])
					blu_scouts_stats.append(player[key_assists])
					blu_scouts_stats.append(player[key_deaths])
					blu_scouts_stats.append(player[key_dmg])
					blu_scouts_stats.append(player[key_dt])
				elif player[key_stats][0][key_type] == "soldier":
					blu_soldiers.append(sid3)
					blu_soldiers_stats.append(player[key_kills])
					blu_soldiers_stats.append(player[key_assists])
					blu_soldiers_stats.append(player[key_deaths])
					blu_soldiers_stats.append(player[key_dmg])
					blu_soldiers_stats.append(player[key_dt])
				elif player[key_stats][0][key_type] == "demoman":
					blu_demo.append(sid3)
					blu_demo_stats.append(player[key_kills])
					blu_demo_stats.append(player[key_assists])
					blu_demo_stats.append(player[key_deaths])
					blu_demo_stats.append(player[key_dmg])
					blu_demo_stats.append(player[key_dt])
				elif player[key_stats][0][key_type] == "medic":
					# key names for each medic statistic
					key_heals = "heal"
					key_ubers = "ubers"
					key_drops = "drops"
					# make sure keys for medic stats exist for each class
					if key_heals not in player:
						print(f"Heals missing from from medic {sid3} in log {log_id}")
						error = True
						break
					if key_ubers not in player:
						print(f"Ubers missing from medic {sid3} in log {log_id}")
						error = True
						break
					if key_drops not in player:
						print(f"Drops missing from medic {sid3} in log {log_id}")
						error = True
						break
					# place them in the correct team / class list for sorting
					blu_med.append(sid3)
					# add their stats to the correct team / class list for sorting
					blu_med_stats.append(player[key_kills])
					blu_med_stats.append(player[key_assists])
					blu_med_stats.append(player[key_deaths])
					blu_med_stats.append(player[key_dmg])
					blu_med_stats.append(player[key_dt])
				# if the class isn't a meta sixes class
				else:
					print(f"Primary class is non-sixes meta for player {sid3} in log {log_id}")
					error = True
					break
			# if the team wasn't recognized
			else:
				print(f"Unknown team for player {sid3} in log {log_id}")
				error = True
				break
		
		# if there was an error encountered inside the loop
		if error:
			continue

		# make sure there are the correct amount of players for each class on each team
		if len(red_scouts) != 2:
			print(f"Not 2 scouts on red team in log {log_id}")
			continue
		elif len(red_soldiers) != 2:
			print(f"Not 2 soldiers on red team in log {log_id}")
			continue
		elif len(red_demo) != 1:
			print(f"Not 1 demo on red team in log {log_id}")
			continue
		elif len(red_med) != 1:
			print(f"Not 1 med on red team in log {log_id}")
			continue
		elif len(blu_scouts) != 2:
			print(f"Not 2 scouts on blu team in log {log_id}")
			continue
		elif len(blu_soldiers) != 2:
			print(f"Not 2 soldiers on blu team in log {log_id}")
			continue
		elif len(blu_demo) != 1:
			print(f"Not 1 demo on blu team in log {log_id}")
			continue
		elif len(blu_med) != 1:
			print(f"Not 1 med on blu team in log {log_id}")
			continue

		# turn steam id 3s into an array
		player_sid3s = np.array(red_scouts + red_soldiers + red_demo + red_med + blu_scouts + blu_soldiers + blu_demo + blu_med)
		# turn stats into an array
		match_stats = np.array(match_length + red_scouts_stats + red_soldiers_stats + red_demo_stats + red_med_stats + blu_scouts_stats + blu_soldiers_stats + blu_demo_stats + blu_med_stats)

		# get all player names
		player_names = data["names"]
		# make sure there are 12 player names
		if len(player_names) != 12:
			print(f"There aren't 12 player names in log {log_id}")
			continue

		# flag to tell function to drop this log if there was en error encountered inside the loop
		error = False

		# print out each player name and steam id 3
		for sid3 in player_sid3s:
			# make sure all of the steam id 3s have a player name linked to them
			if sid3 not in player_names:
				print(f"Player {sid3} doesn't have a name in log {log_id}")
				error = True
				break

		# if there was an error encountered inside the loop
		if error:
			continue

		# get map name
		map_name = np.array(data["info"]["map"], ndmin=1)

		# get match date in us eastern timezone (since that's the standard timezone for tf2 in na)
		match_datetime = datetime.fromtimestamp(data["info"]["date"], tz=pytz.timezone("US/Eastern"))
		# get match year, month, day, and day of the week
		match_date = np.array([match_datetime.year, match_datetime.month, match_datetime.day])
		match_weekday = np.array(match_datetime.strftime("%A"), ndmin=1)

		# add all of the data collected from this match to the collective data arrays
		scores = np.append(scores, score)
		stats = np.append(stats, match_stats)

		input_players = np.append(input_players, player_sid3s)
		maps = np.append(maps, map_name)
		dates = np.append(dates, match_date)
		weekdays = np.append(weekdays, match_weekday)
	
	print("Inputs:")
	print(input_players)
	print(maps)
	print(dates)
	print(weekdays)
	print("Outputs:")
	print(scores)
	print(stats)

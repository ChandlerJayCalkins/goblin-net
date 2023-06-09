########################################################################################################################
#
#
# Goblin Net: Neural Networks that Predict the Outcome of competitive Team Fortress 2 Matches
#
# collect_log_data
#
# Module for collecting data from logs.tf logs and preparing it to be fed into the goblin
#
# Authors / Contributors:
# Chandler Calkins
#
#
########################################################################################################################

# used for getting steam ids from steam profile urls
# pip install python-steam-api
from steam import Steam
from decouple import config
# used for reading html pages
# pip install beautifulsoup4 / conda install beautifulsoup4
from bs4 import BeautifulSoup
# used for getting data from web pages
from urllib.request import urlopen
# used for getting around needing to verify logs.tf's ssl certificate to get data from the website
import ssl
# used for getting json data from logs.tf and turning into a python dictionary
import json
# used for handling date data from match logs
from datetime import datetime
# used for making sure files and folders exist
import os
# used for changing the time zone of the retrieved match times
import pytz
# used for putting data into arrays to be fed into the goblin
import numpy as np
# used for outputting data to a csv file
import pandas as pd
# used for one hot encoding data
from keras.utils import to_categorical
from sklearn.preprocessing import OrdinalEncoder

# name of data folder
data_path = "../data"
# data file extension
file_ext = ".csv"

# name of steam profile data file
profile_data_file = "profiles"
# name of data file containing log ids of logs to extract data from
log_data_file = "logs"
# name of data file containing steamid3s of players inputted from profiles file
sid3s_data_file = "SteamID3s"
# name of data file containing log ids of valid logs that will be used
used_logs_file = "used_logs"

# names of input data files
player_data_file = "players"
gamemode_data_file = "gamemodes"
maps_data_file = "maps"
dates_data_file = "dates"
weekdays_data_file = "weekdays"

# names of output data files
scores_data_file = "scores"
stats_data_file = "stats"

# names of data files that have been prepared to be fed into the goblin
inputs_data_file = "inputs"
outputs_data_file = "outputs"

# path to steam profile data file
profile_data_path = f"{data_path}/{profile_data_file}{file_ext}"
# path to log data file
log_data_path = f"{data_path}/{log_data_file}{file_ext}"
# path to steamid3 data file
sid3_data_path = f"{data_path}/{sid3s_data_file}{file_ext}"
# path to used logs data file
used_logs_path = f"{data_path}/{used_logs_file}{file_ext}"

# paths to input data files
player_data_path = f"{data_path}/{player_data_file}{file_ext}"
gamemode_data_path = f"{data_path}/{gamemode_data_file}{file_ext}"
maps_data_path = f"{data_path}/{maps_data_file}{file_ext}"
dates_data_path = f"{data_path}/{dates_data_file}{file_ext}"
weekdays_data_path = f"{data_path}/{weekdays_data_file}{file_ext}"

# paths to output data files
scores_data_path = f"{data_path}/{scores_data_file}{file_ext}"
stats_data_path = f"{data_path}/{stats_data_file}{file_ext}"

# paths to data files that have been prepared to be fed into the goblin
inputs_data_path = f"{data_path}/{inputs_data_file}{file_ext}"
outputs_data_path = f"{data_path}/{outputs_data_file}{file_ext}"

# parts of logs.tf urls
log_tf_url = "https://logs.tf/"
json_log_url = "json/"
profile_log_url = "profile/"

# create context that doesn't require ssl certificate verification when requesting from website
# since logs.tf's ssl certificate is expired apparently lol
unverified_context = ssl._create_unverified_context()

# used for converting steam community id to steam id 3
# found from https://gist.github.com/bcahue/4eae86ae1d10364bb66d
def commid_to_steamid3(commid):
	# used for converting steam ids
	steamid64ident = 76561197960265728
	# calculate steamid3 value
	steamidacct = int(commid) - steamid64ident
    # return full steamid3
	return "[U:1:" + str(steamidacct) + "]"

# reads from a file of steam profiles and returns the log ids of the last few pages of each of their logs
# returns an array of the log ids and steamid3s of the players it read
def get_logs(pages, verbose=True):
	if verbose:
		print("Getting logs from list of players...")

	# if pages isn't valid
	if type(pages) is not int or pages < 1:
		raise ValueError("Pages parameter must be a positive integer.")
	# if there isn't a folder for the data
	if not os.path.isdir(data_path):
		print("ERROR: Missing data folder.")
		exit(1)
	# if any of the data files are missing
	if not os.path.isfile(profile_data_path):
		print("ERROR: Missing steam profile data file.\
		Must make list of steam profiles to read called 'profiles.csv' in the 'data' folder.")
		exit(1)
	
	# read list of steam profile urls
	try:
		profiles = np.unique(np.array(pd.read_csv(profile_data_path, header=None)))
	# if there aren't any profiles in the data file
	except pd.errors.EmptyDataError:
		print("ERROR: Empty profile data file. Be sure to put a list of Steam profile links in data/profiles.csv")
		exit(1)
	
	profile_count = len(profiles)
	if verbose:
		print(f"Collected {profile_count} Steam profiles...")

	# retrive steam api key to connect to steam api
	# make sure to create a file called ".env" and put it in the root directory of this repo,
	# and in that file put "STEAM_API_KEY=*your steam api key*"
	# you can get a steam api key from https://steamcommunity.com/dev/apikey
	try:
		steam_api_key = config("STEAM_API_KEY")
	except:
		print("No Steam API key was found. Make a .env file and put yours in there in the right format. \
			You can get a person Steam API key from https://steamcommunity.com/dev/apikey.")
		exit(1)
	# connect to steam api
	steam = Steam(steam_api_key)
	# prefix for steam url
	https_steam_prefix = "https://steamcommunity.com/"
	http_steam_prefix = "http://steamcommunity.com/"

	# array of all of the log ids
	log_ids = np.array([], dtype=str)
	# array of all steamid3s
	sid3s = np.array([], dtype=str)
	# counter to keep track of how many profiles have been read
	counter = 0

	# for each steam profile url that was read from the file
	for profile in profiles:
		counter += 1
		print()
		# Get the steam id of the profile

		steam_id = None
		# if the url already contains the steam id, then just get the id from the url
		if profile.startswith(https_steam_prefix + "profiles/") or profile.startswith(http_steam_prefix + "profiles/"):
			steam_id = profile[36:]
		# if this is a custom url
		elif profile.startswith(https_steam_prefix + "id/") or profile.startswith(http_steam_prefix + "id/"):
			# get the name from the custom url
			custom_url = profile[30:-1]
			# get user info from steam api
			steam_user = steam.users.search_user(custom_url)
			# if there was no match found or there was an error in retrieving the steam info of the user
			if steam_user == "No match" or type(steam_user) is not dict:
				print(f"\nERROR: User not found from {profile}")
				exit(1)
			# if the player key is missing from the steam user dict
			if "player" not in steam_user:
				print(f"\nERROR: 'player' key missing from steam info retrieved from {profile}")
				exit(1)
			# if the steamid key is missing from the retrieved steam info
			if "steamid" not in steam_user["player"]:
				print(f"\nERROR: Steam ID missing from steam info retrieved from {profile}")
				exit(1)
			# get the steam id of the user
			steam_id = steam_user["player"]["steamid"]
		else:
			print(f"\nERROR: Could not identify steam profile format from {profile}")
			exit(1)

		# add steamid3 to list of steamid3s
		sid3s = np.append(sid3s, commid_to_steamid3(steam_id))

		# collect list of log ids from last few pages of player's logs.tf profile
		for page in range(1, pages + 1):
			if verbose:
				print(f"\r[{counter}/{profile_count}]: {profile} page {page}...", end="")
			
			# request page of player's log profile
			response = None
			response_success = False
			# in case requesting http response fails on first try, loop until request succeeds
			while not response_success:
				try:
					response = urlopen(log_tf_url + profile_log_url + steam_id + f"?p={page}",\
				    	context=unverified_context)
					response_success = True
				except:
					print(f"\nHTTP Error from log profile page {log_tf_url}{profile_log_url}{steam_id}?p={page},\
						trying again...")

			# if there aren't any pages left in the player's logs (defaults back to logs.tf home page)
			if response.url == log_tf_url:
				if verbose:
					print(f"\nNo more logs on page {page} for {profile}")
				break

			# create html parser to find log ids from web page
			soup = BeautifulSoup(response.read(), "html.parser")
			# find each <tr> element in the page (they contain the log ids)
			trs = soup.find_all("tr", id=True)
			# if no <tr> elements were found, don't check for a next page
			if len(trs) < 1:
				if verbose:
					print(f"\nNo more logs on page {page} for {profile}")
				break

			# loop through each tr element to find all of the log ids
			for tr in soup.find_all("tr", id=True):
				# get the html element id of the <tr> element
				id = tr["id"]
				# if the html element id of the <tr> element isn't of the form "log_*log id*",
				# then it doesn't contain a log id
				if not id.startswith("log_"):
					continue
				# extract the log id from the html element id
				id_index = id.index("_") + 1
				log_id = id[id_index:]
				# append the log id to the array of log ids
				log_ids = np.append(log_ids, log_id)
	
	# make sure there are no duplicate logs
	log_ids = np.unique(log_ids)

	# if there isn't already a folder for the data, create one
	if not os.path.isdir(data_path):
		os.mkdir(data_path)

	# output the log ids to a csv file
	df_logs = pd.DataFrame(log_ids)
	df_logs.to_csv(log_data_path, header=["Log ID"], index=False)

	if verbose:
		print(f"\nStored list of {len(log_ids)} log ids to {log_data_path}")
	
	# output the steamid3s to a csv file
	df_sid3s = pd.DataFrame(sid3s)
	df_sid3s.to_csv(sid3_data_path, header=["SteamID3s"], index=False)
	
	# return the log ids and the steamid3s
	return log_ids, sid3s

# returns an array of log ids that were retrieved from get_logs()
def read_log_ids():
	# if there isn't a folder for the data
	if not os.path.isdir(data_path):
		raise FileNotFoundError("Missing data folder.")
	# if the log ids are missing
	if not os.path.isfile(log_data_path):
		raise FileNotFoundError("Missing log ID data file")
	
	# read the log id file and return an array of the log ids
	return np.array(pd.read_csv(log_data_path), dtype=str).flatten()

# returns an array of steamid3s that were retrieved from get_logs()
def read_sid3s():
	# if there isn't a folder for the data
	if not os.path.isdir(data_path):
		raise FileNotFoundError("Missing data folder.")
	# if the steamid3s are missing
	if not os.path.isfile(sid3_data_path):
		raise FileNotFoundError("Missing SteamID3 data file")
	
	# read the steamid3 file and return an array of the steamid3s
	return np.array(pd.read_csv(sid3_data_path), dtype=str).flatten()

# collects data from log files of list of log ids and puts the data in csv files in the data folder
# returns the number of valid logs that it stored data from, along with numpy arrays the data that was collected
def fetch_log_data(log_ids, sid3s, verbose=True):
	if verbose:
		print("Extracting log data and weeding out invalid logs...")

	# array of ids of logs that were actually used
	used_logs = np.array([], dtype=str)

	# arrays of input data to collect from each log
	players = np.empty((0, 12), str)
	gamemodes = np.empty((0, 1), int)
	maps = np.empty((0, 1), str)
	dates = np.empty((0, 3), int)
	weekdays = np.empty((0, 1), str)

	# arrays of output data to collect from each log
	scores = np.empty((0, 2), int)
	stats = np.empty((0, 67), int)

	# number of logs to check
	log_count = len(log_ids)
	# counter for how many logs have been checked
	counter = 0
	# collect data from each log
	for log_id in log_ids:
		counter += 1
		if verbose:
			print(f"\r[{counter}/{log_count}]: ", end="")
		# Read json data of log

		# concatenate to form json url and original log page url
		log_json_url = log_tf_url + json_log_url + log_id

		# request data from json file of log
		response = None
		response_success = False
		data = None
		# in case requesting http response fails on first try, loop until request succeeds
		while not response_success:
			try:
				response = urlopen(log_json_url, context=unverified_context)
				# turn data from json file into dictionary
				data = json.loads(response.read())
				response_success = True
			except:
				print(f"HTTP Error from log id {log_id}, trying again...")

		# Check that all necessary dictionary keys exist

		# make sure there is version data
		if "version" not in data:
			if verbose:
				print(f"Log version missing from log {log_id}")
			continue
		# make sure there team data
		if "teams" not in data:
			if verbose:
				print(f"Team data missing from log {log_id}")
			continue
		# make sure red and blu teams aren't missing
		if "Red" not in data["teams"]:
			if verbose:
				print(f"Red team missing from log {log_id}")
			continue
		if "Blue" not in data["teams"]:
			if verbose:
				print(f"Blu team missing from log {log_id}")
			continue
		# make sure scores aren't missing from red and blu teams
		if "score" not in data["teams"]["Red"]:
			if verbose:
				print(f"Score missing from red team in log {log_id}")
			continue
		if "score" not in data["teams"]["Blue"]:
			if verbose:
				print(f"Score missing from blu team in log {log_id}")
			continue
		# make sure there is player data
		if "players" not in data:
			if verbose:
				print(f"Player data missing from log {log_id}")
			continue
		# make sure there is name data
		if "names" not in data:
			if verbose:
				print(f"Names are missing from log {log_id}")
			continue
		# make sure there is extra info data
		if "info" not in data:
			if verbose:
				print(f"Info field missing from log {log_id}")
			continue
		# make sure there is match length data
		if "total_length" not in data["info"]:
			if verbose:
				print(f"Match length missing from log {log_id}")
			continue
		# make sure the map was recorded
		if "map" not in data["info"]:
			if verbose:
				print(f"Map missing from log {log_id}")
			continue
		# make sure there was a recorded date
		if "date" not in data["info"]:
			if verbose:
				print(f"Date missing from log {log_id}")
			continue

		# Validate and extract necessary data from log

		# make sure log version is correct
		if data["version"] != 3:
			if verbose:
				print(f"Log version is not 3 in log {log_id}")
			continue

		# make sure scores are valid
		if data["teams"]["Red"]["score"] > 5:
			if verbose:
				print(f"Red team score is too high (>5) in log {log_id}")
			continue
		if data["teams"]["Red"]["score"] < 0:
			if verbose:
				print(f"Red team score is too low (<0) in log {log_id}")
			continue
		if data["teams"]["Blue"]["score"] > 5:
			if verbose:
				print(f"Blu team score is too high (>5) in log {log_id}")
			continue
		if data["teams"]["Blue"]["score"] < 0:
			if verbose:
				print(f"Blu team score is too low (<0) in log {log_id}")
			continue
		# collect the scores of each team
		score = np.array([data["teams"]["Red"]["score"], data["teams"]["Blue"]["score"]])

		# collect the length of the match
		match_length = np.array([data["info"]["total_length"]])

		# find out if map is koth or control points
		if "_" not in data["info"]["map"]:
			if verbose:
				print(f"Gamemode not found from map name in log {log_id}")
			continue
		# get the gamemode from the substring that comes before the first underscore in the map name
		gamemode = np.array([data["info"]["map"][:data["info"]["map"].index("_")]])

		# get map name
		map_name = np.array(data["info"]["map"], ndmin=1)

		# get player data
		player_data = data["players"]
		# make sure there are 12 players
		if len(player_data) != 12:
			if verbose:
				print(f"Player count is not 12 in log {log_id}")
			continue

		# adds player stats to numpy arrays
		def _add_class_stats(player_stats, player, *keys):
			# adds stats to stats list
			for key in keys:
				player_stats = np.append(player_stats, player[key])
			
			return player_stats

		# arrays of players playing each class on each team that will be concatenated later
		red_scouts = np.array([], dtype=str)
		red_soldiers = np.array([], dtype=str)
		red_demo = np.array([], dtype=str)
		red_med = np.array([], dtype=str)
		blu_scouts = np.array([], dtype=str)
		blu_soldiers = np.array([], dtype=str)
		blu_demo = np.array([], dtype=str)
		blu_med = np.array([], dtype=str)
		# arrays of stats for each player on each team
		red_scouts_stats = np.array([], dtype=str)
		red_soldiers_stats = np.array([], dtype=str)
		red_demo_stats = np.array([], dtype=str)
		red_med_stats = np.array([], dtype=str)
		blu_scouts_stats = np.array([], dtype=str)
		blu_soldiers_stats = np.array([], dtype=str)
		blu_demo_stats = np.array([], dtype=str)
		blu_med_stats = np.array([], dtype=str)

		# flag to tell function to drop this log if there was en error encountered inside the loop
		error = False

		# loop through each player and put them in the correct list for their team and class
		for sid3 in player_data:
			# make sure this is a player that was inputted to be trained on by the neural net
			if sid3 not in sid3s:
				if verbose:
					print(f"Player {sid3} in log {log_id} not in list of players to train on")
				error = True
				break

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
				if verbose:
					print(f"Team info missing for player {sid3} in log {log_id}")
				error = True
				break
			# make sure the class_stats value exists for this player
			if key_stats not in player or len(player[key_stats]) < 1:
				if verbose:
					print(f"Class stats missing for player {sid3} in log {log_id}")
				error = True
				break
			# make sure the type of class the player played exists for this player
			if key_type not in player[key_stats][0]:
				if verbose:
					print(f"Class type missing for player {sid3} in log {log_id}")
				error = True
				break
			# make sure the kills were recorded
			if key_kills not in player:
				if verbose:
					print(f"Kills missing from player {sid3} in log {log_id}")
				error = True
				break
			# make sure the assists were recorded
			if key_assists not in player:
				if verbose:
					print(f"Assists missing from player {sid3} in log {log_id}")
				error = True
				break
			# make sure the deaths were recorded
			if key_deaths not in player:
				if verbose:
					print(f"Deaths missing from player {sid3} in log {log_id}")
				error = True
				break
			# make sure the damage was recorded
			if key_dmg not in player:
				if verbose:
					print(f"Damage missing from player {sid3} in log {log_id}")
				error = True
				break
			# make sure the damage taken was recorded
			if key_dt not in player:
				if verbose:
					print(f"Damage taken missing from player {sid3} in log {log_id}")
				error = True
				break

			# if the player was on the red team
			if player[key_team] == "Red":
				# determine which class the player played
				if player[key_stats][0][key_type] == "scout":
					# place them in the correct team / class list for sorting
					red_scouts = np.append(red_scouts, sid3)
					# add their stats to the correct team / class list for sorting
					red_scouts_stats = _add_class_stats(red_scouts_stats, player, key_kills, key_assists,\
						key_deaths, key_dmg, key_dt)
				elif player[key_stats][0][key_type] == "soldier":
					red_soldiers = np.append(red_soldiers, sid3)
					red_soldiers_stats = _add_class_stats(red_soldiers_stats, player, key_kills, key_assists,\
						key_deaths, key_dmg, key_dt)
				elif player[key_stats][0][key_type] == "demoman":
					red_demo = np.append(red_demo, sid3)
					red_demo_stats = _add_class_stats(red_demo_stats, player, key_kills, key_assists,\
						key_deaths, key_dmg, key_dt)
				elif player[key_stats][0][key_type] == "medic":
					# key names for each medic statistic
					key_heals = "heal"
					key_ubers = "ubers"
					key_drops = "drops"
					# make sure keys for medic stats exist for each class
					if key_heals not in player:
						if verbose:
							print(f"Heals missing from from medic {sid3} in log {log_id}")
						error = True
						break
					if key_ubers not in player:
						if verbose:
							print(f"Ubers missing from medic {sid3} in log {log_id}")
						error = True
						break
					if key_drops not in player:
						if verbose:
							print(f"Drops missing from medic {sid3} in log {log_id}")
						error = True
						break
					# place them in the correct team / class list for sorting
					red_med = np.append(red_med, sid3)
					# add their stats to the correct team / class list for sorting
					red_med_stats = _add_class_stats(red_med_stats, player, key_kills, key_assists,\
				    	key_deaths, key_dmg, key_dt, key_heals, key_ubers, key_drops)
				# if the class isn't a meta sixes class
				else:
					if verbose:
						print(f"Primary class is non-sixes meta for player {sid3} in log {log_id}")
					error = True
					break
			# if the player was on the blu team
			elif player[key_team] == "Blue":
				# determine which class the player played
				if player[key_stats][0][key_type] == "scout":
					# place them in the correct team / class list for sorting
					blu_scouts = np.append(blu_scouts, sid3)
					# add their stats to the correct team / class list for sorting
					blu_scouts_stats = _add_class_stats(blu_scouts_stats, player, key_kills, key_assists,\
						key_deaths, key_dmg, key_dt)
				elif player[key_stats][0][key_type] == "soldier":
					blu_soldiers = np.append(blu_soldiers, sid3)
					blu_soldiers_stats = _add_class_stats(blu_soldiers_stats, player, key_kills, key_assists,\
						key_deaths, key_dmg, key_dt)
				elif player[key_stats][0][key_type] == "demoman":
					blu_demo = np.append(blu_demo, sid3)
					blu_demo_stats = _add_class_stats(blu_demo_stats, player, key_kills, key_assists,\
						key_deaths, key_dmg, key_dt)
				elif player[key_stats][0][key_type] == "medic":
					# key names for each medic statistic
					key_heals = "heal"
					key_ubers = "ubers"
					key_drops = "drops"
					# make sure keys for medic stats exist for each class
					if key_heals not in player:
						if verbose:
							print(f"Heals missing from from medic {sid3} in log {log_id}")
						error = True
						break
					if key_ubers not in player:
						if verbose:
							print(f"Ubers missing from medic {sid3} in log {log_id}")
						error = True
						break
					if key_drops not in player:
						if verbose:
							print(f"Drops missing from medic {sid3} in log {log_id}")
						error = True
						break
					# place them in the correct team / class list for sorting
					blu_med = np.append(blu_med, sid3)
					# add their stats to the correct team / class list for sorting
					blu_med_stats = _add_class_stats(blu_med_stats, player, key_kills, key_assists,\
						key_deaths, key_dmg, key_dt, key_heals, key_ubers, key_drops)
				# if the class isn't a meta sixes class
				else:
					if verbose:
						print(f"Primary class is non-sixes meta for player {sid3} in log {log_id}")
					error = True
					break
			# if the team wasn't recognized
			else:
				if verbose:
					print(f"Unknown team for player {sid3} in log {log_id}")
				error = True
				break
		
		# if there was an error encountered inside the loop
		if error:
			# stop using this log and move to next one
			continue

		# make sure there are the correct amount of players for each class on each team
		if len(red_scouts) != 2:
			if verbose:
				print(f"Not 2 scouts on red team in log {log_id}")
			continue
		elif len(red_soldiers) != 2:
			if verbose:
				print(f"Not 2 soldiers on red team in log {log_id}")
			continue
		elif len(red_demo) != 1:
			if verbose:
				print(f"Not 1 demo on red team in log {log_id}")
			continue
		elif len(red_med) != 1:
			if verbose:
				print(f"Not 1 med on red team in log {log_id}")
			continue
		elif len(blu_scouts) != 2:
			if verbose:
				print(f"Not 2 scouts on blu team in log {log_id}")
			continue
		elif len(blu_soldiers) != 2:
			if verbose:
				print(f"Not 2 soldiers on blu team in log {log_id}")
			continue
		elif len(blu_demo) != 1:
			if verbose:
				print(f"Not 1 demo on blu team in log {log_id}")
			continue
		elif len(blu_med) != 1:
			if verbose:
				print(f"Not 1 med on blu team in log {log_id}")
			continue

		# combine all steam id 3s into one array
		player_sid3s = np.hstack((red_scouts, red_soldiers, red_demo, red_med,\
			blu_scouts, blu_soldiers, blu_demo, blu_med))
		# combine all stats into one array including match length
		match_stats = np.hstack((match_length, red_scouts_stats, red_soldiers_stats, red_demo_stats, red_med_stats,\
			blu_scouts_stats, blu_soldiers_stats, blu_demo_stats, blu_med_stats))

		# get all player names
		player_names = data["names"]
		# make sure there are 12 player names
		if len(player_names) != 12:
			if verbose:
				print(f"There aren't 12 player names in log {log_id}")
			continue

		# flag to tell function to drop this log if there was en error encountered inside the loop
		error = False

		# print out each player name and steam id 3
		for sid3 in player_sid3s:
			# make sure all of the steam id 3s have a player name linked to them
			if sid3 not in player_names:
				if verbose:
					print(f"Player {sid3} doesn't have a name in log {log_id}")
				error = True
				break

		# if there was an error encountered inside the loop
		if error:
			continue

		# get match date in us eastern timezone (since that's the standard timezone for tf2 in na)
		match_datetime = datetime.fromtimestamp(data["info"]["date"], tz=pytz.timezone("US/Eastern"))
		# get match year, month, day, and day of the week
		match_date = np.array([match_datetime.year, match_datetime.month, match_datetime.day])
		match_weekday = np.array(match_datetime.strftime("%A"), ndmin=1)

		# add all of the data collected from this match to the collective data arrays

		# log id that was used
		used_logs = np.append(used_logs, log_id)

		# input data
		players = np.vstack((players, player_sid3s))
		gamemodes = np.vstack((gamemodes, gamemode))
		maps = np.vstack((maps, map_name))
		dates = np.vstack((dates, match_date))
		weekdays = np.vstack((weekdays, match_weekday))

		# output data
		scores = np.vstack((scores, score))
		stats = np.vstack((stats, match_stats))

	if verbose:
		print(f"\nUsed {used_logs.size} logs. Storing data into csv files...")

	# if there isn't already a folder for the data, create one
	if not os.path.isdir(data_path):
		os.mkdir(data_path)
	
	# create data frames for each array
	df_used_logs = pd.DataFrame(used_logs)

	df_players = pd.DataFrame(players)
	df_gamemodes = pd.DataFrame(gamemodes)
	df_maps = pd.DataFrame(maps)
	df_dates = pd.DataFrame(dates)
	df_weekdays = pd.DataFrame(weekdays)

	df_scores = pd.DataFrame(scores)
	df_stats = pd.DataFrame(stats)

	# column name of index column in csv files
	index_label = "Index"

	# store ids of logs that were valid and will be used to csv files
	df_used_logs.to_csv(used_logs_path, index_label=index_label, header=["Log ID"])

	# store input data into csv files
	df_players.to_csv(player_data_path, index_label=index_label, header=[\
		"Red Scout 1", "Red Scout 2", "Red Soldier 1", "Red Soldier 2", "Red Demo", "Red Medic",\
		"Blu Scout 1", "Blu Scout 2", "Blu Soldier 1", "Blu Soldier 2", "Blu Demo", "Blu Medic"])
	df_gamemodes.to_csv(gamemode_data_path, index_label=index_label, header=["Gamemode"])
	df_maps.to_csv(maps_data_path, index_label=index_label, header=["Map"])
	df_dates.to_csv(dates_data_path, index_label=index_label, header=["Year", "Month", "Day"])
	df_weekdays.to_csv(weekdays_data_path, index_label=index_label, header=["Weekday"])

	# store output data into csv files
	df_scores.to_csv(scores_data_path, index_label=index_label, header=["Red Score", "Blu Score"])
	df_stats.to_csv(stats_data_path, index_label=index_label, header=["Match Length",\
		"Red Scout 1 Kills", "Red Scout 1 Assists", "Red Scout 1 Deaths",\
		"Red Scout 1 Damage", "Red Scout 1 Damage Taken",\
		"Red Scout 2 Kills", "Red Scout 2 Assists", "Red Scout 2 Deaths",\
		"Red Scout 2 Damage", "Red Scout 2 Damage Taken",\
		"Red Soldier 1 Kills", "Red Soldier 1 Assists", "Red Soldier 1 Deaths",\
		"Red Soldier 1 Damage", "Red Soldier 1 Damage Taken",\
		"Red Soldier 2 Kills", "Red Soldier 2 Assists", "Red Soldier 2 Deaths",\
		"Red Soldier 2 Damage", "Red Soldier 2 Damage Taken",\
		"Red Demo Kills", "Red Demo Assists", "Red Demo Deaths",\
		"Red Demo Damage", "Red Demo Damage Taken",\
		"Red Medic Kills", "Red Medic Assists", "Red Medic Deaths",\
		"Red Medic Damage", "Red Medic Damage Taken",\
		"Red Medic Heals", "Red Medic Ubers", "Red Medic Drops",\
		"Blu Scout 1 Kills", "Blu Scout 1 Assists", "Blu Scout 1 Deaths",\
		"Blu Scout 1 Damage", "Blu Scout 1 Damage Taken",\
		"Blu Scout 2 Kills", "Blu Scout 2 Assists", "Blu Scout 2 Deaths",\
		"Blu Scout 2 Damage", "Blu Scout 2 Damage Taken",\
		"Blu Soldier 1 Kills", "Blu Soldier 1 Assists", "Blu Soldier 1 Deaths",\
		"Blu Soldier 1 Damage", "Blu Soldier 1 Damage Taken",\
		"Blu Soldier 2 Kills", "Blu Soldier 2 Assists", "Blu Soldier 2 Deaths",\
		"Blu Soldier 2 Damage", "Blu Soldier 2 Damage Taken",\
		"Blu Demo Kills", "Blu Demo Assists", "Blu Demo Deaths",\
		"Blu Demo Damage", "Blu Demo Damage Taken",\
		"Blu Medic Kills", "Blu Medic Assists", "Blu Medic Deaths",\
		"Blu Medic Damage", "Blu Medic Damage Taken",\
		"Blu Medic Heals", "Blu Medic Ubers", "Blu Medic Drops"])
	
	if verbose:
		print("Data stored.")

	# return the number of valid logs that it stored data from, along with all of the data collected
	return used_logs.size, used_logs, players, gamemodes, maps, dates, weekdays, scores, stats

# prepares data to be fed into the goblin
# reads data from csv files if data that was passed is none
def prepare_log_data(players=None, gamemodes=None, maps=None, dates=None, weekdays=None, scores=None, stats=None, verbose=True):
	# if there isn't a folder for the data
	if not os.path.isdir(data_path):
		raise FileNotFoundError("Missing data folder.")
	
	if verbose:
		print("Recollecting data to prepare it...")

	# if not data was passed as a parameter, read data from csv files and turn it into arrays
	if players is None:
		# if any of the data files are missing
		if not os.path.isfile(player_data_path):
			raise FileNotFoundError("Missing player data file")
		players = np.delete(np.array(pd.read_csv(player_data_path)), 0, 1)
	if gamemodes is None:
		if not os.path.isfile(gamemode_data_path):
			raise FileNotFoundError("Missing gamemode data file")
		gamemodes = np.delete(np.array(pd.read_csv(gamemode_data_path)), 0, 1)
	if maps is None:
		if not os.path.isfile(maps_data_path):
			raise FileNotFoundError("Missing map data file")
		maps = np.delete(np.array(pd.read_csv(maps_data_path)), 0, 1)
	if dates is None:
		if not os.path.isfile(dates_data_path):
			raise FileNotFoundError("Missing date data file")
		dates = np.delete(np.array(pd.read_csv(dates_data_path)), 0, 1)
	if weekdays is None:
		if not os.path.isfile(weekdays_data_path):
			raise FileNotFoundError("Missing weekday data file")
		weekdays = np.delete(np.array(pd.read_csv(weekdays_data_path)), 0, 1)
	if scores is None:
		if not os.path.isfile(scores_data_path):
			raise FileNotFoundError("Missing score data file")
		scores = np.delete(np.array(pd.read_csv(scores_data_path)), 0, 1)
	if stats is None:
		if not os.path.isfile(stats_data_path):
			raise FileNotFoundError("Missing stats data file")
		stats = np.delete(np.array(pd.read_csv(stats_data_path)), 0, 1)

	# Encode the categorical data with one hot encoding

	if verbose:
		print("Preparing data to be fed into the goblin...")

	# one hot encode players
	classes = np.unique(players)
	players_onehot = np.searchsorted(classes, players)
	players_onehot = to_categorical(players_onehot)
	players_onehot = players_onehot.reshape(players.shape[0], players_onehot.shape[1] * players_onehot.shape[2])

	# one hot encode gamemodes
	classes = np.unique(gamemodes)
	# make sure both koth and control points are encoded in
	if "koth" not in classes:
		classes = np.append(classes, "koth")
	if "cp" not in classes:
		classes = np.append(classes, "cp")
	classes = np.reshape(classes, (classes.size, 1))
	oe = OrdinalEncoder()
	oe.fit(classes)
	gamemodes_onehot = np.eye(classes.size)[oe.transform(gamemodes).flatten().astype(int)]

	# one hot encode maps
	classes = np.unique(maps)
	maps_onehot = np.searchsorted(classes, maps)
	maps_onehot = to_categorical(maps_onehot)

	# one hot encode months
	months = dates[:, 1]
	months_onehot = np.delete(np.eye(13)[months], 0, 1)

	# one hot encode days
	days = dates[:, 2]
	days_onehot = np.delete(np.eye(32)[days], 0, 1)

	# one hot encode weekdays
	oe.fit([["Sunday"], ["Monday"], ["Tuesday"], ["Wednesday"], ["Thursday"], ["Friday"], ["Saturday"]])
	weekdays_onehot = np.delete(np.eye(8)[oe.transform(weekdays).flatten().astype(int)], 7, 1)

	# one hot encode team scores
	score_cap = 6
	scores_onehot = np.eye(score_cap)[scores].reshape(scores.shape[0], scores.shape[1] * score_cap)

	# use np.hstack() to horizontally combine the input arrays together
	inputs = np.hstack((players_onehot, gamemodes_onehot, maps_onehot, dates.astype(float), months_onehot,\
		days_onehot, weekdays_onehot))
	
	if verbose:
		print("Data prepared for goblin feeding. Storing prepared data into csv files...")

	# create dataframes for inputs and outputs
	df_inputs = pd.DataFrame(inputs)
	df_outputs = pd.DataFrame(scores_onehot)

	# write inputs and outputs to csv files
	df_inputs.to_csv(inputs_data_path, header=False, index=False)
	df_outputs.to_csv(outputs_data_path, header=False, index=False)

	if verbose:
		print("Prepared data stored.")
	
	return inputs, scores_onehot, stats

# gets the inputs and outputs
def read_log_data(with_stats=False, verbose=True):
	if verbose:
		print("Reading prepared data from csv files...")

	# if there isn't a folder for the data
	if not os.path.isdir(data_path):
		raise FileNotFoundError("Missing data folder.")
	# make sure csv data files exist
	if not os.path.isfile(inputs_data_path):
		raise FileNotFoundError("Missing input data file")
	if not os.path.isfile(outputs_data_path):
		raise FileNotFoundError("Missing output data file")
	if with_stats and not os.path.isfile(stats_data_path):
		raise FileNotFoundError("Missing stats data file")
	
	# get inputs and outputs for goblin from csv files
	inputs = np.array(pd.read_csv(inputs_data_path, header=None))
	outputs = np.array(pd.read_csv(outputs_data_path, header=None))
	# if the stats were requested
	if with_stats:
		# get stats from csv file
		stats = np.array(pd.read_csv(stats_data_path))

		if verbose:
			print("Prepared data collected")

		return inputs, outputs, stats
	# if stats were not requested
	else:
		if verbose:
			print("Prepared data collected")

		return inputs, outputs

# if this is being run as its own program to collect data
if __name__ == "__main__":
	import sys

	# first argument must be a number telling how many pages to read from each player's log profile
	if len(sys.argv) < 2:
		print("ERROR: No arguments. Must give number of pages to read from each player's log profile.")
		exit(2)
	
	# make sure the argument is an integer
	try:
		pages = int(sys.argv[1])
	except ValueError:
		print("ERROR: First argument must be a positive integer.")
		exit(2)
	
	# make sure the argument is positive
	if pages < 1:
		print("ERROR: First argument must be a positive integer.")
		exit(2)
	
	verbose = True
	# user can provide argument to not print any messages during execution
	if len(sys.argv) > 2:
		if sys.argv[2] == "-s" or sys.argv[2] == "--silent":
			verbose = False
		else:
			print("Second argument not recognized.")
			exit(2)

	delimiter = "-" * 50

	# get a fresh set of logs and data
	log_ids, sid3s = get_logs(pages, verbose=verbose)
	if verbose:
		print(delimiter)
	num_logs, used_logs, players, gamemodes, maps, dates, weekdays, scores, stats = fetch_log_data(\
		log_ids, sid3s, verbose=verbose)
	if verbose:
		print(delimiter)
	inputs, targets, stats = prepare_log_data(\
		players=players, gamemodes=gamemodes, maps=maps, dates=dates, weekdays=weekdays, scores=scores, stats=stats,\
		verbose=verbose)
	if verbose:
		print(delimiter)
		print("Inputs:")
		print(inputs)
		print(inputs.shape)
		print("Targets:")
		print(targets)
		print(targets.shape)

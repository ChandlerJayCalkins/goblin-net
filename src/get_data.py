import ssl
from urllib.request import urlopen
import json
from datetime import datetime
import pytz
import sys
import numpy as np
# pip install beautifulsoup4 / conda install beautifulsoup4
from bs4 import BeautifulSoup
from get_logs import *

# reads json data of log
context = ssl._create_unverified_context()
log_id = 3389150
log_json_url = f"https://logs.tf/json/{log_id}"
log_url = f"https://logs.tf/{log_id}"
response = urlopen(log_json_url, context=context)
data = json.loads(response.read())

# makes sure all necessary keys were retrieved from json file
if "version" not in data:
	print(f"Log version missing from log {log_id}")
	quit()

if "players" not in data:
	print(f"Player data missing from log {log_id}")
	quit()

if "names" not in data:
	print(f"Names are missing from log {log_id}")
	quit()

if "info" not in data:
	print(f"Info field missing from log {log_id}")

if "map" not in data["info"]:
	print(f"Map missing from log {log_id}")
	quit()

if "date" not in data["info"]:
	print(f"Date missing from log {log_id}")
	quit()

# makes sure log version is correct
if data["version"] != 3:
	print(f"Log version is not 3 in log {log_id}")
	quit()

# gets player data
player_data = data["players"].keys()
if len(player_data) != 12:
	print(f"Player count is not 12 in log {log_id}")
	quit()

# gets steam id 3s from player data
player_sid3s = np.array(list(player_data))
player_names = data["names"]
if len(player_names) != 12:
	print(f"There aren't 12 player names in log {log_id}")
	quit()

for sid3 in player_sid3s:
	if sid3 not in player_names:
		print(f"Player {sid3} doesn't have a name in log {log_id}")
		quit()
	
	print(f"{player_names[sid3]}: {sid3}")

# get map name
map_name = data["info"]["map"]
print(map_name)

# get match date in us eastern timezone (since that's the standard timezone for tf2 in na)
match_date = datetime.fromtimestamp(data["info"]["date"], tz=pytz.timezone("US/Eastern"))
match_year = match_date.year
match_month = match_date.month
match_day = match_date.day
match_weekday = match_date.strftime("%A")
print(match_year)
print(match_month)
print(match_day)
print(match_weekday)

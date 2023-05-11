# goblin-net
Neural Network(s) that predicts the outcomes of competitive Team Fortress 2 Sixes matches.

## Usage

### collect_log_data.py

- This program collects the logs from logs.tf of the last given number of pages of each of the players inputted in the `profiles.csv` file
- It only collects valid sixes logs and doesn't include logs that contain players that weren't inputted in the `profiles.csv` file
- Arguments
	- The first argument must be a positive integer that tells how many pages of logs to collect from each player in the `profiles.csv` file
	- There can be a second argument, `-s` or `--slient`, to make it so the program doesn't print any outputs to commandline.

### train_neural_net.py

- This program takes the data collects from `collect_log_data.py` and trains a neural network that you can build
- Arguments
	- `-s` or `--silent`: Makes it so the program doesn't print any outputs to commandline.
	- `-nd` or `--new-data`: Runs `collect_log_data.py` to collect a new batch of data. Must be followed by a positive integer argument to tell the program how many pages to read.
	- `-e` or `--epochs`: Tells the program how many epochs to run through while training.
	- `--name`: Tells the program what to name the file when it stores the neural network (doesn't need to include a file extension).

## Setup
- Create a folder called `data` in the root directory of this repository.
- In this folder, create a file called `profiles.csv`
	- In this file, put a list of URLs to steam profiles of all the players you want to train the network on with each URL being on a new line
	- For example:
		```
		https://steamcommunity.com/id/SOOOOOAPYMEiSTER/
		https://steamcommunity.com/id/dsgxsoldier/
		https://steamcommunity.com/id/kobe1920/
		https://steamcommunity.com/id/habib_6ix_God/
		https://steamcommunity.com/id/branslam/
		https://steamcommunity.com/id/quarterlifecrisis/
		https://steamcommunity.com/id/nuio209e0sfjd0fj209mfd/
		https://steamcommunity.com/id/iikeepitreal/
		https://steamcommunity.com/id/proskeez/
		https://steamcommunity.com/id/asdasdasdadasdasdasdadasda/
		https://steamcommunity.com/profiles/76561198208247285
		https://steamcommunity.com/id/reighyawn/
		https://steamcommunity.com/id/b4nny/
		https://steamcommunity.com/id/comtedelaperouse/
		https://steamcommunity.com/id/donovin/
		https://steamcommunity.com/id/UnskilledSoldier/
		https://steamcommunity.com/id/yumyum7654/
		https://steamcommunity.com/id/supersandblast/
		https://steamcommunity.com/id/mirrorman123/
		https://steamcommunity.com/id/supertramp_/
		https://steamcommunity.com/id/mew2king/
		https://steamcommunity.com/id/GrapeJuiceIII/
		https://steamcommunity.com/profiles/76561198084582998
		https://steamcommunity.com/id/1324511155552555555123/
		...
		```
- Create a file called `.env` in either the root directory of the repository or in the [src](src) folder.
	- Inside that file, put in your Steam API key in the format `STEAM_API_KEY=*your steam api key here*`
	- This gives the goblin access to the Steam API.
	- The program will not be able to get logs from the list of players in the `players.csv` file without this.
	- You can get your own personal Steam API key here if you have your own domain name to give it:
		- [https://steamcommunity.com/dev/apikey](https://steamcommunity.com/dev/apikey)
		- WARNING: This link will show your current personal Steam API key if you already have one
 
## Dependencies
- Anaconda Environment (Python 3.9.13)
- BeautifulSoup
	- `pip install beautifulsoup4`
- Steam API
	- `pip install python-steam-api`
- Tensorflow
	- `pip install tensorflow`

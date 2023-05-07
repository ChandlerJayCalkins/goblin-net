# goblin-net
Neural Network(s) that predicts the outcomes of competitive Team Fortress 2 Sixes matches.

## Setup
- Create a folder called `data` in the root directory of this repository.
	- This will be where you put the `players.csv` file of players for the neural network to train on the logs of.
- Create a file called `.env` in either the root directory of the repository or in the `src` folder.
	- Inside that folder, put `STEAM_API_KEY=*your steam api key here*`
	- This gives the goblin access to the Steam API.
	- The program will not be able to get logs from the list of players in the `players.csv` file without this.
	- You can get your own personal Steam API key here if you have your own domain name to give it:
		- [https://steamcommunity.com/dev/apikey](https://steamcommunity.com/dev/apikey)
 
## Dependencies
- Anaconda Environment (Python 3.9.13)
- BeautifulSoup
	- `pip install beautifulsoup4`
- Steam API
	- `pip install python-steam-api`

########################################################################################################################
#
#
# Goblin Net: Neural Networks that Predict the Outcome of competitive Team Fortress 2 Matches
#
# train_neural_net
#
# Module for training and testing the neural network
#
# Authors / Contributors:
# Chandler Calkins
#
#
########################################################################################################################

# used for splitting data into training and test sets
from sklearn.model_selection import train_test_split
# used for building the neural net(s)
from tensorflow import keras

def train_goblin():
	pass

# if this is being run as its own program to train the neural net(s)
if __name__ == "__main__":
	import sys
	from collect_log_data import *

	verbose = True
	new_data = False
	pages = 0

	# loop through each argument
	i = 1
	while i < len(sys.argv):
		# argument to disable progress messages during execution
		if sys.argv[i] == "-s" or sys.argv[i] == "--silent":
			verbose = False
		# argument to fetch new set of data
		elif sys.argv[i] == "-n" or sys.argv[i] == "--new-data":
			new_data = True
			i += 1
			# make sure there is a follow up argument that is a positive integer
			# this follow up arg tells how many pages of log profiles to read from each inputted player
			if i >= len(sys.argv):
				print(f"ERROR: {sys.arv[i-1]} requires a positive integer after it.")
				exit(2)
			try:
				pages = int(sys.argv[i])
			except ValueError:
				print(f"ERROR: {sys.arv[i-1]} requires a positive integer after it.")
				exit(2)
			if pages < 1:
				print(f"ERROR: {sys.arv[i-1]} requires a positive integer after it.")
				exit(2)
		# if the argument isn't recognized
		else:
			print(f"ERROR: Argument {i} not recognized: {sys.argv[i]}")
			exit(2)

		i += 1
	
	delimiter = "-" * 50

	# if new data was requested with an argument
	if new_data:

		# get a fresh set of logs and data
		log_ids = get_logs(pages, verbose=verbose)
		if verbose:
			print(delimiter)
		num_logs, used_logs, players, gamemodes, maps, dates, weekdays, scores, stats = fetch_log_data(log_ids, verbose=verbose)
		if verbose:
			print(delimiter)
		inputs, targets, stats = prepare_log_data(\
			players=players, gamemodes=gamemodes, maps=maps, dates=dates, weekdays=weekdays, scores=scores, stats=stats,\
			verbose=verbose)
	# if new data was not requested, just fetch the current data from the data file
	else:
		inputs, targets = read_log_data(verbose=verbose)
	
	if verbose:
		print(delimiter)
		print("Inputs:")
		print(inputs)
		print(inputs.shape)
		print("Targets:")
		print(targets)
		print(targets.shape)

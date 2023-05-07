########################################################################################################################
#
#
# Goblin Net: Neural Networks that Predict the Outcome of competitive Team Fortress 2 Matches
#
# goblin
#
# Module for making predictions on TF2 matches
#
# Authors / Contributors:
# Chandler Calkins
#
#
########################################################################################################################

# if this is being run as its own program to make predictions with the neural net(s)
if __name__ == "__main__":
	import sys

	verbose = True
	new_data = False
	pages = 0
	train = False

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
		# argument to train a new neural network
		elif sys.argv[i] == "-t" or sys.argv[i] == "--train":
			train == True
		# if the argument isn't recognized
		else:
			print(f"ERROR: Argument {i} not recognized: {sys.argv[i]}")
			exit(2)

		i += 1
	
	# if new data was requested with an argument
	if new_data:
		from collect_log_data import *

		delimiter = "-" * 50

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
		if verbose:
			print(delimiter)
			print("Inputs:")
			print(inputs)
			print(inputs.shape)
			print("Targets:")
			print(targets)
			print(targets.shape)
	
	# if training a new neural network was requested
	if train:
		from train_neural_net import *

		# if fresh data wasn't requested and is already loaded, load data from csv files
		if not new_data:
			inputs, targets = read_log_data(verbose=verbose)
			if verbose:
				print(delimiter)
				print("Inputs:")
				print(inputs)
				print(inputs.shape)
				print("Targets:")
				print(targets)
				print(targets.shape)

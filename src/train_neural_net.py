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
# used for iterating through multiple lists at once
import itertools

# creates neural network(s) for predicting tf2 matches, trains it, tests it (optional), then stores it in a json file
def train_goblin(inputs, score_targets, stat_targets=None, score_nodes=None, score_activations=None, score_loss = None,\
	score_optimizer=None, file_name="goblin", test=True, verbose=True):
	if verbose:
		print("Building the goblin...")
	
	# default number of hidden layers for the score predictor to 1 with the average of the input and target nodes
	if score_nodes is None:
		#score_nodes = [int((2/3) * inputs.shape[1]) + score_targets.shape[1]]
		score_nodes = [int((inputs.shape[1] + score_targets.shape[1]) / 2)]
	
	# use relu on single hidden default layer
	if score_activations is None:
		score_activations = ["relu"]
	
	# use categorical cross_entropy as default loss function
	if score_loss is None:
		score_loss = "categorical_crossentropy"
	
	# use adam as default optimizer
	if score_optimizer is None:
		score_optimizer = keras.optimizers.Adam()
	
	# make sure both of these lists are requesting the same number of hidden layers
	if len(score_nodes) != len(score_activations):
		raise ValueError("score_nodes and score_activations lists aren't the same length.")

	# split score data into train and test sets
	score_train_inputs, score_test_inputs, score_train_targets, score_test_targets =\
		train_test_split(inputs, score_targets, test_size=0.1)
	
	if verbose:
		print("Shape of train inputs:", score_train_inputs.shape)
		print("Shape of train targets:", score_train_targets.shape)
	
	# create neural network for predicting scores of matches
	score_goblin = keras.models.Sequential()

	if verbose:
		print("Adding input layer...")
	
	# give it an input layer
	score_goblin.add(keras.layers.InputLayer(input_shape=(inputs.shape[1],)))

	# give it hidden layers
	for (nodes, activation, i) in zip(score_nodes, score_activations, range(1, len(score_nodes)+1)):
		if verbose:
			print(f"Adding dense hidden layer with {nodes} nodes and activation function {activation}...")
		
		score_goblin.add(keras.layers.Dense(units=nodes, activation=activation, name=f"Hidden_Layer_{i}"))
	
	if verbose:
		print("Adding output layer...")

	# give it an output layer
	score_goblin.add(keras.layers.Dense(units=score_targets.shape[1], activation="sigmoid", name="Output_Layer"))

	if verbose:
		print(f"Compiling the goblin...")

	# compile the neural network
	score_goblin.compile(optimizer=score_optimizer, loss=score_loss, metrics=["accuracy"])

	train_verbose = 0
	if verbose:
		score_goblin.summary()
		print("Training the goblin...")
		train_verbose = 2
	
	# train the neural network
	score_history = score_goblin.fit(score_train_inputs, score_train_targets, batch_size=32, epochs=1, verbose=train_verbose)

# loads neural network(s) from json file
def load_goblin(load_stats=False, verbose=True):
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
	
	if verbose:
		print(delimiter)
	
	train_goblin(inputs, targets, verbose=verbose)

########################################################################################################################
#
#
# Goblin Net: Neural Networks that Predict the Outcome of competitive Team Fortress 2 Matches
#
# train_neural_net
#
# The neural network training and testing program
#
# Authors / Contributors:
# Chandler Calkins
#
#
########################################################################################################################

from collect_log_data import *

delimiter = "-" * 50

get_logs(1)
log_ids = read_log_ids()
print(delimiter)
num_logs, used_logs, players, gamemodes, maps, dates, weekdays, scores, stats = fetch_log_data(log_ids)
print(delimiter)
inputs, targets, stats = prepare_log_data(\
	players=players, gamemodes=gamemodes, maps=maps, dates=dates, weekdays=weekdays, scores=scores, stats=stats)
print(delimiter)
print("Inputs:")
print(inputs)
print(inputs.shape)
print("Targets:")
print(targets)
print(targets.shape)

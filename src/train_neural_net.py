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

print("Getting logs from list of players...")
get_logs(1)
log_ids = read_log_ids()
print(f"Number of logs fetched: {len(log_ids)}")
print(delimiter)
print("Weeding out logs and extracting data from valid logs...")
num_logs = fetch_log_data(log_ids)
print(f"Number of logs used: {num_logs}")
print(delimiter)
print("Preparing data to be fed into the goblin...")
prepare_log_data()
inputs, targets = read_log_data()
print("Inputs:")
print(inputs)
print(inputs.shape)
print("Targets:")
print(targets)
print(targets.shape)

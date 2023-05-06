########################################################################################################################
#
#
# Goblin Net: Neural Networks that Predict the Outcome of competitive Team Fortress 2 Matches
#
# goblin
#
# The neural network training and testing program
#
# Authors / Contributors:
# Chandler Calkins
#
#
########################################################################################################################

from get_log_data import *

delimiter = "-" * 50

get_logs(1)
log_ids = read_log_ids()
print(f"Number of logs fetched: {len(log_ids)}")
num_logs = fetch_log_data(log_ids)
print(f"Number of logs used: {num_logs}")
prepare_log_data()
inputs, targets = read_log_data()
print("Inputs:")
print(inputs)
print(inputs.shape)
print(delimiter)
print("Targets:")
print(targets)
print(targets.shape)

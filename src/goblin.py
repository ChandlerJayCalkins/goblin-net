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

get_logs(1)
log_ids = read_log_ids()
fetch_log_data(log_ids)
prepare_log_data()
inputs, targets = read_log_data()
print("Inputs:")
print(inputs)
print(inputs.shape)
print("-" * 50)
print("Targets:")
print(targets)
print(targets.shape)

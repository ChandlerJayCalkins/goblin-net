from get_log_data import *

log_ids = ["3389150", "3389174", "3389227", "3389291"]
refresh_log_data(log_ids)
encode_log_data()
inputs, outputs = get_log_data()
print(inputs)
print("-" * 50)
print(outputs)

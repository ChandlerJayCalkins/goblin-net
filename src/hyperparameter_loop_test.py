min_neurons = 10
max_neurons = 20

iterations = 0

for hidden_layer_count in range(1, 4):
	hidden_neuron_counts = [min_neurons] * hidden_layer_count
	outer_layer = 0
	while outer_layer < hidden_layer_count:
		iterations += 1
		print(hidden_neuron_counts)

		current_layer = 0
		for i in range(hidden_layer_count):
			hidden_neuron_counts[i] += 1
			if hidden_neuron_counts[i] > max_neurons:
				hidden_neuron_counts[i] = min_neurons
				current_layer += 1
			else:
				break
		
		outer_layer = max(outer_layer, current_layer)

# Should take 1,463 iterations, because:
# Number of possible neurons in a layer: 20 - 10 + 1 = 11
# Number of possible layers: 3
# 11^1 (number of combinations with 1 layer) = 11
# 11^2 (number of combinations with 2 layers) = 121
# 11^3 (number of combinations with 3 layers) = 1,331
# 1,331 + 121, + 11 = 1,463
print(iterations)
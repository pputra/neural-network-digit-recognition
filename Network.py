import numpy as np


class Network:
    def __init__(self, layer_sizes):
        self.num_hidden_layers = len(layer_sizes) - 2
        self.layer_sizes = layer_sizes
        self.biases = [np.random.rand(num_rows, 1) for num_rows in self.layer_sizes]
        self.weights = []

        for i in range(len(self.layer_sizes) - 1):
            input_neuron_nums = self.layer_sizes[i]
            output_neuron_nums = self.layer_sizes[i + 1]
            self.weights.append(np.random.rand(input_neuron_nums, output_neuron_nums))

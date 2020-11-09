import numpy as np

from random import shuffle


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

    def train(self, training_data, mini_batch_size):
        mini_batches = self.get_mini_batches(training_data, mini_batch_size)
        print('mini batch size: ', mini_batch_size)
        print('number of batches: ', len(mini_batches))


    def get_mini_batches(self, training_data, mini_batch_size):
        shuffle(training_data)
        return [training_data[i:i + mini_batch_size] for i in range(0, len(training_data), mini_batch_size)]

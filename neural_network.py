import numpy as np

from random import shuffle
import math_utils


class Network:
    def __init__(self, layer_sizes):
        self.num_hidden_layers = len(layer_sizes) - 2
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(num_rows, 1) for num_rows in self.layer_sizes[1:]]
        self.weights = []

        for i in range(len(self.layer_sizes) - 1):
            input_neuron_nums = self.layer_sizes[i]
            output_neuron_nums = self.layer_sizes[i + 1]
            self.weights.append(np.random.randn(output_neuron_nums, input_neuron_nums))
            # self.weights.append(np.ones((output_neuron_nums, input_neuron_nums)))

    def train(self, training_data, mini_batch_size, epochs):
        for epoch_i in range(epochs):
            mini_batches = self.get_mini_batches(training_data, mini_batch_size)
            for mini_batch in mini_batches:
                for data in mini_batch:
                    # forward pass
                    activations = [data[0]]
                    activation = activations[0]
                    weight_activation_dot_products = []
                    for i in range(len(self.biases)):
                        bias = self.biases[i]
                        weight = self.weights[i]
                        z = np.dot(weight, activation) + bias
                        activation = math_utils.sigmoid(z)
                        activations.append(activation)
                        weight_activation_dot_products.append(z)

                    # backward propagation
                    output_diff = activations[-1] - data[1]
                    err = output_diff * math_utils.sigmoid_derivative(weight_activation_dot_products[-1])

                    print(err)
                    exit()


    def get_mini_batches(self, training_data, mini_batch_size):
        # TODO: shuffle data later
        # shuffle(training_data)
        return [training_data[i:i + mini_batch_size] for i in range(0, len(training_data), mini_batch_size)]

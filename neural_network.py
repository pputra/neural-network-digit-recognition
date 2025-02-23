import numpy as np

from random import shuffle
import math_utils
from training_io import TrainingIO


class Network:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(num_rows, 1) for num_rows in self.layer_sizes[1:]]
        self.weights = []

        for i in range(len(self.layer_sizes) - 1):
            input_neuron_nums = self.layer_sizes[i]
            output_neuron_nums = self.layer_sizes[i + 1]
            self.weights.append(np.random.randn(output_neuron_nums, input_neuron_nums))

    def train(self, training_data, test_data, mini_batch_size, epochs, learning_rate, verbose,
              test_predictions_file_name):
        for epoch_i in range(epochs):
            mini_batches = self.get_mini_batches(training_data, mini_batch_size)

            for mini_batch in mini_batches:
                to_learn_biases = [np.zeros(bias.shape) for bias in self.biases]
                to_learn_weights = [np.zeros(weight.shape) for weight in self.weights]

                for data in mini_batch:
                    gradients = self.get_gradients(data[0], data[1])
                    bias_gradients = gradients['bias_gradients']
                    weight_gradients = gradients['weight_gradients']
                    self.update_to_learn_biases_weights(to_learn_biases, to_learn_weights,
                                                        bias_gradients, weight_gradients)

                self.update_nn_biases_weights(learning_rate, to_learn_biases, to_learn_weights)

            if test_data and verbose:
                TrainingIO.print_accuracy(self.biases, self.weights, test_data, epoch_i)

        if test_data and test_predictions_file_name:
            TrainingIO.write_predictions(self.biases, self.weights, test_data, test_predictions_file_name)

    def update_nn_biases_weights(self, learning_rate, to_learn_biases, to_learn_weights):
        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i] - learning_rate * to_learn_biases[i]
            self.weights[i] = self.weights[i] - learning_rate * to_learn_weights[i]

    def update_to_learn_biases_weights(self, biases, weights, bias_gradients, weight_gradients):
        for i in range(len(self.biases)):
            biases[i] = biases[i] + bias_gradients[i]
            weights[i] = weights[i] + weight_gradients[i]

    def get_gradients(self, inputs, expected_outputs):
        bias_gradients = [np.zeros(bias.shape) for bias in self.biases]
        weight_gradients = [np.zeros(weight.shape) for weight in self.weights]

        # forward pass
        activations = [inputs]
        activation = activations[0]
        sigmoid_z_caches = []

        for i in range(len(self.biases)):
            bias = self.biases[i]
            weight = self.weights[i]
            z = np.dot(weight, activation) + bias

            if i == len(self.biases) - 1:
                # output layer
                activation = math_utils.softmax(z)
            else:
                # hidden layers
                activation = math_utils.sigmoid(z)

            activations.append(activation)
            sigmoid_z_caches.append(z)

        # backward propagation

        # err for the output layer
        err = math_utils.cross_entropy(activations[-1], expected_outputs)

        bias_gradients[-1] = err
        weight_gradients[-1] = np.dot(err, activations[-2].T)

        # err for the hidden layers
        for i in range(len(self.layer_sizes) - 1, 1, -1):
            err = np.dot(self.weights[i-1].T, err) * \
                  math_utils.sigmoid_derivative(sigmoid_z_caches[i-2])
            bias_gradients[i-2] = err
            weight_gradients[i-2] = np.dot(err, activations[i-2].T)

        return {
            'bias_gradients': bias_gradients,
            'weight_gradients': weight_gradients
        }

    def get_mini_batches(self, training_data, mini_batch_size):
        shuffle(training_data)
        return [training_data[i:i + mini_batch_size] for i in range(0, len(training_data), mini_batch_size)]

import numpy as np

from random import shuffle
import math_utils


class Network:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(num_rows, 1) for num_rows in self.layer_sizes[1:]]
        self.weights = []

        for i in range(len(self.layer_sizes) - 1):
            input_neuron_nums = self.layer_sizes[i]
            output_neuron_nums = self.layer_sizes[i + 1]
            self.weights.append(np.random.randn(output_neuron_nums, input_neuron_nums))

    def train(self, training_data, test_data, mini_batch_size, epochs, learning_rate, verbose):
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

                self.update_nn_biases_weights(learning_rate, to_learn_biases, to_learn_weights, len(mini_batch))

            if test_data and verbose:
                self.print_accuracy(test_data, epoch_i)

    def update_nn_biases_weights(self, learning_rate, to_learn_biases, to_learn_weights, mini_batch_size):
        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i] - (learning_rate / mini_batch_size) * to_learn_biases[i]
            self.weights[i] = self.weights[i] - (learning_rate / mini_batch_size) * to_learn_weights[i]

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
            activation = math_utils.sigmoid(z)
            activations.append(activation)
            sigmoid_z_caches.append(z)

        # backward propagation

        # err for the output layer
        output_diff = activations[-1] - expected_outputs
        err = output_diff * math_utils.sigmoid_derivative(sigmoid_z_caches[-1])

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

    def print_accuracy(self, test_data, epoch_i):
        nums_correct_result = 0

        for data in test_data:
            expected_result = data[1]
            activation = data[0]

            for i in range(len(self.biases)):
                bias = self.biases[i]
                weight = self.weights[i]
                z = np.dot(weight, activation) + bias
                activation = math_utils.sigmoid(z)
            actual_result = np.argmax(activation)

            if actual_result == expected_result:
                nums_correct_result += 1

        print('epoch', epoch_i, ' accuracy:', nums_correct_result / len(test_data))

from numpy import dot, argmax

import math_utils


class TrainingIO:

    @staticmethod
    def print_accuracy(biases, weights, test_data, epoch_i):
        nums_correct_result = 0

        for data in test_data:
            expected_result = data[1]
            activation = data[0]

            for i in range(len(biases)):
                bias = biases[i]
                weight = weights[i]
                z = dot(weight, activation) + bias
                activation = math_utils.sigmoid(z)
            actual_result = argmax(activation)

            if actual_result == expected_result:
                nums_correct_result += 1

        print('epoch', epoch_i, ' accuracy:', nums_correct_result / len(test_data))

    @staticmethod
    def write_predictions(biases, weights, test_data, test_predictions_filename):
        with open(test_predictions_filename, 'w', encoding='utf-8') as f:
            for data in test_data:
                activation = data[0]

                for i in range(len(biases)):
                    bias = biases[i]
                    weight = weights[i]
                    z = dot(weight, activation) + bias
                    activation = math_utils.sigmoid(z)
                actual_result = argmax(activation)

                f.write(str(actual_result) + '\n')

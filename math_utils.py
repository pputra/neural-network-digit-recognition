from numpy import exp


def sigmoid(z):
    return 1.0 / (1.0 + exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

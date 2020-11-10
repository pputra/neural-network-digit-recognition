from numpy import exp


def sigmoid(z):
    return 1.0 / (1.0 + exp(-z))

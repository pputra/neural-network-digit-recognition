from numpy import exp, sum


def sigmoid(z):
    return 1.0 / (1.0 + exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    return exp(z) / sum(exp(z))


def cross_entropy(x, y):
    return x - y

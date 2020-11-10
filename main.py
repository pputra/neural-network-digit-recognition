from numpy import genfromtxt, zeros
import numpy as np

from neural_network import Network
from nn_config import LAYER_SIZES, MINI_BATCH_SIZE, EPOCHS


def get_training_data():
    train_image = genfromtxt('train_image.csv', delimiter=',', skip_footer=59900)
    train_label = genfromtxt('train_label.csv', dtype='int', delimiter=',', skip_footer=59900)
    train_label_binary = zeros((len(train_label), 10))

    for i in range(train_label.shape[0]):
        val = train_label[i]
        train_label_binary[i][val] = 1

    train_image = np.array([x.reshape(784, 1) for x in train_image])
    train_image /= 255.0
    # print('checking input shape...')
    # print('train image shape: ', train_image.shape)
    # print('train label shape: ', train_label_binary.shape)
    return [[x, y] for x, y in zip(train_image, train_label_binary)]


training_data = get_training_data()

neural_network = Network(LAYER_SIZES)
neural_network.train(training_data, MINI_BATCH_SIZE, EPOCHS)

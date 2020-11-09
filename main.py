from numpy import genfromtxt, zeros

import Network


def get_training_data():
    train_image = genfromtxt('train_image.csv', delimiter=',', skip_footer=59900)
    train_label = genfromtxt('train_label.csv', dtype='int', delimiter=',', skip_footer=59900)
    train_label_binary = zeros((len(train_label), 10))

    for i in range(train_label.shape[0]):
        val = train_label[i]
        train_label_binary[i][val] = 1

    print('checking input shape...')
    print('train image')
    print(train_image.shape)
    print('train label')
    print(train_label_binary.shape)

    return list(zip(train_image, train_label_binary))


training_data = get_training_data()

neural_network = Network.Network([10, 5, 3, 2])

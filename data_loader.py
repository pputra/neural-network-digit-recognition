from numpy import genfromtxt, zeros, array


class DataLoader:
    @staticmethod
    def get_training_data(train_image_dir, train_label_dir, skip_footer = 0):
        train_image = genfromtxt(train_image_dir, delimiter=',', skip_footer=skip_footer)
        train_label = genfromtxt(train_label_dir, dtype='int', delimiter=',', skip_footer=skip_footer)
        train_label_binary = zeros((len(train_label), 10, 1))
        for i in range(train_label.shape[0]):
            val = train_label[i]
            train_label_binary[i][val] = 1
        train_image = array([x.reshape(784, 1) for x in train_image])
        train_image /= 255.0
        return [[x, y] for x, y in zip(train_image, train_label_binary)]

    @staticmethod
    def get_test_data(test_image_dir, test_label_dir, skip_footer = 0):
        test_image = genfromtxt(test_image_dir, delimiter=',', skip_footer=skip_footer)
        test_label = genfromtxt(test_label_dir, dtype='int', delimiter=',', skip_footer=skip_footer)
        test_image = array([x.reshape(784, 1) for x in test_image])
        test_image /= 255.0
        return [[x, y] for x, y in zip(test_image, test_label)]

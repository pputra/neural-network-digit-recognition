from data_loader import DataLoader
from neural_network import Network
from training_parameters import LAYER_SIZES, MINI_BATCH_SIZE, EPOCHS, LEARNING_RATE
from config import VERBOSE, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR,\
    TRAINING_DATA_SKIP_FOOTER, TEST_IMAGE_DIR, TEST_LABEL_DIR, TEST_DATA_SKIP_FOOTER

if VERBOSE:
    print('loading training data...')
training_data = DataLoader.get_training_data(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, TRAINING_DATA_SKIP_FOOTER)

if VERBOSE:
    print('loading test data...')
test_data = DataLoader.get_test_data(TEST_IMAGE_DIR, TEST_LABEL_DIR, TEST_DATA_SKIP_FOOTER)

if VERBOSE:
    print('begin training...')

neural_network = Network(LAYER_SIZES)
neural_network.train(training_data, test_data, MINI_BATCH_SIZE, EPOCHS, LEARNING_RATE, VERBOSE)

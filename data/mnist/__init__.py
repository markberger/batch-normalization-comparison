import gzip
import idx2numpy
import numpy as np
import os

TRAIN_DATA = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_DATA = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

def training_data():
    return read_data(TRAIN_DATA, TRAIN_LABELS)

def test_data():
    return read_data(TEST_DATA, TEST_LABELS)

def read_data(data_fname, labels_fname):
    data_fname = get_full_path(data_fname)
    with gzip.open(data_fname, 'rb') as f:
        X = idx2numpy.convert_from_file(f)

    labels_fname = get_full_path(labels_fname)
    with gzip.open(labels_fname, 'rb') as f:
        Y = idx2numpy.convert_from_file(f)

    # Convert Y from list of scalars to categories
    Y = np.eye(np.unique(Y).shape[0])[Y]

    return X, Y

def get_full_path(fname):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, fname)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# noinspection PyUnresolvedReferences
from six.moves import range
# noinspection PyUnresolvedReferences
from six.moves import cPickle

import os
import gzip
import numpy as np

from tensorflow.python.platform import gfile

from ..utils.data_utils import get_file
from ..datasets import ImageDataset
from ..datasets import Datasets


TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_DEPTH = 1

NUM_CLASSES = 10
VALIDATION_SIZE = 5000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 60000 - VALIDATION_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _extract_data(filepath, data_format='NHWC'):
    print('Extracting', filepath)
    with gfile.Open(filepath, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, filepath))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        if data_format == 'NCHW':
            data.transform((0, 3, 1, 2))
        return data


def _dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def _extract_labels(filepath, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filepath)
    with gfile.Open(filepath, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                           (magic, filepath))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return _dense_to_one_hot(labels, num_classes)
        return labels


def load_data(shuffle=True, dtype='float32', reshape=True, data_format='NHWC'):
    origin = 'http://yann.lecun.com/exdb/mnist/'

    train_dir = get_file('mnist', origin + TRAIN_IMAGES)
    train_data = _extract_data(os.path.join(train_dir, TRAIN_IMAGES),
                               data_format=data_format)

    train_dir = get_file('mnist', origin + TRAIN_LABELS)
    train_labels = _extract_labels(os.path.join(train_dir, TRAIN_LABELS))

    test_dir = get_file('mnist', origin + TEST_IMAGES)
    test_data = _extract_data(os.path.join(test_dir, TEST_IMAGES),
                              data_format=data_format)

    test_dir = get_file('mnist', origin + TEST_LABELS)
    test_labels = _extract_labels(os.path.join(test_dir, TEST_LABELS))

    if shuffle:
        perm = np.arange(train_data.shape[0])
        np.random.shuffle(perm)
        train_data = train_data[perm]
        train_labels = train_labels[perm]

    validation_data = train_data[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    train = ImageDataset(train_data, train_labels, dtype=dtype, reshape=reshape)
    validation = ImageDataset(validation_data, validation_labels, dtype=dtype, reshape=reshape)
    test = ImageDataset(test_data, test_labels, dtype=dtype, reshape=reshape)

    return Datasets(train=train, validation=validation, test=test)

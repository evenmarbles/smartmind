from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import collections

from ..utils import to_dtype

__all__ = ['Dataset',
           'Datasets',
           'ImageDataset']


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class Dataset(object):
    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape[1:]

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def __init__(self, data, labels):
        self._num_examples = data.shape[0]

        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set.

        Parameters
        ----------
        batch_size: int
            The batch size.
        shuffle: bool
            Whether to shuffle the data at the beginning of each epoch.
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            if start < self._num_examples:
                self._index_in_epoch = self._num_examples
            else:
                # Finished epoch
                self._epochs_completed += 1
                if shuffle or shuffle == 'batch':
                    self._shuffle(batch_size, shuffle)

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]

    def _shuffle(self, batch_size, shuffle=True):
        # Shuffle the data
        if shuffle == 'batch':
            perm = self._batch_shuffle(batch_size)
        else:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)

        self._data = self._data[perm]
        self._labels = self._labels[perm]

    def _batch_shuffle(self, batch_size):
        """This shuffles the array in a batch-wise fashion.
        Useful for shuffling HDF5 arrays (where one cannot
        access arbitrary indices).
        """
        index_array = np.arange(self._num_examples)
        batch_count = int(len(index_array) / batch_size)
        # to reshape we need to be cleanly divisible by batch size
        # we stash extra items and re-append them after shuffling
        last_batch = index_array[batch_count * batch_size:]
        index_array = index_array[:batch_count * batch_size]
        index_array = index_array.reshape((batch_count, batch_size))
        np.random.shuffle(index_array)
        index_array = index_array.flatten()
        return np.append(index_array, last_batch)


class ImageDataset(Dataset):
    def __init__(self, data, labels, dtype='float32', reshape=True, preprocess=None):
        dtype = to_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert data.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (data.shape, labels.shape))

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns*depth]
        if reshape:
            data = data.reshape(data.shape[0], np.prod(data.shape[1:]))

        if dtype == tf.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            data = data.astype(np.float32)
            data = np.multiply(data, 1.0 / 255.0)

        super(ImageDataset, self).__init__(data, labels)

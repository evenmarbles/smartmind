from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# noinspection PyUnresolvedReferences
from six.moves import range

import os
import tensorflow as tf

from ..utils.data_utils import get_file
from ..utils.data_utils import read_data_using_reader_op

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24


# Global constants describing the CIFAR-10 data set.
LABEL_BYTES = 1
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

        Parameters
        ----------
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32,
            Minimum number of samples to retain in the queue that
             provides of batches of examples.
        batch_size: int
            Number of images per batch.
        shuffle: boolean
            Indicator whether to use a shuffling queue.

        Returns
        -------
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def load_data():
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

    def pre_process(record_bytes, result):
        """

        Parameters
        ----------
        record_bytes:
        result:

        Returns
        -------
        DataRecord: An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
                for this example.
            label: an int32 Tensor with the label in the range 0..9.
            uint8image: a [height, width, depth] uint8 Tensor with the image data

        """
        result.height = IMAGE_HEIGHT
        result.width = IMAGE_WIDTH
        result.depth = IMAGE_DEPTH

        # The first bytes represent the label, which we convert from uint8->int32.
        result.label = tf.cast(
            tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                                 [result.depth, result.height, result.width])
        # Convert from [depth, height, width] to [height, width, depth].
        result.uint8image = tf.transpose(depth_major, [1, 2, 0])
        return result

    label_bytes = LABEL_BYTES
    image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH

    # Every record consists of a label followed by the data
    record_bytes = label_bytes + image_bytes

    data_dir = get_file('cifar-10-batches-bin', origin, untar=True)
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
    return read_data_using_reader_op(filenames,
                                     reader_params=(record_bytes,),
                                     cb_preprocess=pre_process)


def distorted_inputs(batch_size):
    """Construct distorted inputs for CIFAR training using the Reader ops"""
    read_input = load_data()
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    print('Filling queue with %d CIFAR images before starting to train. '
          'This may take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)

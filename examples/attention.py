import os
import numpy as np
import tensorflow as tf

from PIL import Image

filename = os.path.join('../assets','lena.png')
img1 = Image.open(filename)
img1_arr = np.array(img1)

filename = os.path.join('../assets','grace_hopper_512.jpg')
img2 = Image.open(filename)
img2_arr = np.array(img2)

img_arr_4d = np.stack((img1_arr, img2_arr))
t_img_4d = tf.convert_to_tensor(img_arr_4d, dtype=tf.float32)
# t_img_4d = tf.expand_dims(t_img, 0)

img_placeholder = tf.placeholder(tf.float32, shape=(1, 512, 512, 3))

size = tf.constant([64, 64], shape=[2])
offset = tf.constant([0., 0.], shape=[2, 2])

glimpse = tf.contrib.attention.extract_spatial_glimpse(t_img_4d, size, offset, depth=4)
glimpse_with_feed = tf.contrib.attention.extract_spatial_glimpse(img_placeholder, size, offset)

glimse_old = tf.image.extract_glimpse(t_img_4d, size, offset)

# init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    # sess.run(init_op)

    # value_old = sess.run(glimse_old)
    value = sess.run(glimpse)
    # value_with_feed = sess.run(glimpse_with_feed, feed_dict={img_placeholder: img1_arr[np.newaxis, :]})

    # Start populating the filename queue.

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    #
    # for i in range(1):          # length of your filename list
    #     image = glimpse.eval()  # here is your image Tensor :)

for img1 in value[0]:
    Image.fromarray(np.asarray(img1.astype(np.uint8))).show()

# coord.request_stop()
# coord.join(threads)
#
# Image.fromarray(np.asarray(image[0].astype(np.uint8))).show()


def verifyValues(tensor_in_sizes, glimpse_sizes, offsets, expected_rows, expected_cols):
    """Verifies the output values of the glimpse extraction kernel.

    Args:
      tensor_in_sizes: Input tensor dimensions in [input_rows, input_cols].
      glimpse_sizes: Dimensions of the glimpse in [glimpse_rows, glimpse_cols].
      offsets: Relative location of the center of the glimpse in the input
        image expressed as [row_offset, col_offset].
      expected_rows: A list containing the expected row numbers (None for
         out of bound entries that are expected to be replaced by uniform
         random entries in [0,1) ).
      expected_cols: Same as expected_rows, but for column numbers.
    """

    rows = tensor_in_sizes[0]
    cols = tensor_in_sizes[1]
    # Row Tensor with entries by row.
    # [[ 1 1 1 ... ]
    #  [ 2 2 2 ... ]
    #  [ 3 3 3 ... ]
    #  [ ...
    # ]
    t_rows = tf.tile(
        [[1.0 * r] for r in range(1, rows + 1)], [1, cols],
        name='tile_rows')

    t_rows_3d = tf.reshape(tf.tile(t_rows, (3, 1)), (rows, cols, 3))

    # Shuffle to switch to a convention of (batch_size, height, width, depth).
    t_rows_4d = tf.transpose(
        tf.expand_dims(t_rows_3d, 0), [0, 2, 1, 3])

    # Column Tensor with entries by column.
    # [[ 1 2 3 4 ... ]
    #  [ 1 2 3 4 ... ]
    #  [ 1 2 3 4 ... ]
    #  [ ...         ]
    # ]
    t_cols = tf.tile(
        [[1.0 * r for r in range(1, cols + 1)]],
        [rows, 1], name='tile_cols')

    # Shuffle to switch to a convention of (batch_size, height, width, depth).
    t_cols_4d = tf.transpose(
        tf.expand_dims(
            tf.expand_dims(t_cols, 0), 3), [0, 2, 1, 3])

    # extract_glimpses from Row and Column Tensor, respectively.
    # Switch order for glimpse_sizes and offsets to switch from (row, col)
    # convention to tensorflows (height, width) convention.
    t1 = tf.constant([glimpse_sizes[1], glimpse_sizes[0]], shape=[2])
    t2 = tf.constant([offsets[1], offsets[0]], shape=[1, 2])
    glimpse_rows = (tf.transpose(
        tf.image.extract_glimpse(t_rows_4d, t1, t2), [0, 2, 1, 3]))
    glimpse_cols = (tf.transpose(
        tf.image.extract_glimpse(t_cols_4d, t1, t2), [0, 2, 1, 3]))

    # Evaluate the TensorFlow Graph.
    with tf.Session() as sess:
        # gl = sess.run([glimpse])
        value_rows = sess.run([glimpse_rows])
    pass

    # # Check dimensions of returned glimpse.
    # assertEqual(value_rows.shape[1], glimpse_sizes[0])
    # self.assertEqual(value_rows.shape[2], glimpse_sizes[1])
    # self.assertEqual(value_cols.shape[1], glimpse_sizes[0])
    # self.assertEqual(value_cols.shape[2], glimpse_sizes[1])
    #
    # # Check entries.
    # min_random_val = 0
    # max_random_val = max(rows, cols)
    # for i in range(glimpse_sizes[0]):
    #     for j in range(glimpse_sizes[1]):
    #         if expected_rows[i] is None or expected_cols[j] is None:
    #             self.assertGreaterEqual(value_rows[0][i][j][0], min_random_val)
    #             self.assertLessEqual(value_rows[0][i][j][0], max_random_val)
    #             self.assertGreaterEqual(value_cols[0][i][j][0], min_random_val)
    #             self.assertLessEqual(value_cols[0][i][j][0], max_random_val)
    #         else:
    #             self.assertEqual(value_rows[0][i][j][0], expected_rows[i])
    #             self.assertEqual(value_cols[0][i][j][0], expected_cols[j])


# verifyValues(tensor_in_sizes=[41, 61],
#                    glimpse_sizes=[3, 5],
#                    offsets=[0.0, 0.0],
#                    expected_rows=[20, 21, 22],
#                    expected_cols=[29, 30, 31, 32, 33])

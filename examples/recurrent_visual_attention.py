import argparse
import numpy as np

from smartmind.datasets import mnist

from smartmind.layers import Model
from smartmind.models import Sequential
from smartmind.layers import FullyConnected
from smartmind.layers import Activation
from smartmind.layers import SpatialGlimpse
from smartmind.layers import Flatten
from smartmind.layers import Merge
from smartmind.layers import ReinforceNormal
from smartmind.layers import AttentionRNN
from smartmind.layers import Constant
from smartmind.layers import AddBias
from smartmind.layers import Input


# batch_size = 20  # number of examples per batch
# learning_rate = 0.01  # learning rate at t=0
# min_learning_rate = 0.00001  # minimum learning rate
#
# # glimpse layer
# glimpse_patch_size = 64  # size of glimpse patch at highest res (height = width)
# glimpse_depth = 3  # number of concatenated downscaled patches
# glimpse_scale = 2  # scale of successive patches w.r.t. original input image
# glimpse_hidden_size = 128  # size of glimpse hidden layer
# locator_hidden_size = 128  # size of locator hidden layer
# image_hidden_size = 256  # size of hidden layer combining glimpse and locator hidden layers
#
# # reinforce
# reward_scale = 1  # scale of positive reward (negative is 0)
# unit_pixels = 13  # the locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13)
# locator_std = 0.11  # stddev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)
# stochastic = False  # Reinforce modules forward inputs stochastically during evaluation
#
# # recurrent layer
# rho = 7  # back-propagate through time (BPTT) for rho time-steps
# hidden_size = 256  # number of hidden units used in Simple RNN.
# use_lstm = False  # use LSTM instead of linear layer


def main(args):
    def create_glimpse_network(input_shape, batch_size, locator_hidden_size, glimpse_patch_size,
                               glimpse_depth, glimpse_scale, glimpse_hidden_size,
                               image_hidden_size, hidden_size):
        # create location sensor network
        local_sensor = Sequential()
        local_sensor.add(FullyConnected(locator_hidden_size,
                                        input_shape=(2,), batch_size=batch_size))
        local_sensor.add(Activation('relu'))

        # create glimpse sensor
        glimpse_sensor = Sequential()
        glimpse_sensor.add(SpatialGlimpse(glimpse_patch_size, glimpse_depth, glimpse_scale,
                                          input_shape=input_shape, batch_size=batch_size))
        glimpse_sensor.add(Flatten())
        glimpse_sensor.add(FullyConnected(glimpse_hidden_size))
        glimpse_sensor.add(Activation('relu'))

        # glimpse network (rnn input)
        model_ = Sequential()
        model_.add(Merge([local_sensor, glimpse_sensor], mode='concat'))
        model_.add(FullyConnected(image_hidden_size))
        model_.add(Activation('relu'))
        model_.add(FullyConnected(hidden_size))
        return model_

    def create_location_network(hidden_size, batch_size, locator_std, stochastic):
        # location network (next location to attend to)
        model_ = Sequential()
        model_.add(FullyConnected(2, input_shape=(hidden_size,), batch_size=batch_size))
        model_.add(Activation('hard_tanh'))  # bound mean between -1 and 1
        model_.add(ReinforceNormal(2 * locator_std, stochastic))
        model_.add(Activation('hard_tanh'))  # bound mean between -1 and 1
        return model_

    def create_baseline_reward_predictor(batch_size):
        model_ = Sequential()
        model_.add(Constant(1, input_shape=(1,)))
        stdv = 1. / np.sqrt(batch_size)
        model_.add(AddBias(output_dim=1, init='uniform', init_params={'minval': -stdv, 'maxval': stdv}))
        return model_

    dataset = mnist.load_data(reshape=False)
    input_shape = dataset.train.shape

    num_batches_per_epoch = dataset.train.num_examples / args.batch_size

    if args.num_epochs_per_decay:
        decay_steps = int(num_batches_per_epoch * args.num_epochs_per_decay)
        decay_rate = args.learning_rate_decay_factor
    elif args.saturate_epoch:
        decay_steps = num_batches_per_epoch
        decay_rate = (args.min_learning_rate - args.learning_rate) / args.saturate_epoch
    else:
        raise Exception("Either `num_epochs_per_decay` must be specified or `saturate_epochs`.")

    print('Build model...')
    glimpse = create_glimpse_network(input_shape, args.batch_size, args.locator_hidden_size, args.glimpse_patch_size,
                                     args.glimpse_depth, args.glimpse_scale, args.glimpse_hidden_size,
                                     args.image_hidden_size, args.hidden_size)
    locator = create_location_network(args.hidden_size, args.batch_size, args.locator_std, args.stochastic)

    input_ = Input(shape=input_shape)

    # model is a reinforcement learning agent
    agent = AttentionRNN(args.hidden_size, glimpse, locator,
                         cell='BasicLSTMCell' if args.use_lstm else 'BasicRNNCell',
                         sequence_length=args.rho,
                         activation='relu')(input_)

    # classifier
    agent = FullyConnected(mnist.NUM_CLASSES)(agent)

    reward_predictor = create_baseline_reward_predictor(args.batch_size)
    baseline = reward_predictor(agent)

    model = Model(input_=[input_], output=[agent, [agent, baseline]])
    model.compile(
        optimizer=('sgd', decay_steps, decay_rate, args.min_learning_rate, True, {'learning_rate': args.learning_rate}),
        loss=['sparse_categorical_xentropy', 'vr_class_reward'],
        metrics=['accuracy'])

    print('Training...')
    model.fit(dataset.train.data,
              [dataset.train.labels, dataset.train.labels], args.batch_size,
              n_epoch=args.max_epoch,
              verbose=args.verbosity,
              validation_data=(dataset.validation.data, [dataset.validation.labels, dataset.validation.labels]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train a Recurrent Model for Visual Attention')
    ap.add_argument('--batch_size', type=int, default=20, help='Number of examples per batch.')
    ap.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate at t=0.')
    ap.add_argument('--learning_rate_decay_factor', type=float, default=0.1,
                    help='The factor at which the learning rate decays.')
    ap.add_argument('--num_epochs_per_decay', type=int, help='Epochs after which learning rate decays.')
    ap.add_argument('--min_learning_rate', type=float, help='Minimum learning rate.')
    ap.add_argument('--saturate_epoch', type=int,
                    help='Epoch at which linear decayed learning rate will reach minimum learning rate.')
    ap.add_argument('--max_epoch', type=int, default=2000, help='Maximum number of epochs to run.')
    ap.add_argument('--max_tries', type=int, default=100,
                    help='Maximum number of epochs to try to find a better local minima for early-stopping.')
    ap.add_argument('--activation', type=str, default='relu', help='Activation function.')
    ap.add_argument('--verbosity', type=int, default=1, help='Increase output verbosity.')

    # glimpse layer
    ap.add_argument('--glimpse_patch_size', type=int, default=8,
                    help='Size of glimpse patch at highest res (height = width).')
    ap.add_argument('--glimpse_depth', type=int, default=3, help='Number of concatenated downscaled patches.')
    ap.add_argument('--glimpse_scale', type=int, default=2,
                    help='Scale of successive patches w.r.t. original input image.')
    ap.add_argument('--glimpse_hidden_size', type=int, default=128, help='Size of glimpse hidden layer.')
    ap.add_argument('--locator_hidden_size', type=int, default=128, help='Size of locator hidden layer.')
    ap.add_argument('--image_hidden_size', type=int, default=256,
                    help='Size of hidden layer combining glimpse and locator hidden layers.')

    # reinforce
    ap.add_argument('--reward_scale', type=int, default=1, help='Scale of positive reward (negative is 0).')
    ap.add_argument('--unit_pixels', type=int, default=13,
                    help='The locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13).')
    ap.add_argument('--locator_std', type=float, default=0.11,
                    help='Stddev of gaussian location sampler (between 0 and 1) (low values may cause NaNs).')
    ap.add_argument('--stochastic', action='store_true',
                    help='Reinforce modules forward inputs stochastically during evaluation.')

    # recurrent layer
    ap.add_argument('--rho', type=int, default=7, help='Back-propagate through time (BPTT) for rho time-steps.')
    ap.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units used in Simple RNN.')
    ap.add_argument('--use_lstm', action='store_true', help='Use LSTM instead of linear layer.')

    # data
    ap.add_argument('--dataset', type=str, default='Mnist',
                    help='Which dataset to use : Mnist | TranslattedMnist | etc.')
    ap.add_argument('--train_epoch_size', type=int, default=-1,
                    help='Number of train examples seen between each epoch.')
    ap.add_argument('--valid_epoch_size', type=int, default=-1,
                    help='Number of valid examples used for early stopping and cross-validation.')

    try:
        main(ap.parse_args())
    except Exception as e:
        pass

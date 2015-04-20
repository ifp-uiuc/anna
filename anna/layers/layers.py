import sys
import os
import cPickle as pickle

import numpy
import theano.tensor as T
import theano

from theano.tensor.signal.conv import conv2d as sconv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet.conv import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.dnn import GpuDnnConv

srng = RandomStreams()


class data_order(object):
    _MINIBATCH_SIZE = 'minibatch-size'
    _WIDTH = 'width'
    _HEIGHT = 'height'
    _CHANNELS = 'num-channels'
    # data order, bc01 type used by theano
    type1 = (_MINIBATCH_SIZE, _CHANNELS, _WIDTH, _HEIGHT)
    # data order, c01b type used by cuda_convnet
    type2 = (_CHANNELS, _WIDTH, _HEIGHT, _MINIBATCH_SIZE)

# nonlinearities

sigmoid = T.nnet.sigmoid

tanh = T.tanh


def identity(x):
    # To create a linear layer.
    return x


def rectify(x):
    return T.maximum(x, 0.0)


def trec(x):
    return x * (x > 1)


def softmax(x):
    return T.nnet.softmax(x)


def all_layers(layer):
    """
    Recursive function to gather all layers below the given layer (including
    the given layer)
    """
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
        return [layer]
    elif isinstance(layer, ConcatenateLayer):
        return sum([all_layers(i) for i in layer.input_layers], [layer])
    else:
        return [layer] + all_layers(layer.input_layer)


def all_parameters(layer):
    """
    Recursive function to gather all parameters, starting from the output layer
    """
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
        return []
    elif isinstance(layer, ConcatenateLayer):
        return sum([all_parameters(i) for i in layer.input_layers], [])
    else:
        return layer.params + all_parameters(layer.input_layer)


def all_trainable_parameters(layer):
    """
    Recursive function to gather all training parameters, starting from the
    output layer
    """
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
        return []
    elif isinstance(layer, ConcatenateLayer):
        return sum([all_trainable_parameters(i)
                    for i in layer.input_layers], [])
    else:
        if layer.trainable:
            return layer.params + all_trainable_parameters(layer.input_layer)
        else:
            return [] + all_trainable_parameters(layer.input_layer)


def all_bias_parameters(layer):
    """
    Recursive function to gather all bias parameters, starting from the output
    layer
    """
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
        return []
    elif isinstance(layer, ConcatenateLayer):
        return sum([all_bias_parameters(i) for i in layer.input_layers], [])
    else:
        return layer.bias_params + all_bias_parameters(layer.input_layer)


def all_non_bias_parameters(layer):
    return [p for p in all_parameters(layer)
            if p not in all_bias_parameters(layer)]


def gather_rescaling_updates(layer, c):
    """
    Recursive function to gather weight rescaling updates when the constant is
    the same for all layers.
    """
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
        return []
    elif isinstance(layer, ConcatenateLayer):
        return sum([gather_rescaling_updates(i, c)
                    for i in layer.input_layers], [])
    else:
        if hasattr(layer, 'rescaling_updates'):
            updates = layer.rescaling_updates(c)
        else:
            updates = []
        return updates + gather_rescaling_updates(layer.input_layer, c)


def get_param_values(layer):
    params = all_parameters(layer)
    return [p.get_value() for p in params]


def set_param_values(layer, param_values):
    params = all_parameters(layer)
    for p, pv in zip(params, param_values):
        p.set_value(pv)


def reset_all_params(layer):
    for l in all_layers(layer):
        if hasattr(l, 'reset_params'):
            l.reset_params()


def gen_updates_regular_momentum(loss, all_parameters, learning_rate, momentum,
                                 weight_decay):
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(param_i.get_value() * 0.)
        v = (momentum * mparam_i - weight_decay * learning_rate * param_i -
             learning_rate * grad_i)
        updates.append((mparam_i, v))
        updates.append((param_i, param_i + v))
    return updates


# using the alternative formulation of nesterov momentum described at
# https://github.com/lisa-lab/pylearn2/pull/136
# such that the gradient can be evaluated at the current parameters.
def gen_updates_nesterov_momentum(loss, all_parameters, learning_rate,
                                  momentum, weight_decay):
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(param_i.get_value() * 0.)
        full_grad = grad_i + weight_decay * param_i
        # new momemtum
        v = momentum * mparam_i - learning_rate * full_grad
        # new parameter values
        w = param_i + momentum * v - learning_rate * full_grad
        updates.append((mparam_i, v))
        updates.append((param_i, w))
    return updates


def gen_updates_sgd(loss, all_parameters, learning_rate):
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        updates.append((param_i, param_i - learning_rate * grad_i))
    return updates


def gen_updates_adagrad(loss, all_parameters, learning_rate=1.0, epsilon=1e-6):
    """
    epsilon is not included in the typical formula,

    See "Notes on AdaGrad" by Chris Dyer for more info.
    """
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    # initialise to zeroes with the right shape
    all_accumulators = [theano.shared(param.get_value() * 0.)
                        for param in all_parameters]

    updates = []
    for param_i, grad_i, acc_i in zip(all_parameters, all_grads,
                                      all_accumulators):
        acc_i_new = acc_i + grad_i ** 2
        updates.append((acc_i, acc_i_new))
        updates.append(
            (param_i,
             param_i - learning_rate * grad_i / T.sqrt(acc_i_new + epsilon)))

    return updates


def gen_updates_rmsprop(loss, all_parameters,
                        learning_rate=1.0,
                        rho=0.9,
                        epsilon=1e-6):
    """
    epsilon is not included in Hinton's video, but to prevent problems with
    relus repeatedly having 0 gradients, it is included here.

    Watch this video for more info: http://www.youtube.com/watch?v=O3sxAc4hxZU
    (formula at 5:20)

    also check http://climin.readthedocs.org/en/latest/rmsprop.html
    """
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    # initialise to zeroes with the right shape
    all_accumulators = [theano.shared(param.get_value() * 0.)
                        for param in all_parameters]

    updates = []
    for param_i, grad_i, acc_i in zip(all_parameters, all_grads,
                                      all_accumulators):
        acc_i_new = rho * acc_i + (1 - rho) * grad_i ** 2
        updates.append((acc_i, acc_i_new))
        updates.append(
            (param_i,
             param_i - learning_rate * grad_i / T.sqrt(acc_i_new + epsilon)))

    return updates


def gen_updates_adadelta(loss, all_parameters,
                         learning_rate=1.0,
                         rho=0.95,
                         epsilon=1e-6):
    """
    in the paper, no learning rate is considered (so learning_rate=1.0).
    Probably best to keep it at this value. epsilon is important for the very
    first update (so the numerator does not become 0).

    rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to work
    for multiple datasets (MNIST, speech).

    see "Adadelta: an adaptive learning rate method" by Matthew Zeiler for more
    info.
    """
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    # initialise to zeroes with the right shape
    all_accumulators = [theano.shared(param.get_value() * 0.)
                        for param in all_parameters]
    all_delta_accumulators = [theano.shared(param.get_value() * 0.)
                              for param in all_parameters]

    # all_accumulators: accumulate gradient magnitudes
    # all_delta_accumulators: accumulate update magnitudes (recursive!)

    updates = []
    for param_i, grad_i, acc_i, acc_delta_i in zip(all_parameters, all_grads,
                                                   all_accumulators,
                                                   all_delta_accumulators):
        acc_i_new = rho * acc_i + (1 - rho) * grad_i ** 2
        updates.append((acc_i, acc_i_new))

        update_i = (grad_i * T.sqrt(acc_delta_i + epsilon) /
                    T.sqrt(acc_i_new + epsilon))
        updates.append((param_i, param_i - learning_rate * update_i))

        acc_delta_i_new = rho * acc_delta_i + (1 - rho) * update_i ** 2
        updates.append((acc_delta_i, acc_delta_i_new))

    return updates


def shared_single(dim=2):
    """
    Shortcut to create an undefined single precision Theano shared variable.
    """
    shp = tuple([1] * dim)
    return theano.shared(numpy.zeros(shp, dtype='float32'))


# TODO(tpaine) remove this
def sparse_initialisation(n_inputs, n_outputs, sparsity=0.05, std=0.01):
    """
    sparsity: fraction of the weights to each output unit that should be
    nonzero
    """
    weights = numpy.zeros((n_inputs, n_outputs), dtype='float32')
    size = int(sparsity * n_inputs)
    for k in xrange(n_outputs):
        indices = numpy.arange(n_inputs)
        numpy.random.shuffle(indices)
        indices = indices[:size]
        values = numpy.random.randn(size).astype(numpy.float32) * std
        weights[indices, k] = values

    return weights


class Layer(object):
    def __init__(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement this method")

    def get_output_shape(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement this method")

    def output(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement this method")

    def reset_params(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement this method")


class InputLayer(Layer):
    def __init__(self, mb_size, n_features, length):
        self.mb_size = mb_size
        self.n_features = n_features
        self.length = length
        self.input_var = T.ftensor3('input')

    def get_output_shape(self):
        return (self.mb_size, self.n_features, self.length)

    def output(self, *args, **kwargs):
        """
        return theano variable
        """
        return self.input_var


class FlatInputLayer(InputLayer):
    def __init__(self, mb_size, n_features):
        self.mb_size = mb_size
        self.n_features = n_features
        self.input_var = T.fmatrix('input')

    def get_output_shape(self):
        return (self.mb_size, self.n_features)

    def output(self, *args, **kwargs):
        """
        return theano variable
        """
        return self.input_var


class Input2DLayer(Layer):
    def __init__(self, mb_size, n_features, width, height):
        self.mb_size = mb_size
        self.n_features = n_features
        self.width = width
        self.height = height
        self.input_var = T.ftensor4('input')

        self.data_order = data_order.type1

    def get_output_shape(self):
        return (self.mb_size, self.n_features, self.width, self.height)

    def output(self, *args, **kwargs):
        return self.input_var


class PoolingLayer(Layer):
    def __init__(self, input_layer, ds_factor, ignore_border=False):
        self.ds_factor = ds_factor
        self.input_layer = input_layer
        self.ignore_border = ignore_border
        self.trainable = True
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

    def get_output_shape(self):
        output_shape = list(self.input_layer.get_output_shape())
        if self.ignore_border:
            output_shape[-1] = int(numpy.floor(float(output_shape[-1]) /
                                               self.ds_factor))
        else:
            output_shape[-1] = int(numpy.ceil(float(output_shape[-1]) /
                                              self.ds_factor))
        return tuple(output_shape)

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        return max_pool_2d(input, (1, self.ds_factor), self.ignore_border)


class Pool2DLayer(object):
    def __init__(self,
                 input,
                 filter_size,
                 stride=(2, 2)):

        self.input_layer = input
        self.filter_size = filter_size
        self.n_channels = self.input_layer.n_features
        self.n_features = self.n_channels
        self.stride = stride

        self.mb_size = self.input_layer.mb_size

        self.trainable = False
        self.params = []

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        filter_shape = (self.n_features, self.n_channels,
                        self.filter_size, self.filter_size)
        return GpuDnnConv.get_out_shape(input_shape,
                                        filter_shape,
                                        'valid',
                                        self.stride)

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        input = self.input_layer.output()
        ws = (self.filter_size, self.filter_size)

        contiguous_input = gpu_contiguous(input)
        output = theano.sandbox.cuda.dnn.dnn_pool(contiguous_input,
                                                  ws,
                                                  stride=self.stride)
        return output


class GlobalPooling2DLayer(Layer):
    """
    Global pooling across the entire feature map, useful in NINs.
    """

    def __init__(self, input_layer,
                 pooling_function='mean',
                 nonlinearity=softmax):
        self.input_layer = input_layer
        self.pooling_function = pooling_function
        self.trainable = True
        self.nonlinearity = nonlinearity

        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.data_order = data_order.type1

        assert (len(self.input_layer.get_output_shape()) == 4), \
            'Input must have 4 dimensions.'

        assert (self.input_layer.data_order == self.data_order), \
            'Input data order does not match this layer\'s data order.'

    def get_output_shape(self):
        # Removes the last 2 dimensions
        return self.input_layer.get_output_shape()[:2]

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        if self.pooling_function == 'mean':
            out = input.mean([2, 3])
        elif self.pooling_function == 'max':
            out = input.max([2, 3])
        elif self.pooling_function == 'l2':
            out = T.sqrt((input ** 2).mean([2, 3]))

        return self.nonlinearity(out)


class DenseLayer(Layer):
    def __init__(self, input_layer, n_outputs, weights_std, init_bias_value,
                 nonlinearity=rectify,
                 dropout=0.):
        self.n_outputs = n_outputs
        self.input_layer = input_layer
        self.weights_std = numpy.float32(weights_std)
        self.init_bias_value = numpy.float32(init_bias_value)
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.mb_size = self.input_layer.mb_size

        input_shape = self.input_layer.get_output_shape()
        self.n_inputs = int(numpy.prod(input_shape[1:]))
        self.flatinput_shape = (self.mb_size, self.n_inputs)

        self.W = shared_single(2)
        self.b = shared_single(1)
        self.trainable = True
        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()

    def reset_params(self):
        self.W.set_value(
            numpy.random.randn(self.n_inputs, self.n_outputs).astype(
                numpy.float32) * self.weights_std)
        self.b.set_value(numpy.ones(self.n_outputs).astype(numpy.float32) *
                         self.init_bias_value)

    def get_output_shape(self):
        return (self.mb_size, self.n_outputs)

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        # use the 'dropout_active' keyword argument to disable it at test time.
        # It is on by default.
        if input is None:
            input = self.input_layer.output(dropout_active=dropout_active,
                                            *args, **kwargs)
        if len(self.input_layer.get_output_shape()) > 2:
            input = input.reshape(self.flatinput_shape)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            input = (input / retain_prob *
                     srng.binomial(input.shape,
                                   p=retain_prob,
                                   dtype='int32').astype('float32'))
            # apply the input mask and rescale the input accordingly.
            # By doing this it's no longer necessary to rescale the weights
            # at test time.

        return self.nonlinearity(
            T.dot(input, self.W) + self.b.dimshuffle('x', 0))


class DenseNoBiasLayer(Layer):
    def __init__(self, input_layer, n_outputs, weights_std,
                 nonlinearity=rectify,
                 dropout=0.):
        self.n_outputs = n_outputs
        self.input_layer = input_layer
        self.weights_std = numpy.float32(weights_std)
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.mb_size = self.input_layer.mb_size

        input_shape = self.input_layer.get_output_shape()
        self.n_inputs = int(numpy.prod(input_shape[1:]))
        self.flatinput_shape = (self.mb_size, self.n_inputs)

        self.W = shared_single(2)
        self.trainable = True
        self.params = [self.W]
        self.reset_params()

    def reset_params(self):
        self.W.set_value(
            numpy.random.randn(self.n_inputs, self.n_outputs).astype(
                numpy.float32) * self.weights_std)

    def get_output_shape(self):
        return (self.mb_size, self.n_outputs)

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        # use the 'dropout_active' keyword argument to disable it at test time.
        # It is on by default.
        if input is None:
            input = self.input_layer.output(dropout_active=dropout_active,
                                            *args, **kwargs)
        if len(self.input_layer.get_output_shape()) > 2:
            input = input.reshape(self.flatinput_shape)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            input = (input / retain_prob *
                     srng.binomial(input.shape,
                                   p=retain_prob,
                                   dtype='int32').astype('float32'))
            # apply the input mask and rescale the input accordingly.
            # By doing this it's no longer necessary to rescale the weights
            # at test time.

        return self.nonlinearity(T.dot(input, self.W))


class DenseBatchNormLayer(Layer):
    def __init__(self, input):

        self.input_layer = input
        self.epsilon = 1e-5
        self.gamma = shared_single(1)
        self.beta = shared_single(1)

        self.mb_size = self.input_layer.mb_size

        self.trainable = False
        self.params = [self.gamma, self.beta]
        self.reset_params()

    def get_output_shape(self):
        return self.input_layer.get_output_shape()

    def output(self, input=None, *args, **kwargs):
        mean = T.mean(self.input_layer.output(), axis=0, keepdims=True)
        std = T.std(self.input_layer.output(), axis=0, keepdims=True)
        x = (self.input_layer.output() - mean)/(std + self.epsilon)

        gamma = self.gamma.dimshuffle('x', 0)
        beta = self.beta.dimshuffle('x', 0)
        y = gamma * x + beta

        output = y
        return output

    def reset_params(self):
        _, num_features = self.get_output_shape()
        gamma_values = numpy.ones((num_features, )).astype(numpy.float32)
        self.gamma.set_value(gamma_values)

        beta_values = numpy.zeros((num_features, )).astype(numpy.float32)
        self.beta.set_value(beta_values)


class Conv2DLayer(object):
    def __init__(self,
                 input,
                 n_features,
                 filter_size,
                 weights_std,
                 stride=(1, 1),
                 trainable=True):

        self.input_layer = input
        self.n_features = n_features
        self.filter_size = filter_size
        self.n_channels = self.input_layer.n_features

        self.filter_shape = (self.n_features,
                             self.n_channels,
                             self.filter_size,
                             self.filter_size)

        self.weights_std = weights_std
        self.stride = stride

        self.mb_size = self.input_layer.mb_size

        self.trainable = trainable

        self.W = shared_single(4)
        self.params = [self.W]
        self.reset_params()

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        return GpuDnnConv.get_out_shape(input_shape,
                                        self.filter_shape,
                                        'valid',
                                        self.stride)

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        input = self.input_layer.output()
        contiguous_input = gpu_contiguous(input)
        contiguous_filters = gpu_contiguous(self.W)
        # output = theano.sandbox.cuda.dnn.dnn_conv(contiguous_input,
        #                                           contiguous_filters,
        #                                           subsample=self.stride)
        output = T.nnet.conv2d(contiguous_input, contiguous_filters,
                                    subsample=self.stride)
        return output

    def reset_params(self):
        self.W.set_value(numpy.random.randn(*self.filter_shape).astype(
            numpy.float32) * self.weights_std)


class ConvBatchNormLayer(Layer):
    def __init__(self, input):

        self.input_layer = input
        self.epsilon = 1e-5
        self.gamma = shared_single(1)
        self.beta = shared_single(1)

        self.n_channels = self.input_layer.n_features
        self.n_features = self.n_channels
        self.mb_size = self.input_layer.mb_size

        self.trainable = False
        self.params = [self.gamma, self.beta]
        self.reset_params()

    def get_output_shape(self):
        return self.input_layer.get_output_shape()

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        mean = T.mean(self.input_layer.output(), axis=(0, 2, 3), keepdims=True)
        std = T.std(self.input_layer.output(), axis=(0, 2, 3), keepdims=True)
        x = (self.input_layer.output() - mean)/(std + self.epsilon)

        gamma = self.gamma.dimshuffle('x', 0, 'x', 'x')
        beta = self.beta.dimshuffle('x', 0, 'x', 'x')
        y = gamma * x + beta
        output = y
        return output

    def reset_params(self):
        _, num_features, _, _ = self.get_output_shape()
        gamma_values = numpy.ones((num_features, )).astype(numpy.float32)
        self.gamma.set_value(gamma_values)

        beta_values = numpy.zeros((num_features, )).astype(numpy.float32)
        self.beta.set_value(beta_values)


class ConcatenateLayer(Layer):
    def __init__(self, input_layers, nonlinearity=identity):
        self.input_layers = input_layers
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layers[0].mb_size
        self.nonlinearity = nonlinearity

    def get_output_shape(self):
        # this assumes the layers are already flat!
        sizes = [i.get_output_shape()[1] for i in self.input_layers]
        return (self.mb_size, sum(sizes))

    def output(self, *args, **kwargs):
        inputs = [i.output(*args, **kwargs) for i in self.input_layers]
        return self.nonlinearity(T.concatenate(inputs, axis=1))


class NonlinearityLayer(Layer):
    def __init__(self, input, nonlinearity=identity):
        self.input_layer = input
        self.nonlinearity = nonlinearity
        self.n_channels = self.input_layer.n_features
        self.n_features = self.n_channels
        self.mb_size = self.input_layer.mb_size

        self.trainable = False
        self.params = []

    def get_output_shape(self):
        return self.input_layer.get_output_shape()

    def output(self, *args, **kwargs):
        input = self.input_layer.output()
        return self.nonlinearity(input)
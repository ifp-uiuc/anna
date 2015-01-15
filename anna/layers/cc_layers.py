"""
Layers using the cuda-convnet Theano wrappers that are part of pylearn2.
"""

import theano
import theano.tensor as T
import numpy
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.img_acts import ImageActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool, MaxPoolGrad
from pylearn2.sandbox.cuda_convnet.stochastic_pool import StochasticMaxPool
from pylearn2.sandbox.cuda_convnet.stochastic_pool import WeightedMaxPool
from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda import host_from_gpu
from theano.tensor import as_tensor_variable

import layers

# TODO(tpaine) refactor the convolution layers to get rid of code repitition.


class Input2DLayer(layers.Input2DLayer):
    """
    Like Input2DLayer, but the data is expected to be in c01b order instead of
    bc01.
    """
    def get_output_shape(self):
        # c01b instead of bc01
        return (self.n_features, self.width, self.height, self.mb_size)


class Conv2DLayer(object):
    def __init__(self,
                 input_layer,
                 n_filters,
                 filter_size,
                 weights_std,
                 init_bias_value,
                 stride=1,
                 nonlinearity=layers.rectify,
                 dropout=0.,
                 partial_sum=None,
                 pad=0,
                 untie_biases=False,
                 trainable=True):
        """
        Only the valid border mode is supported.

        n_filters should be a multiple of 16
        """
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.n_filters = n_filters
        n_channels = self.input_shape[0]
        self.n_channels = n_channels
        self.filter_size = filter_size
        self.weights_std = numpy.float32(weights_std)
        self.init_bias_value = numpy.float32(init_bias_value)
        self.stride = stride
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.partial_sum = partial_sum
        self.pad = pad
        self.untie_biases = untie_biases
        # if untie_biases == True, each position in the output map has its own
        # bias (as opposed to having the same bias everywhere for a given
        # filter)
        self.mb_size = self.input_layer.mb_size

        self.filter_shape = (n_channels, filter_size, filter_size, n_filters)

        self.trainable = trainable
        self.W = layers.shared_single(4)

        if self.untie_biases:
            self.b = layers.shared_single(3)
        else:
            self.b = layers.shared_single(1)

        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()

        self.filter_acts_op = FilterActs(stride=self.stride,
                                         partial_sum=self.partial_sum,
                                         pad=self.pad)

    def reset_params(self):
        self.W.set_value(numpy.random.randn(*self.filter_shape).astype(
            numpy.float32) * self.weights_std)

        if self.untie_biases:
            self.b.set_value(
                numpy.ones(self.get_output_shape()[:3]).astype(numpy.float32)
                * self.init_bias_value)
        else:
            self.b.set_value(numpy.ones(self.n_filters).astype(numpy.float32)
                             * self.init_bias_value)

    def get_output_shape(self):
        output_width = int(numpy.ceil((
            self.input_shape[1] + 2 * self.pad - self.filter_size
            + self.stride)*1.0 / self.stride))
        output_height = int(numpy.ceil((
            self.input_shape[2] + 2 * self.pad - self.filter_size
            + self.stride)*1.0 / self.stride))
        output_shape = (self.n_filters, output_width, output_height,
                        self.mb_size)
        return output_shape

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input is None:
            input = self.input_layer.output(dropout_active=dropout_active,
                                            *args, **kwargs)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            mask = layers.srng.binomial(input.shape, p=retain_prob,
                                        dtype='int32').astype('float32')
            # apply the input mask and rescale the input accordingly.
            # By doing this it's no longer necessary to rescale the weights
            # at test time.
            input = input / retain_prob * mask

        contiguous_input = gpu_contiguous(input)
        contiguous_filters = gpu_contiguous(self.W)
        conved = self.filter_acts_op(contiguous_input, contiguous_filters)

        if self.untie_biases:
            conved += self.b.dimshuffle(0, 1, 2, 'x')
        else:
            conved += self.b.dimshuffle(0, 'x', 'x', 'x')

        return self.nonlinearity(conved)


class Conv2DNoBiasLayer(object):
    def __init__(self,
                 input_layer,
                 n_filters,
                 filter_size,
                 weights_std,
                 stride=1,
                 nonlinearity=layers.rectify,
                 dropout=0.,
                 partial_sum=None,
                 pad=0,
                 trainable=True):
        """
        Only the valid border mode is supported.

        n_filters should be a multiple of 16
        """
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.n_filters = n_filters
        n_channels = self.input_shape[0]
        self.n_channels = n_channels
        self.filter_size = filter_size
        self.weights_std = numpy.float32(weights_std)
        self.stride = stride
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.partial_sum = partial_sum
        self.pad = pad
        self.mb_size = self.input_layer.mb_size

        self.filter_shape = (n_channels, filter_size, filter_size, n_filters)

        self.trainable = trainable
        self.W = layers.shared_single(4)

        self.params = [self.W]
        self.reset_params()

        self.filter_acts_op = FilterActs(stride=self.stride,
                                         partial_sum=self.partial_sum,
                                         pad=self.pad)

    def reset_params(self):
        self.W.set_value(numpy.random.randn(*self.filter_shape).astype(
            numpy.float32) * self.weights_std)

    def get_output_shape(self):
        output_width = int(numpy.ceil((
            self.input_shape[1] + 2 * self.pad - self.filter_size
            + self.stride)*1.0 / self.stride))
        output_height = int(numpy.ceil((
            self.input_shape[2] + 2 * self.pad - self.filter_size
            + self.stride)*1.0 / self.stride))
        output_shape = (self.n_filters, output_width, output_height,
                        self.mb_size)
        return output_shape

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input is None:
            input = self.input_layer.output(dropout_active=dropout_active,
                                            *args, **kwargs)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            mask = layers.srng.binomial(input.shape, p=retain_prob,
                                        dtype='int32').astype('float32')
            # apply the input mask and rescale the input accordingly.
            # By doing this it's no longer necessary to rescale the weights
            # at test time.
            input = input / retain_prob * mask

        contiguous_input = gpu_contiguous(input)
        contiguous_filters = gpu_contiguous(self.W)
        conved = self.filter_acts_op(contiguous_input, contiguous_filters)

        return self.nonlinearity(conved)


class Deconv2DLayer(object):
    def __init__(self,
                 input_layer,
                 mirror_layer,
                 nonlinearity=None):
        """
        Only the valid border mode is supported.

        n_filters should be a multiple of 16
        """

        self.mirror_layer = mirror_layer

        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        n_filters = self.input_shape[0]

        if nonlinearity:
            self.nonlinearity = nonlinearity
        else:
            self.nonlinearity = mirror_layer.nonlinearity

        self.n_channels = mirror_layer.n_channels
        self.n_filters = mirror_layer.n_filters
        self.filter_size = mirror_layer.filter_size
        self.weights_std = mirror_layer.weights_std
        self.init_bias_value = mirror_layer.init_bias_value
        self.stride = mirror_layer.stride
        self.dropout = mirror_layer.dropout
        self.partial_sum = mirror_layer.partial_sum
        self.pad = mirror_layer.pad
        self.untie_biases = mirror_layer.untie_biases
        # if untie_biases == True, each position in the output map has its own
        # bias (as opposed to having the same bias everywhere for a filter)
        self.mb_size = self.input_layer.mb_size

        self.filter_shape = mirror_layer.filter_shape

        self.trainable = False
        self.W = mirror_layer.W

        self.b = mirror_layer.b

        # self.params = [self.W, self.b]
        self.params = []
        self.bias_params = [self.b]

        self.image_acts_op = ImageActs(stride=self.stride,
                                       partial_sum=self.partial_sum,
                                       pad=self.pad)

    def get_output_shape(self):
        output_shape = self.mirror_layer.input_layer.get_output_shape()
        return output_shape

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input is None:
            input = self.input_layer.output(dropout_active=dropout_active,
                                            *args, **kwargs)

        if self.untie_biases:
            input -= self.b.dimshuffle(0, 1, 2, 'x')
        else:
            input -= self.b.dimshuffle(0, 'x', 'x', 'x')

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            mask = layers.srng.binomial(input.shape, p=retain_prob,
                                        dtype='int32').astype('float32')
            # apply the input mask and rescale the input accordingly.
            # By doing this it's no longer necessary to rescale the weights
            # at test time.
            input = input / retain_prob * mask

        contiguous_input = gpu_contiguous(input)
        contiguous_filters = gpu_contiguous(self.W)
        if self.stride == 1:
            deconved = self.image_acts_op(contiguous_input, contiguous_filters)
        else:
            _, x, y, _ = self.get_output_shape()
            deconved = self.image_acts_op(contiguous_input, contiguous_filters,
                                          as_tensor_variable((x, y)))
        return self.nonlinearity(deconved)


class DeconvUntied2DLayer(object):
    def __init__(self,
                 input_layer,
                 mirror_layer,
                 nonlinearity=None):
        """
        Only the valid border mode is supported.

        n_filters should be a multiple of 16
        """

        self.mirror_layer = mirror_layer

        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        n_filters = self.input_shape[0]

        if nonlinearity:
            self.nonlinearity = nonlinearity
        else:
            self.nonlinearity = mirror_layer.nonlinearity

        self.n_channels = mirror_layer.n_channels
        self.n_filters = mirror_layer.n_filters
        self.filter_size = mirror_layer.filter_size
        self.weights_std = mirror_layer.weights_std
        self.init_bias_value = mirror_layer.init_bias_value
        self.stride = mirror_layer.stride
        self.dropout = mirror_layer.dropout
        self.partial_sum = mirror_layer.partial_sum
        self.pad = mirror_layer.pad
        self.untie_biases = mirror_layer.untie_biases

        self.mb_size = self.input_layer.mb_size

        self.filter_shape = mirror_layer.filter_shape

        self.trainable = False
        self.W = layers.shared_single(4)

        if self.untie_biases:
            self.b = layers.shared_single(3)
        else:
            self.b = layers.shared_single(1)

        # self.params = [self.W, self.b]
        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()

        self.image_acts_op = ImageActs(stride=self.stride,
                                       partial_sum=self.partial_sum,
                                       pad=self.pad)

    def reset_params(self):
        self.W.set_value(numpy.random.randn(*self.filter_shape).astype(
            numpy.float32) * self.weights_std)

        if self.untie_biases:
            self.b.set_value(
                numpy.ones(self.get_output_shape()[:3]).astype(numpy.float32)
                * self.init_bias_value)
        else:
            self.b.set_value(numpy.ones(self.n_filters).astype(numpy.float32)
                             * self.init_bias_value)

    def get_output_shape(self):
        output_shape = self.mirror_layer.input_layer.get_output_shape()
        return output_shape

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input is None:
            input = self.input_layer.output(dropout_active=dropout_active,
                                            *args, **kwargs)

        if self.untie_biases:
            input -= self.b.dimshuffle(0, 1, 2, 'x')
        else:
            input -= self.b.dimshuffle(0, 'x', 'x', 'x')

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            mask = layers.srng.binomial(input.shape, p=retain_prob,
                                        dtype='int32').astype('float32')
            # apply the input mask and rescale the input accordingly.
            # By doing this it's no longer necessary to rescale the weights
            # at test time.
            input = input / retain_prob * mask

        contiguous_input = gpu_contiguous(input)
        contiguous_filters = gpu_contiguous(self.W)
        if self.stride == 1:
            deconved = self.image_acts_op(contiguous_input, contiguous_filters)
        else:
            _, x, y, _ = self.get_output_shape()
            deconved = self.image_acts_op(contiguous_input, contiguous_filters,
                                          as_tensor_variable((x, y)))
        return self.nonlinearity(deconved)


class Deconv2DNoBiasLayer(object):
    def __init__(self,
                 input_layer,
                 mirror_layer,
                 nonlinearity=None):
        """
        Only the valid border mode is supported.

        n_filters should be a multiple of 16
        """

        self.mirror_layer = mirror_layer

        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        n_filters = self.input_shape[0]

        if nonlinearity:
            self.nonlinearity = nonlinearity
        else:
            self.nonlinearity = mirror_layer.nonlinearity

        self.n_channels = mirror_layer.n_channels
        self.n_filters = mirror_layer.n_filters
        self.filter_size = mirror_layer.filter_size
        self.weights_std = mirror_layer.weights_std
        self.stride = mirror_layer.stride
        self.dropout = mirror_layer.dropout
        self.partial_sum = mirror_layer.partial_sum
        self.pad = mirror_layer.pad
        self.mb_size = self.input_layer.mb_size

        self.filter_shape = mirror_layer.filter_shape

        self.trainable = False
        self.W = mirror_layer.W

        self.params = []

        self.image_acts_op = ImageActs(stride=self.stride,
                                       partial_sum=self.partial_sum,
                                       pad=self.pad)

    def get_output_shape(self):
        output_shape = self.mirror_layer.input_layer.get_output_shape()
        return output_shape

    def output(self, input=None, dropout_active=True, *args, **kwargs):
        if input is None:
            input = self.input_layer.output(dropout_active=dropout_active,
                                            *args, **kwargs)

        if dropout_active and (self.dropout > 0.):
            retain_prob = 1 - self.dropout
            mask = layers.srng.binomial(input.shape, p=retain_prob,
                                        dtype='int32').astype('float32')
            # apply the input mask and rescale the input accordingly.
            # By doing this it's no longer necessary to rescale the weights
            # at test time.
            input = input / retain_prob * mask

        contiguous_input = gpu_contiguous(input)
        contiguous_filters = gpu_contiguous(self.W)
        if self.stride == 1:
            deconved = self.image_acts_op(contiguous_input, contiguous_filters)
        else:
            _, x, y, _ = self.get_output_shape()
            deconved = self.image_acts_op(contiguous_input, contiguous_filters,
                                          as_tensor_variable((x, y)))
        return self.nonlinearity(deconved)


class Pooling2DLayer(object):
    def __init__(self, input_layer, pool_size, stride=None):
        """
        pool_size is an INTEGER, not a tuple. We can only do square pooling.
        If the stride is none, it is taken to be the same as the pool size.

        borders are never ignored.
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.input_layer = input_layer
        self.trainable = False
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.pool_op = MaxPool(ds=self.pool_size, stride=self.stride)

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        w, h = input_shape[1], input_shape[2]

        new_w = int(numpy.ceil(float(w - self.pool_size + self.stride)
                               / self.stride))
        new_h = int(numpy.ceil(float(h - self.pool_size + self.stride)
                               / self.stride))

        return (input_shape[0], new_w, new_h, input_shape[3])

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        contiguous_input = gpu_contiguous(input)
        return self.pool_op(contiguous_input)


class Unpooling2DLayer(object):
    def __init__(self, input_layer, pooling_layer):
        """
        pool_size is an INTEGER, not a tuple. We can only do square pooling.
        if the stride is none, it is taken to be the same as the pool size.

        borders are never ignored.
        """
        self.pool_size = pooling_layer.pool_size
        self.stride = pooling_layer.stride
        self.input_layer = input_layer
        self.pooling_layer = pooling_layer
        self.trainable = False
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.unpool_op = MaxPoolGrad(ds=self.pool_size, stride=self.stride,
                                     start=0)

    def get_output_shape(self):
        shape = self.pooling_layer.input_layer.get_output_shape()
        return shape

    def output(self, *args, **kwargs):
        input = self.input_layer.output()
        max_out = self.pooling_layer.output()
        orig_input = self.pooling_layer.input_layer.output()
        return self.unpool_op(orig_input, max_out, input)


class ShuffleC01BToBC01Layer(object):
    """
    This layer dimshuffles 4D input for interoperability for C01B and BC01 ops.
    C01B (cuda convnet) -> BC01 (theano)
    """
    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.trainable = False
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        return (input_shape[3], input_shape[0], input_shape[1], input_shape[2])

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        return input.dimshuffle(3, 0, 1, 2)


class ShuffleBC01ToC01BLayer(object):
    """
    This layer dimshuffles 4D input for interoperability for C01B and BC01 ops.
    BC01 (theano) -> C01B (cuda convnet)
    """
    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.trainable = False
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        return (input_shape[1], input_shape[2], input_shape[3], input_shape[0])

    def output(self, *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        return input.dimshuffle(1, 2, 3, 0)
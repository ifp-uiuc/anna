from copy import copy

import numpy
import matplotlib.pyplot as pyplot

import theano
import theano.tensor as T  #TODO(tpaine) remove this dependency, can be done by factoring out the cost theano equation

from fastor.layers import layers, cc_layers
from fastor.datasets import unsupervised_dataset

theano.config.floatX = 'float32'

def orthogonalize(w):
    # Orthogonalize square matrices.
    # Or left orthogonalize overcomplete matrices.
    # Simply gets an SVD decomposition, and sets the singular values to ones.
    dim2, dim1 = w.shape
    u, s, v = numpy.linalg.svd(w)
    S = numpy.zeros((dim2,dim1))
    s = s/s
    S[:dim1,:dim1] = numpy.diag(s)
    w = numpy.dot(u,numpy.dot(S,v))
    w = numpy.float32(w)
    return w

def conv_orthogonalize(w, k=1.0):
    # Reshape filters into a matrix
    channels, width, height, filters = w.shape
    w = w.reshape(channels*width*height, filters).transpose(1,0)

    # Orthogonalize the matrix
    w = orthogonalize(w)

    # Contruct 2D hamming window
    hamming1 = numpy.hamming(width)
    hamming2 = numpy.hamming(height)
    hamming = numpy.outer(hamming1, hamming2)

    # Use it to mask the input to w
    mask = numpy.tile(hamming[None,:,:], (channels,1,1))
    mask = mask.reshape(channels*width*height)*k
    m = numpy.diag(mask)
    w = numpy.dot(w, m)

    # Reshape the matrix into filters
    w = w.transpose(1,0)
    w =  w.reshape(channels, width, height, filters)
    w = numpy.float32(w)
    return w

def get_image(batch, i):
    image = numpy.uint8(batch[:,:,:,i].transpose(1,2,0)/2.0*255.0)
    return image

class Model(object):
    input = cc_layers.CudaConvnetInput2DLayer(128, 3, 96, 96)
    conv1 = cc_layers.CudaConvnetConv2DLayer(input, 
                                             n_filters=64,
                                             filter_size=4,
                                             weights_std=0.01,
                                             init_bias_value=0.0)
    deconv2 = cc_layers.CudaConvnetDeconv2DLayer(conv1, conv1)
    
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.all_parameters_symbol = layers.all_parameters(self._get_output_layer())
    
        # can switch to gen_updates_regular_momentum
        self.learning_rate_symbol = theano.shared(numpy.array(0.01, dtype=theano.config.floatX))
        self.updates_symbol = layers.gen_updates_sgd(self._get_cost_symbol(),
                                                     self.all_parameters_symbol,
                                                     learning_rate=self.learning_rate_symbol)

        self.train_func = theano.function([self._get_input_symbol()],
                                           self._get_cost_symbol(),
                                           updates=self.updates_symbol)
        self.eval_func = theano.function([self._get_input_symbol()],
                                         self._get_cost_symbol())
        self.prediction_func = theano.function([self._get_input_symbol()],
                                          self._get_output_symbol())
    def _get_input_symbol(self):
        return self.input.output()
    
    def _get_output_symbol(self):
        return self.deconv2.output()
    
    def _get_cost_symbol(self):
        input = self._get_input_symbol()
        output = self._get_output_symbol()
        cost = T.mean((output - input) ** 2)
        return cost

    def _get_output_layer(self):
        return self.deconv2
    
    def train(self, batch):
        return self.train_func(batch)
    
    def eval(self, batch):
        return self.eval_func(batch)

    def prediction(self, batch):
        return self.prediction_func(batch)

data = numpy.load('/data/stl10_matlab/unsupervised.npy')
batch = numpy.float32(data[0:128, :, :, :])
batch /= 255.0
batch *= 2.0

batch = batch.transpose(1,2,3,0)
model = Model('deconv-stl10-96x128', '/experiments/deconv/stl10-96x128')
w = model.conv1.W.get_value()
w_old = copy(w)

k = 1.05
w = conv_orthogonalize(w_old,k=k)
# Initialize the layers with orthogonal weights
model.conv1.W.set_value(w)
model.deconv2.W.set_value(w)
batch_hat = model.prediction(batch)
print 'k: %0.2f, error: %0.4f' % (k, model.eval(batch))
model.conv1.W.set_value(w_old)
model.deconv2.W.set_value(w_old)

i = numpy.random.randint(128)
image = get_image(batch, i)
image_hat = get_image(batch_hat, i)
pyplot.imshow(numpy.hstack((image, image_hat)))
pyplot.colorbar()

pyplot.show()
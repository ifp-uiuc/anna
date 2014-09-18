import numpy
import matplotlib.pyplot as pyplot

import theano
import theano.tensor as T  #TODO(tpaine) remove this dependency, can be done by factoring out the cost theano equation

import pylearn2.datasets.mnist as mnist

from fastor.layers import layers, cc_layers
from fastor import util

theano.config.floatX = 'float32'

def orthogonalize(w):
    # Orthogonalize square matrices.
    # Or left orthogonalize overcomplete matrices.
    # Simply gets an SVD decomposition, and sets the singular values to ones.
    dim2, dim1 = w.shape
    u, s, v = numpy.linalg.svd(w)
    S = numpy.zeros((dim2,dim1))
    s = s/s
    S[:dim2,:dim2] = numpy.diag(s)
    w = numpy.dot(u,numpy.dot(S,v))
    w = numpy.float32(w)
    return w

class Model(object):
    input = layers.FlatInputLayer(128, 28*28)
    layer1 = layers.DenseLayer(input,
                               n_outputs=1000,
                               weights_std=0.01,
                               init_bias_value=0.0)                               
    layer2 = layers.DenseLayer(layer1,
                               n_outputs=1000,
                               weights_std=0.01,
                               init_bias_value=0.0)                               
    layer3 = layers.DenseLayer(layer2,
                               n_outputs=1000,
                               weights_std=0.01,
                               init_bias_value=0.0)                               
    layer4 = layers.DenseLayer(layer3,
                               n_outputs=1000,
                               weights_std=0.01,
                               init_bias_value=0.0)                               
    layer5 = layers.DenseLayer(layer4,
                               n_outputs=1000,
                               weights_std=0.01,
                               init_bias_value=0.0)                               
    layer6 = layers.DenseLayer(layer5,
                               n_outputs=1000,
                               weights_std=0.01,
                               init_bias_value=0.0)                               
    layer7 = layers.DenseLayer(layer6,
                               n_outputs=1000,
                               weights_std=0.01,
                               init_bias_value=0.0)                               
    layer8 = layers.DenseLayer(layer7,
                               n_outputs=28*28,
                               weights_std=0.01,
                               init_bias_value=0.0)                               

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.all_parameters_symbol = layers.all_parameters(self._get_output_layer())
    
        # can switch to gen_updates_regular_momentum
        self.learning_rate_symbol = theano.shared(numpy.array(0.01, dtype=theano.config.floatX))
        # self.updates_symbol = layers.gen_updates_sgd(self._get_cost_symbol(),
        #                                              self.all_parameters_symbol,
        #                                              learning_rate=self.learning_rate_symbol)
        self.updates_symbol = layers.gen_updates_regular_momentum(self._get_cost_symbol(),
                                                                  self.all_parameters_symbol,
                                                                  learning_rate=self.learning_rate_symbol,
                                                                  momentum=0.9,
                                                                  weight_decay=1e-5)

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
        return self.layer8.output()
    
    def _get_cost_symbol(self):
        input = self._get_input_symbol()
        output = self._get_output_symbol()
        cost = T.sum((output - input) ** 2)/128
        return cost

    def _get_output_layer(self):
        return self.layer8
    
    def train(self, batch):
        return self.train_func(batch)
    
    def eval(self, batch):
        return self.eval_func(batch)

    def prediction(self, batch):
        return self.prediction_func(batch)

print 'Creating model...'
model = Model('ortho-train','/experiments/results/mnist-dense-orthogonal')
monitor = util.Monitor(model)

train_dataset = mnist.MNIST('train')
train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=1000)
batch = train_iterator.next()

# Orthogonalize weights
print 'Orthogonalizing weights...'
w1 = orthogonalize(model.layer1.W.get_value())
model.layer1.W.set_value(w1)
w2 = orthogonalize(model.layer2.W.get_value())
model.layer2.W.set_value(w2)
w3 = orthogonalize(model.layer3.W.get_value())
model.layer3.W.set_value(w3)
w4 = orthogonalize(model.layer4.W.get_value())
model.layer4.W.set_value(w4)
model.layer5.W.set_value(w4.T)
model.layer6.W.set_value(w3.T)
model.layer7.W.set_value(w2.T)
model.layer8.W.set_value(w1.T)

print model.eval(batch)
model.learning_rate_symbol.set_value(0.001)

print 'Start training layer1'
train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=100000)
for batch in train_iterator:
        monitor.start()
        error = model.train(batch)
        monitor.stop(error)
        

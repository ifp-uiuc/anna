import numpy
import matplotlib.pyplot as pyplot

import theano
import theano.tensor as T  #TODO(tpaine) remove this dependency, can be done by factoring out the cost theano equation

from fastor.layers import layers, cc_layers
from fastor import util

theano.config.floatX = 'float32'

class Model(object):
    winit = 0.1
    binit = 0.0
    nonlinearity = layers.rectify
    input = layers.FlatInputLayer(64, 3*32*32)
    layer1 = layers.DenseLayer(input,
                               n_outputs=4000,
                               weights_std=winit,
                               init_bias_value=binit,
                               nonlinearity=nonlinearity)                               
    layer2 = layers.DenseLayer(layer1,
                               n_outputs=4000,
                               weights_std=winit,
                               init_bias_value=binit,
                               nonlinearity=nonlinearity)                               
    layer3 = layers.DenseLayer(layer2,
                               n_outputs=4000,
                               weights_std=winit,
                               init_bias_value=binit,
                               nonlinearity=nonlinearity)                               
    layer4 = layers.DenseLayer(layer3,
                               n_outputs=4000,
                               weights_std=winit,
                               init_bias_value=binit,
                               nonlinearity=nonlinearity)                               
    layer5 = layers.DenseLayer(layer4,
                               n_outputs=4000,
                               weights_std=winit,
                               init_bias_value=binit,
                               nonlinearity=nonlinearity)                               
    layer6 = layers.DenseLayer(layer5,
                               n_outputs=4000,
                               weights_std=winit,
                               init_bias_value=binit,
                               nonlinearity=nonlinearity)                               
    layer7 = layers.DenseLayer(layer6,
                               n_outputs=4000,
                               weights_std=winit,
                               init_bias_value=binit,
                               nonlinearity=nonlinearity)                               
    layer8 = layers.DenseLayer(layer7,
                               n_outputs=3*32*32,
                               weights_std=winit,
                               init_bias_value=binit,
                               nonlinearity=nonlinearity)                               

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.all_parameters_symbol = layers.all_parameters(self._get_output_layer())
    
        self.learning_rate_symbol = theano.shared(numpy.array(0.0001, dtype=theano.config.floatX))
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
        cost = T.sum((output - input) ** 2)/64
        return cost

    def _get_output_layer(self):
        return self.layer8
    
    def train(self, batch):
        return self.train_func(batch)
    
    def eval(self, batch):
        return self.eval_func(batch)

    def prediction(self, batch):
        return self.prediction_func(batch)
        

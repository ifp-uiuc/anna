import numpy

import theano
import theano.tensor as T  #TODO(tpaine) remove this dependency, can be done by factoring out the cost theano equation

from layers import layers, cc_layers

theano.config.floatX = 'float32'

class Model(object):
    input = cc_layers.CudaConvnetInput2DLayer(128, 3, 32, 32)
    conv1 = cc_layers.CudaConvnetConv2DLayer(input, 
                                             n_filters=32,
                                             filter_size=6,
                                             weights_std=0.01,
                                             init_bias_value=0.1)
    conv2 = cc_layers.CudaConvnetConv2DLayer(conv1, 
                                             n_filters=64,
                                             filter_size=3,
                                             weights_std=0.01,
                                             init_bias_value=0.1)
    deconv3 = cc_layers.CudaConvnetDeconv2DLayer(conv2,
                                                 n_channels=32,
                                                 filter_size=3,
                                                 weights_std=0.01,
                                                 init_bias_value=0.1)
    deconv4 = cc_layers.CudaConvnetDeconv2DLayer(deconv3,
                                                 n_channels=3,
                                                 filter_size=6,
                                                 weights_std=0.01,
                                                 init_bias_value=0.1)
    
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
        return self.deconv4.output()
    
    def _get_cost_symbol(self):
        input = self._get_input_symbol()
        output = self._get_output_symbol()
        cost = T.mean((output - input) ** 2)
        return cost

    def _get_output_layer(self):
        return self.deconv4
    
    def train(self, batch):
        return self.train_func(batch)
    
    def eval(self, batch):
        return self.eval_func(batch)

    def prediction(self, batch):
        return self.prediction_func(batch)
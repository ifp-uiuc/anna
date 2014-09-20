import numpy

import theano
import theano.tensor as T  #TODO(tpaine) remove this dependency, can be done by factoring out the cost theano equation

from fastor.layers import layers, cc_layers

theano.config.floatX = 'float32'

class Model(object):
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 32, 32)    
    k = float(numpy.random.rand()*6.5 + 0.5)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(3*3*3)
    winit2 = k/numpy.sqrt(3*3*32)
    binit = 0.0
    nonlinearity = layers.rectify
    
    conv1 = cc_layers.CudaConvnetConv2DLayer(input, 
                                             n_filters=32,
                                             filter_size=3,
                                             weights_std=winit1,
                                             init_bias_value=binit,
                                             nonlinearity=nonlinearity)
    conv2 = cc_layers.CudaConvnetConv2DLayer(conv1,
                                             n_filters=64,
                                             filter_size=3,
                                             weights_std=winit2,
                                             init_bias_value=binit,
                                             nonlinearity=nonlinearity)
    deconv3 = cc_layers.CudaConvnetDeconv2DLayer(conv2, conv2)
    output = cc_layers.CudaConvnetDeconv2DLayer(deconv3, conv1)
    
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.all_parameters_symbol = layers.all_parameters(self._get_output_layer())
    
        # can switch to gen_updates_regular_momentum
        self.learning_rate_symbol = theano.shared(numpy.array(0.000001, dtype=theano.config.floatX))
        # self.updates_symbol = layers.gen_updates_sgd(self._get_cost_symbol(),
        #                                             self.all_parameters_symbol,
        #                                             learning_rate=self.learning_rate_symbol)
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
        return self.output.output()
    
    def _get_cost_symbol(self):
        input = self._get_input_symbol()
        output = self._get_output_symbol()
        cost = T.sum((output - input) ** 2)/self.batch
        return cost

    def _get_output_layer(self):
        return self.output
    
    def train(self, batch):
        return self.train_func(batch)
    
    def eval(self, batch):
        return self.eval_func(batch)

    def prediction(self, batch):
        return self.prediction_func(batch)


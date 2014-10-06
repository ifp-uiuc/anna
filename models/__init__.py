import numpy

import theano
import theano.tensor as T  #TODO(tpaine) remove this dependency, can be done by factoring out the cost theano equation

from fastor.layers import layers, cc_layers

theano.config.floatX = 'float32'

## Abstract Model
class AbstractModel(object):

    def __init__(self, name, path, learning_rate=0.000001):
        self.name = name
        self.path = path
        self.learning_rate_symbol = theano.shared(
            numpy.array(learning_rate, dtype=theano.config.floatX))

        # Automatically compile theano functions needed for model.
        self._compile()        

    def _compile(self):
        raise NotImplementedError(str(type(self)) + 
            " does not implement _compile.")
    
    def _get_cost_symbol(self):
        raise NotImplementedError(str(type(self)) + 
            " does not implement _get_cost_symbol.")
        
    def train(self, batch):
        raise NotImplementedError(str(type(self)) + 
            " does not implement train.")
            
    def eval(self, batch):
        raise NotImplementedError(str(type(self)) + 
            " does not implement eval.")
        
    def prediction(self, batch):
        raise NotImplementedError(str(type(self)) + 
            " does not implement prediction.")

    def _get_input_symbol(self):
        # First layer of Model must be called input
        return self.input.output()
    
    def _get_output_symbol(self):
        # Last layer of Model must be called output
        return self.output.output()
    
    def _get_output_layer(self):
        # Last layer of Model must be called output
        return self.output


class UnsupervisedModel(AbstractModel):                
    #def __init__(self, name, path, learning_rate=0.000001):
    #    super(UnsupervisedModel, self).__init__(
    #        name, path, learning_rate=learning_rate)

    def _compile(self):
        self.all_parameters_symbol = layers.all_parameters(
            self._get_output_layer())
        # can switch to gen_updates_regular_momentum
        self.updates_symbol = layers.gen_updates_regular_momentum(
            self._get_cost_symbol(),
            self.all_parameters_symbol,
            learning_rate=self.learning_rate_symbol,
            momentum=0.9,
            weight_decay=1e-5)

        self.train_func = theano.function(
            [self._get_input_symbol()],
            self._get_cost_symbol(),
            updates=self.updates_symbol)

        self.eval_func = theano.function(
            [self._get_input_symbol()],
            self._get_cost_symbol())

        self.prediction_func = theano.function(
            [self._get_input_symbol()],
            self._get_output_symbol())

    def _get_cost_symbol(self):
        input = self._get_input_symbol()
        output = self._get_output_symbol()
        cost = T.sum((output - input) ** 2)/self.batch
        return cost

    def train(self, batch):
        return self.train_func(batch)
    
    def eval(self, batch):
        return self.eval_func(batch)

    def prediction(self, batch):
        return self.prediction_func(batch)
        

class SupervisedModel(AbstractModel):  
    def __init__(self, name, path, learning_rate=0.000001):
        self.y = T.lvector(name='labels')
        super(SupervisedModel, self).__init__(
            name, path, learning_rate=learning_rate)

    def _compile(self):
        self.all_parameters_symbol = layers.all_parameters(
            self._get_output_layer())

        self.updates_symbol = layers.gen_updates_regular_momentum(
            self._get_cost_symbol(),
            self.all_parameters_symbol,
            learning_rate=self.learning_rate_symbol,
            momentum=0.9,
            weight_decay=1e-5)
        
        self.train_func = theano.function(
            [self._get_input_symbol(), self._get_y_symbol()],
            [self._get_cost_symbol(), self._get_accuracy_symbol()],
            updates=self.updates_symbol)
                
        self.eval_func = theano.function(
            [self._get_input_symbol(), self._get_y_symbol()], 
            self._get_accuracy_symbol())

        self.prediction_func = theano.function(
            [self._get_input_symbol()],
            self._get_output_symbol())

    def _get_cost_symbol(self):
        y = self._get_y_symbol()
        output = self._get_output_symbol()
        cost = -T.mean(T.log(output)[T.arange(y.shape[0]), y])
        return cost

    def train(self, x_batch, y_batch):
        return self.train_func(x_batch, y_batch)
            
    def eval(self, x_batch, y_batch):
        return self.eval_func(x_batch, y_batch)

    def prediction(self, batch):
        return self.prediction_func(x_batch)

    # New methods for SupervisedModel class     
    def _get_y_symbol(self):
        return self.y
    
    def _get_accuracy_symbol(self):
        predicted_label_symbol = T.argmax(
            self._get_output_symbol(), axis=1)
        y = self._get_y_symbol()
        accuracy_symbol = 1.0*T.sum(T.eq(
            predicted_label_symbol, y))/y.shape[0]
        return accuracy_symbol
    

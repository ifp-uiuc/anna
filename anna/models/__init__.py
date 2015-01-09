import numpy

import theano
import theano.tensor as T  #TODO(tpaine) remove this dependency, can be done by factoring out the cost theano equation

from anna.layers import layers, cc_layers

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
        self.all_trainable_parameters_symbol = layers.all_trainable_parameters(
            self._get_output_layer())

        self.all_save_parameters_symbol = layers.all_parameters(
            self._get_output_layer())
        
        # can switch to gen_updates_regular_momentum
        self.updates_symbol = layers.gen_updates_regular_momentum(
            self._get_cost_symbol(),
            self.all_trainable_parameters_symbol,
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
        
class RegressionModel(AbstractModel):
    def _compile(self):
        self.all_trainable_parameters_symbol = layers.all_trainable_parameters(
            self._get_output_layer())

        self.all_save_parameters_symbol = layers.all_parameters(
            self._get_output_layer())
        
        # can switch to gen_updates_regular_momentum
        self.updates_symbol = layers.gen_updates_regular_momentum(
            self._get_cost_symbol(),
            self.all_trainable_parameters_symbol,
            learning_rate=self.learning_rate_symbol,
            momentum=0.9,
            weight_decay=1e-5)

        self.train_func = theano.function(
            [self._get_input_symbol(), self._get_y_symbol()],
            self._get_cost_symbol(),
            updates=self.updates_symbol)

        self.eval_func = theano.function(
            [self._get_input_symbol(), self._get_y_symbol()],
            self._get_cost_symbol())

        self.prediction_func = theano.function(
            [self._get_input_symbol()],
            self._get_output_symbol())

    def _get_cost_symbol(self):
        y = self._get_y_symbol()
        output = self._get_output_symbol()
        cost = (T.sum(T.mean((output - y) ** 2, axis=(1,2,3)))/3)**(0.5)
        return cost

    def train(self, batch_x, batch_y):
        return self.train_func(batch_x, batch_y)
    
    def eval(self, batch_x, batch_y):
        return self.eval_func(batch_x, batch_y)

    def prediction(self, batch_x):
        return self.prediction_func(batch_x)

    # New methods for RegressionModel
    def _get_y_symbol(self):
        return self.y.output()

class SupervisedModel(AbstractModel):  
    def __init__(self, name, path, learning_rate=0.000001):
        self.y = T.lvector(name='labels')
        super(SupervisedModel, self).__init__(
            name, path, learning_rate=learning_rate)

    def _compile(self):
        self.all_trainable_parameters_symbol = layers.all_trainable_parameters(
            self._get_output_layer())

        self.all_save_parameters_symbol = layers.all_parameters(
            self._get_output_layer())

        self.updates_symbol = layers.gen_updates_regular_momentum(
            self._get_cost_symbol(),
            self.all_trainable_parameters_symbol,
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

    def prediction(self, x_batch):
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

class RegressionModel(AbstractModel):  
    def __init__(self, name, path, learning_rate=0.000001):
        super(RegressionModel, self).__init__(
            name, path, learning_rate=learning_rate)

    def _compile(self):
        self.all_trainable_parameters_symbol = layers.all_trainable_parameters(
            self._get_output_layer())

        self.all_save_parameters_symbol = layers.all_parameters(
            self._get_output_layer())

        self.updates_symbol = layers.gen_updates_regular_momentum(
            self._get_cost_symbol(),
            self.all_trainable_parameters_symbol,
            learning_rate=self.learning_rate_symbol,
            momentum=0.9,
            weight_decay=1e-5)
        
        self.train_func = theano.function(
            [self._get_input_symbol(), self._get_y_symbol()],
            self._get_cost_symbol(),
            updates=self.updates_symbol)
                
        self.eval_func = theano.function(
            [self._get_input_symbol(), self._get_y_symbol()], 
            self._get_cost_symbol())

        self.prediction_func = theano.function(
            [self._get_input_symbol()],
            self._get_output_symbol())

    def _get_cost_symbol(self):
        y = self._get_y_symbol()
        output = self._get_output_symbol()
        cost = T.mean((y - output)**2)
        return cost

    def train(self, x_batch, y_batch):
        return self.train_func(x_batch, y_batch)
            
    def eval(self, x_batch, y_batch):
        return self.eval_func(x_batch, y_batch)

    def prediction(self, x_batch):
        return self.prediction_func(x_batch)

    # New methods for this class     
    def _get_y_symbol(self):
        return self.y.output()

class KRegressionModel(AbstractModel):  
    def __init__(self, name, path, learning_rate=0.000001):
        self.cluster_symbol = T.fmatrix('cluster')
        super(KRegressionModel, self).__init__(
            name, path, learning_rate=learning_rate)

    def _compile(self):
        self.all_trainable_parameters_symbol = layers.all_trainable_parameters(
            self._get_output_layer())

        self.all_save_parameters_symbol = layers.all_parameters(
            self._get_output_layer())

        self.updates_symbol = layers.gen_updates_regular_momentum(
            self._get_cost_symbol(),
            self.all_trainable_parameters_symbol,
            learning_rate=self.learning_rate_symbol,
            momentum=0.9,
            weight_decay=1e-5)
        
        self.train_func = theano.function(
            [self._get_input_symbol(),
            self._get_y_symbol(),
            self.cluster_symbol],
            self._get_cost_symbol(),
            updates=self.updates_symbol)
                
        self.eval_func = theano.function(
            [self._get_input_symbol(),
            self._get_y_symbol(),
            self.cluster_symbol], 
            self._get_cost_symbol())

        self.prediction_func = theano.function(
            [self._get_input_symbol()],
            self._get_output_symbol())

        self.cluster_func = theano.function(
            [self._get_input_symbol(), self._get_y_symbol()],
            self._get_cluster_symbol())

    def _get_cost_symbol(self):
        cluster = self.cluster_symbol
        mask = T.tile(cluster[:, None, :], (1, self.y_n, 1))
        y = self._get_y_symbol()
        output = self._get_output_symbol()
        Y_hat = T.reshape(output, (self.batch, self.y_n, self.k))
        y_hat = T.sum(Y_hat*mask, axis=2)
        cost = T.mean((y - y_hat)**2)
        return cost

    def _get_cluster_symbol(self):
        output = self._get_output_symbol()
        Y_hat = T.reshape(output, (self.batch, self.y_n, self.k))
        y = self._get_y_symbol()
        Y = T.tile(y[:, :, None], (1, 1, self.k))
        diff = T.mean((Y - Y_hat)**2, axis=1)
        cluster = T.argmin(diff, axis=1)
        return cluster

    def train(self, x_batch, y_batch, mask):
        return self.train_func(x_batch, y_batch, mask)
            
    def eval(self, x_batch, y_batch, mask):
        return self.eval_func(x_batch, y_batch, mask)

    def prediction(self, x_batch):
        return self.prediction_func(x_batch)

    def cluster(self, x_batch, y_batch):
        clusters = self.cluster_func(x_batch, y_batch)

        one_hot = numpy.zeros((self.batch, self.k))
        for i, cluster in enumerate(clusters):
            one_hot[i, cluster] = 1
        return one_hot

    # New methods for this class     
    def _get_y_symbol(self):
        return self.y.output()
    
class ReinforcementModel(AbstractModel):
    def __init__(self, name, path, learning_rate=0.001):
        self.r_symbol = T.fvector('r')
        self.gamma_symbol = T.fscalar('gamma')
        self.action_symbol = T.fmatrix('action')
        self.y_symbol = T.fvector('y')
        super(ReinforcementModel, self).__init__(
            name, path, learning_rate=learning_rate)

    def _compile(self):
        self.all_trainable_parameters_symbol = layers.all_trainable_parameters(
            self._get_output_layer())

        self.all_save_parameters_symbol = layers.all_parameters(
            self._get_output_layer())

        self.updates_symbol = layers.gen_updates_regular_momentum(
            self._get_cost_symbol(),
            self.all_trainable_parameters_symbol,
            learning_rate=self.learning_rate_symbol,
            momentum=0.9,
            weight_decay=1e-5)

        self.train_func = theano.function(
            [self._get_input_symbol(),
             self.action_symbol,
             self.y_symbol],
            self._get_cost_symbol(),
            updates=self.updates_symbol)

        self.eval_func = theano.function(
            [self._get_input_symbol(),
             self.action_symbol,
             self.y_symbol],
            self._get_cost_symbol()
            )

        self.prediction_func = theano.function(
            [self._get_input_symbol()],
            self._get_output_symbol()
            )

        # New funcs for ReinforcementModel

        self.action_func = theano.function(
            [self._get_input_symbol()],
            self._get_action_symbol()
            )

        self.max_q_func = theano.function(
            [self._get_input_symbol()],
            self._get_max_q_symbol()
            )

        self.y_func = theano.function(
            [self._get_input_symbol(),
             self.r_symbol,
             self.gamma_symbol],
            self._get_y_symbol()
            )

        self.value_func = theano.function(
            [self._get_input_symbol(),
             self.action_symbol],
            self._get_value_symbol()
            )

    def prediction(self, x_batch):
        '''Output of the Q network
        '''
        return self.prediction_func(x_batch)

    def _get_action_symbol(self):
        output = self._get_output_symbol()
        action = T.argmax(output, axis=1)
        return action

    def _get_max_q_symbol(self):
        output = self._get_output_symbol()
        max_q = T.max(output, axis=1)
        return max_q

    def _get_y_symbol(self):
        max_q = self._get_max_q_symbol()
        r = self.r_symbol
        gamma = self.gamma_symbol
        y = r + gamma * max_q
        return y

    def _get_value_symbol(self):
        action = self.action_symbol
        output = self._get_output_symbol()
        value = T.sum(action*output, axis=1)
        return value

    def _get_cost_symbol(self):
        value = self._get_value_symbol()
        y = self.y_symbol
        cost = T.mean((y - value)**2)
        return cost

    def action(self, batch_x):
        '''Action with max Q value
        '''
        action_index = self.action_func(batch_x)
        actions = numpy.zeros((self.batch, self.action_dims),
                              dtype=numpy.float32)
        for i, action in enumerate(action_index):
            actions[i, action] = 1
        return actions

    def max_q(self, batch_x):
        '''Max Q value
        '''
        return self.max_q_func(batch_x)

    def y(self, batch_x, r, gamma):
        '''Target value - function of input, reward, and gamma.
        '''
        return self.y_func(batch_x, r, gamma)

    def value(self, batch_x, action):
        '''Q value associated with given action, opposed to the max Q value.
        '''
        return self.value_func(batch_x, action)

    def eval(self, batch_x, action, y):
        '''Average squared difference between Q value of action and target, y.
        '''
        return self.eval_func(batch_x, action, y)

    def train(self, batch_x, action, y):
        '''Train (i.e. update the model parameters) based on data.
        '''
        return self.train_func(batch_x, action, y)

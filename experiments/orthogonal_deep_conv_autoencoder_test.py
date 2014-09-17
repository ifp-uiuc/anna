from copy import copy

import numpy
import matplotlib.pyplot as pyplot

import theano
import theano.tensor as T  #TODO(tpaine) remove this dependency, can be done by factoring out the cost theano equation

from layers import layers, cc_layers
import util
import datasets.unsupervised_dataset as unsupervised_dataset

theano.config.floatX = 'float32'

def orthogonalize(w):
    # Orthogonalize square matrices.
    # Or left orthogonalize overcomplete matrices.
    # Simply gets an SVD decomposition, and sets the singular values to ones.
    dim2, dim1 = w.shape
    short = numpy.min((dim2, dim1))
    u, s, v = numpy.linalg.svd(w)
    S = numpy.zeros((dim2,dim1))
    s = s/s
    S[:short,:short] = numpy.diag(s)
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
    conv1 = cc_layers.ACudaConvnetConv2DLayer(input, 
                                             n_filters=32,
                                             filter_size=3,
                                             weights_std=0.01,
                                             init_bias_value=0.0)
    conv2 = cc_layers.ACudaConvnetConv2DLayer(conv1,
                                             n_filters=64,
                                             filter_size=3,
                                             weights_std=0.01,
                                             init_bias_value=0.0)
    deconv3 = cc_layers.ACudaConvnetDeconv2DLayer(conv2, conv2)
    output = cc_layers.ACudaConvnetDeconv2DLayer(deconv3, conv1)
    
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.all_parameters_symbol = layers.all_parameters(self._get_output_layer())
    
        # can switch to gen_updates_regular_momentum
        self.learning_rate_symbol = theano.shared(numpy.array(0.01, dtype=theano.config.floatX))
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
        cost = T.mean((output - input) ** 2)
        return cost

    def _get_output_layer(self):
        return self.output
    
    def train(self, batch):
        return self.train_func(batch)
    
    def eval(self, batch):
        return self.eval_func(batch)

    def prediction(self, batch):
        return self.prediction_func(batch)

data = numpy.load('/data/stl10_matlab/unsupervised.npy')
data = numpy.float32(data)
data = data/255.0*2.0
train_data = data[0:90000, :, :, :]
test_data = data[90000::, :, :, :]

train_dataset = unsupervised_dataset.UnsupervisedDataset(train_data)
test_dataset = unsupervised_dataset.UnsupervisedDataset(test_data)
test_iterator = test_dataset.iterator(mode='sequential', batch_size=128)
train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=100000)

model = Model('deconv-stl10-32x64', '/experiments/deconv/stl10-32x64-ortho')
#util.load_checkpoint(model, checkpoint)
monitor = util.Monitor(model)
# w1 = model.conv1.W.get_value()
# w2 = model.conv2.W.get_value()

k = 1.05

# w1 = conv_orthogonalize(w1,k=k)
# w2 = conv_orthogonalize(w2,k=k)
# # Initialize the layers with orthogonal weights
# model.conv1.W.set_value(w1)
# model.conv2.W.set_value(w2)
# model.deconv3.W.set_value(w2)
# model.output.W.set_value(w1)
batch = test_iterator.next()
batch = batch.transpose(1,2,3,0)
batch_hat = model.prediction(batch)
print 'k: %0.2f, error: %0.4f' % (k, model.eval(batch))

alpha = numpy.float32(1.0)
model.conv1.alpha.set_value(alpha)
model.conv2.alpha.set_value(alpha)
model.deconv3.alpha.set_value(alpha)
model.output.alpha.set_value(alpha)

model.learning_rate_symbol.set_value(0.01)
count = 1
for batch in train_iterator:
    # if count % 100 == 0:
    #     alpha += numpy.float32(0.1)
    #     model.conv1.alpha.set_value(alpha)
    #     model.conv2.alpha.set_value(alpha)
    #     model.deconv3.alpha.set_value(alpha)
    #     model.output.alpha.set_value(alpha)
    #     print 'alpha set to %0.3f' % alpha
    batch = batch.transpose(1,2,3,0)
    monitor.start()
    error = model.train(batch)
    monitor.stop(error)
    count += 1
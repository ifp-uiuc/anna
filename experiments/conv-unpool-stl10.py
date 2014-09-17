import numpy
import matplotlib.pyplot as pyplot

import theano

from pylearn2.space import Conv2DSpace

import unsupervised_experiments.datasets.unsupervised_dataset as unsupervised_dataset

from unsupervised_experiments.layers import layers, cc_layers
import unsupervised_experiments.util

theano.config.floatX = 'float32'

def get_image(batch, i):
    image = numpy.uint8(batch[:,:,:,i].transpose(1,2,0)/2.0*255.0)
    return image

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

def conv_orthogonalize(w, k=1.05):
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

import numpy

import theano
import theano.tensor as T  #TODO(tpaine) remove this dependency, can be done by factoring out the cost theano equation

from layers import layers, cc_layers

theano.config.floatX = 'float32'

class Model(object):
    input = cc_layers.CudaConvnetInput2DLayer(128, 3, 96, 96)
    conv1 = cc_layers.CudaConvnetConv2DLayer(input, 
                                             n_filters=80,
                                             filter_size=5,
                                             weights_std=0.01,
                                             init_bias_value=0.0)
    pool2 = cc_layers.CudaConvnetPooling2DLayer(conv1, 2, stride=2)
    unpool3 = cc_layers.CudaConvnetUnpooling2DLayer(pool2, pool2)
    output = cc_layers.CudaConvnetDeconv2DLayer(unpool3, conv1)
    
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.all_parameters_symbol = layers.all_parameters(self._get_output_layer())
    
        self.learning_rate_symbol = theano.shared(numpy.array(0.01, dtype=theano.config.floatX))
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

model = Model('conv-unpool-stl10-lr001','/experiments/conv-unpool-stl10')
monitor = util.Monitor(model)

data = numpy.load('/data/stl10_matlab/unsupervised.npy')
data = numpy.float32(data)
data = data/255.0*2.0
train_data = data[0:90000, :, :, :]
test_data = data[90000::, :, :, :]

train_dataset = unsupervised_dataset.UnsupervisedDataset(train_data)
test_dataset = unsupervised_dataset.UnsupervisedDataset(test_data)
train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=100000)
test_iterator = train_dataset.iterator(mode='sequential', batch_size=128)

# evaluator = util.Evaluator(model, test_dataset, steps=1000)
w1 = model.conv1.W.get_value()

w1 = conv_orthogonalize(w1)
# Initialize the layers with orthogonal weights
model.conv1.W.set_value(w1)

batch = test_iterator.next()
batch = batch.transpose(1,2,3,0)
batch_hat = model.prediction(batch)
print 'error: %0.4f' % (model.eval(batch))

model.learning_rate_symbol.set_value(numpy.float32(0.001))

for batch in train_iterator:
    monitor.start()
    batch = batch.transpose(1,2,3,0)
    error = model.train(batch)
    monitor.stop(error)
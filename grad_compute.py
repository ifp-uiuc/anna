import numpy

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous

from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.filter_acts import ImageActs

from layers import layers

# Class to perform convolution and deconvolution operations
filter_acts_op = FilterActs()
image_acts_op = ImageActs()

# Theano objects to construct symbolic function
input = T.ftensor4(name='input')
#filters1 = T.ftensor4(name='filters1')
#filters2 = T.ftensor4(name='filters2')
filters1 = layers.shared_single(dim=4)
filters2 = layers.shared_single(dim=4)
output = T.ftensor4(name='output')

# Convert to cudaconvnet-friendly format
gpu_input = gpu_contiguous(input)
gpu_filters1 = gpu_contiguous(filters1)
gpu_filters2 = gpu_contiguous(filters2)
gpu_output = gpu_contiguous(output)

# Specify symoblic function for each layer (convolution)
conv_layer1 = filter_acts_op(gpu_input, gpu_filters1)
conv_layer2 = filter_acts_op(conv_layer1, gpu_filters2)

# Specify symbolic function for each layer (deconvolution)
deconv_layer3 = image_acts_op(conv_layer2, gpu_filters2)
deconv_layer4 = image_acts_op(deconv_layer3, gpu_filters1)

# cost function to evaluate reconstruction
recon = deconv_layer4
cost = T.mean((recon - input) ** 2)

# input channels, rows, cols, batch_size
x = numpy.ones((3, 100, 100, 128),dtype=numpy.float32)
# input channels, filter rows, filter cols, output channels
#w1 = numpy.ones((3, 5, 5, 16),dtype=numpy.float32)
#w2 = numpy.ones((16, 5, 5, 16),dtype=numpy.float32)
filters_shape1 = (3, 5, 5, 16)
filters_shape2 = (16, 5, 5, 16)
weights_std1 = 0.001
weights_std2 = 0.001
w1 = np.random.randn(*filters_shape1).astype(np.float32) * weights_std1
w2 = np.random.randn(*filters_shape2).astype(np.float32) * weights_std2
filters1.set_value(w1)
filters2.set_value(w2)

all_params = [filters1, filters2]
all_grads = [theano.grad(cost, param) for param in all_params]
learning_rate = T.fscalar(name='learning_rate')
updates=[]
for param_i, grad_i in zip(all_params, all_grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

f = theano.function([input, learning_rate], cost, updates=updates)

for i in range(2000):
    a1 = f(x, numpy.float32(0.1))
    print a1
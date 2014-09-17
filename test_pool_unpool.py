"""Script to test unpooling op.

Please clone the repo, and add the path to the directory it is in to your PYTHONPATH environment variable.
Also you will need a copy of the stl10 dataset which is numpy compatible.

Please scp from:
kronos:/data/stl10_matlab/

You will need to modify the line below which says:
data = numpy.load('/data/stl10_matlab/unsupervised.npy')
"""

import numpy
import matplotlib.pyplot as pyplot

import theano.tensor as T
import theano
from theano.sandbox.cuda.basic_ops import gpu_contiguous
import pylearn2.sandbox.cuda_convnet.pool as pool
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.img_acts import ImageActs

from fastor.datasets import unsupervised_dataset

from fastor.layers import cc_layers

def get_image(batch, i):
    image = numpy.uint8(batch[:,:,:,i].transpose(1,2,0)/2.0*255.0)
    return image

def find_switches(max_val, image_patch, recon_patch, count_patch):
    #max_val = numpy.max(image_patch)
    switches = image_patch == max_val
    recon_patch += image_patch*switches
    count_patch += switches
    
def sexy_unpool(image, image_pool, ds=3, stride=1):
    image_size = image.shape[0]
    recon = numpy.zeros(image.shape)
    count = numpy.zeros(image.shape)
    j_x = -1
    for i_x in range(0, image_size-ds+1, stride):
        j_x += 1
        j_y = -1
        for i_y in range(0, image_size-ds+1, stride):
            j_y += 1
            x_slice = slice(i_x, i_x+ds)
            y_slice = slice(i_y, i_y+ds)
            image_patch = image[x_slice, y_slice]
            recon_patch = recon[x_slice, y_slice]
            count_patch = count[x_slice, y_slice]
            max_val = image_pool[j_x, j_y]
            find_switches(max_val, image_patch, recon_patch, count_patch)
    count[count==0]=1
    return recon/count

# Get data and format it properly
data = numpy.load('/data/stl10_matlab/unsupervised.npy')
test_data = numpy.float32(data[0:128, :, :, :]) # grap enough data for a minibatch
test_data /= 255.0
test_data *= 2.0

input = test_data.transpose(1,2,3,0) # Output (channels, width, height, batch_size)
input = numpy.tile(input[0,:,:,:][None, :, :, :], (16, 1, 1, 1)) # (channels must be a multiple of 16)

# Theano symbolic functions to do pooling and unpooling
input_symbol = T.ftensor4(name='input')

# Make ops
pool_stride = 1
ds = 5
pool_op = pool.MaxPool(ds, pool_stride)
unpool_op = pool.MaxPoolGrad(ds, pool_stride,0)

# Make symbolic expressions
pool_symbol = pool_op(input_symbol)
unpool_symbol = unpool_op(input_symbol, pool_symbol, pool_symbol)

pool_func = theano.function([input_symbol], pool_symbol)
unpool_func = theano.function([input_symbol], unpool_symbol)

# Run pooling and unpooling
pool_out = numpy.array(pool_func(input))
unpool_out = numpy.array(unpool_func(input))

# Display results on a random image
i = numpy.random.randint(128)
print i
image_input = input[0,:,:,i]
image_pool = pool_out[0,:,:,i]
image_unpool = unpool_out[0,:,:,i]
image_sexy_unpool = sexy_unpool(image_input, image_pool, ds=ds, stride=pool_stride)

pyplot.subplot(2,2,1)
pyplot.imshow(image_input)
pyplot.title('original image')
pyplot.colorbar()

pyplot.subplot(2,2,2)
pyplot.imshow(image_pool)
pyplot.title('max pool image')
pyplot.colorbar()

pyplot.subplot(2,2,3)
pyplot.imshow(image_unpool)
pyplot.title('Alex unpool image')
pyplot.colorbar()

pyplot.subplot(2,2,4)
pyplot.imshow(image_sexy_unpool)
pyplot.title('Correct unpool image')
pyplot.colorbar()

pyplot.show()
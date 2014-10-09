import sys

import numpy
from fastor import util

import pylearn2.datasets.cifar10 as cifar10
from model import Model

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


print('Start')
model = Model(sys.argv[1], sys.argv[2])
monitor = util.Monitor(model)

print('Loading Data')
train_iterator = util.get_cifar_iterator('train', 
                                    mode='random_uniform', 
                                    batch_size=128, 
                                    num_batches=100000,                                     
                                    rescale=True)

print('Orthogonalizing Weights')
w1 = model.conv1.W.get_value()
w2 = model.conv2.W.get_value()
w1 = conv_orthogonalize(w1)
w2 = conv_orthogonalize(w2)

# Initialize the layers with orthogonal weights
model.conv1.W.set_value(w1)
model.conv2.W.set_value(w2)

print('Trainig Model')
for batch in train_iterator:    
    monitor.start()
    error = model.train(batch/2)
    monitor.stop(error)    
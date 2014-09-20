import sys
import numpy

import pylearn2.datasets.mnist as mnist

from fastor import util
from fastor.datasets import unsupervised_dataset

from model import Model

def orthogonalize(w):
    # Orthogonalize square matrices.
    # Or left orthogonalize overcomplete matrices.
    # Simply gets an SVD decomposition, and sets the singular values to ones.
    dim2, dim1 = w.shape
    u, s, v = numpy.linalg.svd(w)
    S = numpy.zeros((dim2,dim1))
    s = s/s
    S[:dim2,:dim2] = numpy.diag(s)
    w = numpy.dot(u,numpy.dot(S,v))
    w = numpy.float32(w)
    return w

print 'Creating model...'
model = Model(sys.argv[1], sys.argv[2])
monitor = util.Monitor(model)

print 'Loading dataset...'
train_dataset = mnist.MNIST('train')
train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=1000000)

# Orthogonalize weights
print 'Orthogonalizing weights...'
w1 = orthogonalize(model.layer1.W.get_value())
model.layer1.W.set_value(w1)
w2 = orthogonalize(model.layer2.W.get_value())
model.layer2.W.set_value(w2)
w3 = orthogonalize(model.layer3.W.get_value())
model.layer3.W.set_value(w3)
w4 = orthogonalize(model.layer4.W.get_value())
model.layer4.W.set_value(w4)
model.layer5.W.set_value(w4.T)
model.layer6.W.set_value(w3.T)
model.layer7.W.set_value(w2.T)
model.layer8.W.set_value(w1.T)

print 'Start training...'
for batch in train_iterator:
        monitor.start()
        error = model.train(batch*2-1)
        monitor.stop(error)
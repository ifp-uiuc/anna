import sys
import numpy

import pylearn2.datasets.cifar10 as cifar10

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
train_dataset = cifar10.CIFAR10('train', rescale=True)
train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=64, num_batches=1000000)

# Orthogonalize weights
print 'Orthogonalizing weights...'
w1 = orthogonalize(model.layer1.W.get_value())
print 'Layer 1 get.'
model.layer1.W.set_value(w1)
print 'Layer 1 set.'
w2 = orthogonalize(model.layer2.W.get_value())
print 'Layer 2 get.'
model.layer2.W.set_value(w2)
print 'Layer 2 set.'
w3 = orthogonalize(model.layer3.W.get_value())
print 'Layer 3 get.'
model.layer3.W.set_value(w3)
print 'Layer 3 set.'
w4 = orthogonalize(model.layer4.W.get_value())
print 'Layer 4 get.'
model.layer4.W.set_value(w4)
print 'Layer 4 set.'
model.layer5.W.set_value(w4.T)
print 'Layer 5 set.'
model.layer6.W.set_value(w3.T)
print 'Layer 6 set.'
model.layer7.W.set_value(w2.T)
print 'Layer 7 set.'
model.layer8.W.set_value(w1.T)
print 'Layer 8 set.'

print 'Start training...'
for batch in train_iterator:
        monitor.start()
        error = model.train(batch-1)
        monitor.stop(error)
import os
import sys
sys.path.append('../../..')

import numpy

from fastor import util
from fastor.datasets import supervised_dataset

from model import SupervisedModel

def orthogonalize(w):
    # Orthogonalize square matrices.
    # Or left orthogonalize overcomplete matrices.
    # Simply gets an SVD decomposition, and sets the singular values to ones.
    dim2, dim1 = w.shape
    dim = numpy.min((dim1, dim2))
    u, s, v = numpy.linalg.svd(w)
    S = numpy.zeros((dim2,dim1))
    s = s/s
    S[:dim,:dim] = numpy.diag(s)
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

print('Start')

pid = os.getpid()
print('PID: {}'.format(pid))
f = open('pid', 'wb')
f.write(str(pid)+'\n')
f.close()

model = SupervisedModel('experiment', './')
monitor = util.Monitor(model)

# Loading CIFAR-10 dataset
print('Loading Data')
train_iterator = util.get_cifar_iterator_reduced('train',
                                    mode='random_uniform',
                                    batch_size=128,
                                    num_batches=100000,
                                    rescale=True,
                                    num_samples_per_class=1000, 
                                    which_split=0)

test_iterator = util.get_cifar_iterator('test',
                                    mode='random_uniform',
                                    batch_size=128,
                                    num_batches=100000,
                                    rescale=True)

normer = util.Normer2(filter_size=5, num_channels=3)
augmenter = util.DataAugmenter(2, (32, 32))

# Grab batch for patch extraction.
x_batch, y_batch = train_iterator.next()
# Grab some patches to initialize weights.
patch_grabber = util.PatchGrabber(96, 5)
patches = patch_grabber.run(x_batch)*0.05
model.conv1.W.set_value(patches)
# Orthogonalize second layer weights.
W2 = model.conv2.W.get_value()
W2 = conv_orthogonalize(W2)
# Scale second layer weights.
s=5.0
model.conv2.W.set_value(W2*s)
# Orthogonalize third layer weights.
W3 = model.conv3.W.get_value()
W3 = conv_orthogonalize(W3)
# Scale third layer weights.
s=5.0
model.conv3.W.set_value(W3*s)

print('Training Model')
for x_batch, y_batch in train_iterator:   
    x_batch = augmenter.run(x_batch)
    x_batch = normer.run(x_batch)
    y_batch = numpy.int64(numpy.argmax(y_batch, axis=1))
    monitor.start()
    log_prob, accuracy = model.train(x_batch, y_batch)
    monitor.stop(1-accuracy) # monitor takes error instead of accuracy    
    
    if monitor.test:
        monitor.start()
        x_test_batch, y_test_batch = test_iterator.next()
        x_test_batch = normer.run(x_test_batch)
        y_test_batch = numpy.int64(numpy.argmax(y_test_batch, axis=1))
        test_accuracy = model.eval(x_test_batch, y_test_batch)
        monitor.stop_test(1-test_accuracy)

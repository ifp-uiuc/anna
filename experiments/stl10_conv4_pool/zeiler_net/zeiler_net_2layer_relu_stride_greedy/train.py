import sys
import cPickle

import numpy
from fastor import util

from model import Model
from fastor.datasets import unsupervised_dataset

import theano
import theano.tensor as T

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
model = Model(sys.argv[1], sys.argv[2])
#checkpoint = '/experiments/cifar_conv4_pool_relu_right/checkpoints/cifar_conv4_pool_relu_right-09m-23d-00h-37m-40s.pkl'
#util.load_checkpoint(model, checkpoint)
layer1_weight_checkpoint = '/experiments/zeiler_net_1layer_relu_stride2/checkpoints/zeiler_net_1layer_relu_stride2-09m-30d-14h-33m-04s.pkl'
f = open(layer1_weight_checkpoint, 'rb')
W1_init = cPickle.load(f)[0]
f.close()
monitor = util.Monitor(model, save_steps=1000)

print('Loading Data')
data = numpy.load('/data/stl10_matlab/unsupervised.npy')
data = numpy.float32(data)
data /= 255.0
data *= 2.0
train_data = data[0:90000, :, :, :]
test_data = data[90000::, :, :, :]

train_dataset = unsupervised_dataset.UnsupervisedDataset(train_data)
test_dataset = unsupervised_dataset.UnsupervisedDataset(test_data)
train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=100000)
test_iterator = test_dataset.iterator(mode='sequential', batch_size=128)

test_x_batch = test_iterator.next()
test_x_batch = test_x_batch.transpose(1, 2, 3, 0)
#recon_visualizer = util.ReconVisualizer(model, test_x_batch, steps=50)
filter_visualizer = util.FilterVisualizer(model, steps=50)
normer = util.Normer(filter_size=7)
filter2_visualizer = util.WeightVisualizer(model, model.conv2, 'filters2', steps=50)

sparsity = theano.function([model.input.output()], T.mean(T.eq(model.conv2.output(),0), axis=(1,2,3)))
# Grab some patches to initialize weights
# patch_grabber = util.PatchGrabber(96, 7)
# x_batch = train_iterator.next()
# x_batch = x_batch.transpose(1, 2, 3, 0)
# x_batch = normer.run(x_batch)
# patches = patch_grabber.run(x_batch)*0.05
# W = model.conv1.W.get_value()
# print W.shape
# print patches.shape
# model.conv1.W.set_value(patches)

model.conv1.W.set_value(W1_init)

W2 = model.conv2.W.get_value()
W2 = conv_orthogonalize(W2)

s=5.0
model.conv2.W.set_value(W2*s)

# Grab test data to give to NormReconVisualizer
test_x_batch = test_iterator.next()
test_x_batch = test_x_batch.transpose(1, 2, 3, 0)
test_x_batch = normer.run(test_x_batch)
recon_visualizer = util.NormReconVisualizer(model, test_x_batch, steps=50)
recon_visualizer.run()   
filter_visualizer.run()
filter2_visualizer.run()

#model.learning_rate_symbol.set_value(0.000005/10)
count = 0
print('Training Model')
for x_batch in train_iterator:
    x_batch = x_batch.transpose(1, 2, 3, 0)    
    monitor.start()
    x_batch = normer.run(x_batch)
    error = model.train(x_batch)
    monitor.stop(error) 
    recon_visualizer.run()   
    filter_visualizer.run()
    filter2_visualizer.run()
    if (count != 0) and (count % 1000) == 0:
        lr = model.learning_rate_symbol.get_value()
        model.learning_rate_symbol.set_value(numpy.float32(lr*0.75))
        print('Learning Rate Now Equals: {}'.format(lr*0.75))
    if count % 200 == 0:
        print 'sparsity:'
        print numpy.sort(sparsity(x_batch))
        print 'norms:'
        W = numpy.array(model.conv2.W.get_value())
        print numpy.sort(numpy.sum(W**2,axis=(0,1,2)))
    count+=1
import os
import sys
sys.path.append('../..')

import numpy
from sklearn.cluster import MiniBatchKMeans
import theano

from fastor import util
from fastor.datasets import unsupervised_dataset

import checkpoints
from model import UnsupervisedModel


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

def get_clusters(
        model,
        model_layer,
        iterator,
        patch_size=3,
        num_clusters=96,
        num_samples=10000):

    output = theano.function([model.input.output()], model_layer.output())
    samples_per_batch = int(numpy.ceil(num_samples/iterator.num_batches))

    patch_grabber = util.PatchGrabber(
        samples_per_batch, patch_size, num_channels=96)

    print 'Grabbing patches'
    patch_list = []
    for i, x_batch in enumerate(iterator):
        print i
        x_batch = x_batch.transpose(1, 2, 3, 0)
        x_batch = normer.run(x_batch)
        patches = patch_grabber.run(numpy.array(output(x_batch)))
        patch_list.append(patches)

    patches = numpy.concatenate(patch_list, axis=3)

    # Reshape
    flat_patches = numpy.reshape(patches, (patch_grabber.num_channels*patch_grabber.patch_size**2, patches.shape[-1]))

    flat_patches = flat_patches.transpose(1, 0)

    print 'Running k-means...'
    mbk = MiniBatchKMeans(
        init='k-means++',
        n_clusters=num_clusters,
        batch_size=128,
        n_init=10,
        max_no_improvement=10,
        verbose=0)
    mbk.fit(flat_patches)

    flat_clusters = mbk.cluster_centers_.transpose(1, 0)
    clusters = numpy.reshape(
        flat_clusters,
        (patches.shape[0], patches.shape[1], patches.shape[2], num_clusters))

    return clusters

print('Start')

pid = os.getpid()
print('PID: {}'.format(pid))
f = open('pid', 'wb')
f.write(str(pid)+'\n')
f.close()

model = UnsupervisedModel('experiment', './')
checkpoint = checkpoints.unsupervised_greedy
util.set_parameters_from_unsupervised_model(model, checkpoint)
monitor = util.Monitor(model, save_steps=200)

model.conv1.trainable = False
model._compile()

# Loading STL-10 dataset
print('Loading Data')
data = numpy.load('/data/stl10_matlab/unsupervised.npy')
data = numpy.float32(data)
data /= 255.0
data *= 2.0
train_data = data[0:90000, :, :, :]
test_data = data[90000::, :, :, :]

train_dataset = unsupervised_dataset.UnsupervisedDataset(train_data)
test_dataset = unsupervised_dataset.UnsupervisedDataset(test_data)
train_iterator = train_dataset.iterator(
    mode='random_uniform', batch_size=128, num_batches=100000)
test_iterator = test_dataset.iterator(mode='sequential', batch_size=128)

# Create object to local contrast normalize a batch.
# Note: Every batch must be normalized before use.
normer = util.Normer(filter_size=5)

patch_iterator = train_dataset.iterator(
    mode='even_sequential', batch_size=128)
clusters = get_clusters(
    model,
    model.pool1,
    test_iterator)
# Orthogonalize second layer weights.
#W2 = model.conv2.W.get_value()
#W2 = conv_orthogonalize(W2)
# Scale second layer weights.
s=5.0
model.conv2.W.set_value(clusters*s)

# Grab test data to give to NormReconVisualizer.
test_x_batch = test_iterator.next()
test_x_batch = test_x_batch.transpose(1, 2, 3, 0)
test_x_batch = normer.run(test_x_batch)
recon_visualizer = util.NormReconVisualizer(model, test_x_batch, steps=200)
recon_visualizer.run()   

# Create object to display first layer filter weights.
filter_visualizer = util.FilterVisualizer(model, steps=200)
filter_visualizer.run()

#model.learning_rate_symbol.set_value(0.000005/10)
print('Training Model')
for x_batch in train_iterator:
    x_batch = x_batch.transpose(1, 2, 3, 0)    
    monitor.start()
    x_batch = normer.run(x_batch)
    error = model.train(x_batch)
    monitor.stop(error) 
    recon_visualizer.run()   
    filter_visualizer.run()

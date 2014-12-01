import os
import sys
sys.path.append('../..')

import numpy
import scipy.misc

from fastor import util
from fastor.datasets import unsupervised_dataset

import checkpoints
from model import RegressionModel2


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

def smooth(im, sigma=2.0):
    im_filt_r = scipy.ndimage.filters.gaussian_filter(im[:, :, 0], sigma)
    im_filt_g = scipy.ndimage.filters.gaussian_filter(im[:, :, 1], sigma)
    im_filt_b = scipy.ndimage.filters.gaussian_filter(im[:, :, 2], sigma)
    im_filt = numpy.dstack((im_filt_r, im_filt_g, im_filt_b))
    return im_filt

def degrade(image, factor=2, interp='bicubic', sigma=2.0):
    """Downsample, and then upsample an image.

    Parameters
    ----------
    image : ndarray
        The array of image to be resized.

    factor : int
        factor to scale by.

    interp : str
        Interpolation to use for re-sizing ('nearest', 'bilinear', 'bicubic'
        or 'cubic').

    Returns
    -------
    image_up : ndarray
        The downgraded array of image.
    """
    old_shape = image.shape
    x, y, c = old_shape
    new_shape = (x/factor, y/factor, c)
    image_filt = smooth(image, sigma=sigma)
    image_down = scipy.misc.imresize(image_filt, new_shape, interp=interp)
    image_up = scipy.misc.imresize(image_down, old_shape, interp=interp)
    return image_up


print('Start')

pid = os.getpid()
print('PID: {}'.format(pid))
f = open('pid', 'wb')
f.write(str(pid)+'\n')
f.close()

model = RegressionModel2('experiment', './', learning_rate=0.0001)
monitor = util.Monitor(model, save_steps=200)


# Loading STL-10 dataset
print('Loading Data')
data = numpy.load('/data/stl10_matlab/unsupervised.npy')
data = numpy.float32(data)
train_data = data[0:90000, :, :, :]
test_data = data[90000::, :, :, :]

train_dataset = unsupervised_dataset.UnsupervisedDataset(train_data)
test_dataset = unsupervised_dataset.UnsupervisedDataset(test_data)
train_iterator = train_dataset.iterator(
    mode='random_uniform', batch_size=128, num_batches=100000)
test_iterator = test_dataset.iterator(mode='sequential', batch_size=128)

# Create object to local contrast normalize a batch.
# Note: Every batch must be normalized before use.
#normer = util.Normer(filter_size=7)

# Grab batch for patch extraction.
x_batch = train_iterator.next()
x_batch = x_batch.transpose(1, 2, 3, 0)
#x_batch = normer.run(x_batch)

# Grab some patches to initialize weights.
#W1 = model.conv1.W.get_value()
#W1 = conv_orthogonalize(W1)
#model.conv1.W.set_value(W1)

#model.output.W.set_value(W1)

# Grab test data to give to NormReconVisualizer.
test_x_batch = test_iterator.next()
test_x_batch = test_x_batch.transpose(1, 2, 3, 0)
#test_x_batch = normer.run(test_x_batch)
#recon_visualizer = util.ReconVisualizer(model, test_x_batch, steps=200)
#recon_visualizer.run()   

# Create object to display first layer filter weights.
filter_visualizer = util.FilterVisualizer(model, steps=200)
filter_visualizer.run()

#model.learning_rate_symbol.set_value(0.000005/10)
print('Training Model')
shape = x_batch.shape
c, x, y, b = shape
y_batch = numpy.zeros((16, x, y, b), dtype=numpy.float32)

for x_batch in train_iterator:
    x_batch = x_batch.transpose(1, 2, 3, 0)
    monitor.start()
    # x_batch = normer.run(x_batch)
    for i in range(128):
        im = numpy.uint8(x_batch[:, :, :, i].transpose(1,2,0))
        im_degrade = degrade(im)
        #rmse = numpy.sqrt(numpy.mean((numpy.float32(im)-numpy.float32(im_degrade)**2))
        y_batch[0:3, :, :, i] = numpy.float32(im_degrade).transpose(2,0,1)
    error = model.train(x_batch, y_batch)
    monitor.stop(error) 
    # recon_visualizer.run()   
    filter_visualizer.run()
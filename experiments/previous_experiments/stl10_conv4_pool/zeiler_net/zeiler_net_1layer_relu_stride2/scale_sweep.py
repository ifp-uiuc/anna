import sys

import numpy
from fastor import util

from model import Model
from fastor.datasets import unsupervised_dataset

model = Model('xxx', './')

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
recon_visualizer = util.ReconVisualizer(model, test_x_batch, steps=200)
filter_visualizer = util.FilterVisualizer(model, steps=200)
normer = util.Normer(filter_size=7)

# Grab some patches to initialize weights
patch_grabber = util.PatchGrabber(48, 7)
x_batch = train_iterator.next()
x_batch = x_batch.transpose(1, 2, 3, 0)
x_batch = normer.run(x_batch)
patches = patch_grabber.run(x_batch)
W = model.conv1.W.get_value()
print W.shape
print patches.shape
model.conv1.W.set_value(patches)

# Grab test data to give to NormReconVisualizer
test_x_batch = test_iterator.next()
test_x_batch = test_x_batch.transpose(1, 2, 3, 0)
test_x_batch = normer.run(test_x_batch)
recon_visualizer = util.NormReconVisualizer(model, test_x_batch, steps=200)

for s in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
	print s
	patches2 = patches * s
	model.conv1.W.set_value(patches2)
	#prediction = numpy.array(model.prediction(test_x_batch))
	#print prediction.min()
	#print prediction.max()
	#recon_visualizer.count=0
	#recon_visualizer.run()
	print model.eval(test_x_batch)
	#g=raw_input('...')
import sys

import numpy
from fastor import util

from model import Model
from fastor.datasets import unsupervised_dataset

print('Start')
model = Model(sys.argv[1], sys.argv[2])
#checkpoint = '/experiments/cifar_conv4_pool_relu_right/checkpoints/cifar_conv4_pool_relu_right-09m-23d-00h-37m-40s.pkl'
#util.load_checkpoint(model, checkpoint)
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
recon_visualizer = util.ReconVisualizer(model, test_x_batch, steps=50)
filter_visualizer = util.FilterVisualizer(model, steps=50)
normer = util.Normer(filter_size=7)

# Grab some patches to initialize weights
patch_grabber = util.PatchGrabber(96, 7)
x_batch = train_iterator.next()
x_batch = x_batch.transpose(1, 2, 3, 0)
x_batch = normer.run(x_batch)
patches = patch_grabber.run(x_batch)*0.05
W = model.conv1.W.get_value()
print W.shape
print patches.shape
model.conv1.W.set_value(patches)

# Grab test data to give to NormReconVisualizer
test_x_batch = test_iterator.next()
test_x_batch = test_x_batch.transpose(1, 2, 3, 0)
test_x_batch = normer.run(test_x_batch)
recon_visualizer = util.NormReconVisualizer(model, test_x_batch, steps=50)

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
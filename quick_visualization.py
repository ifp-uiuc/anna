import os
import Image

import numpy

import pylearn2.datasets.cifar10 as cifar10
from pylearn2.space import Conv2DSpace
import model as Model
import util

def rescale(data):
	data = data/2.0*255.0
	data[data > 255.0] = 255.0
	return data

model = Model.Model('deconv-cifar10', '/experiments/deconv/cifar10/')
util.load_checkpoint(model, '/experiments/cifar10/cifar10-09m-09d-01h-25m-08s.pkl')

test_dataset = cifar10.CIFAR10(which_set='test',
                               rescale=True,
                               axes=['c', 0, 1, 'b']
                               )
input_convspace = Conv2DSpace(shape=(32, 32), num_channels=3,
                              axes=('c', 0, 1, 'b'))
data_specs = (input_convspace,'features')
test_iterator = test_dataset.iterator(mode='sequential', batch_size=128, data_specs=data_specs)

x = test_iterator.next()

#print model.eval(x)
x_hat = model.prediction(x)
x_hat = x_hat.transpose(3,1,2,0)
x = x.transpose(3,1,2,0)

for i in range(128):
	image_array = numpy.uint8(rescale(numpy.hstack((x[i,:,:,:],x_hat[i,:,:,:]))))
	image = Image.fromarray(image_array)
	filename = 'recon-%02d.jpeg' % i
	filepath = os.path.join('/experiments/results/cifar10-fixed/', filename)
	image.save(filepath)
import os
import Image

import numpy

from fastor.datasets import unsupervised_dataset
from fastor import util
from model import Model

def rescale(data):
    data = data/2.0*255.0
    data[data > 255.0] = 255.0
    return data

model = Model('deconv-stl10_lr01', '/experiments/deconv/stl10')
#util.load_checkpoint(model, '/experiments/deconv/stl10/deconv-stl10_lr01-09m-11d-10h-51m-31s.pkl')


data = numpy.load('/data/stl10_matlab/unsupervised.npy')
data = numpy.float32(data)
data /= 255.0
data *= 2.0
test_data = data[90000::, :, :, :]

test_dataset = unsupervised_dataset.UnsupervisedDataset(test_data)
test_iterator = test_dataset.iterator(mode='sequential', batch_size=128)

x = test_iterator.next()
x = x.transpose(1,2,3,0)

x_hat = model.prediction(x)
x_hat = x_hat.transpose(3,1,2,0)
x = x.transpose(3,1,2,0)

for i in range(128):
    image_array = numpy.uint8(rescale(numpy.hstack((x[i,:,:,:],x_hat[i,:,:,:]))))
    image = Image.fromarray(image_array)
    filename = 'recon-%02d.jpeg' % i
    filepath = os.path.join('/experiments/results/stl10-e006/', filename)
    image.save(filepath)
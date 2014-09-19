import sys

import numpy
from fastor import util

import pylearn2.datasets.cifar10 as cifar10
from model import Model

print('Start')
model = Model(sys.argv[1], sys.argv[2])
monitor = util.Monitor(model)

print('Loading Data')
train_iterator = util.get_cifar_iterator('train', 
                                    mode='random_uniform', 
                                    batch_size=128, 
                                    num_batches=100000,                                     
                                    rescale=True)

print('Trainig Model')
for batch in train_iterator:    
    monitor.start()
    error = model.train(batch)
    monitor.stop(error)    
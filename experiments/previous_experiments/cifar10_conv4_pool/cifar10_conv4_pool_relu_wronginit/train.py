import sys

import numpy
from fastor import util

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

test_iterator = util.get_cifar_iterator('test', 
                                    mode='sequential', 
                                    batch_size=128,                                      
                                    rescale=True)

test_batch = test_iterator.next()
recon_visualizer = util.ReconVisualizer(model, test_batch)

print('Trainig Model')
for batch in train_iterator:    
    monitor.start()
    error = model.train(batch/2)
    monitor.stop(error) 
    recon_visualizer.run()   
import sys

import numpy
from fastor import util

from model import Model

print('Start')
model = Model(sys.argv[1], sys.argv[2])
#checkpoint = '/experiments/cifar_conv4_pool_relu_right/checkpoints/cifar_conv4_pool_relu_right-09m-23d-00h-37m-40s.pkl'
#util.load_checkpoint(model, checkpoint)
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

test_x_batch, test_y_batch = test_iterator.next()
recon_visualizer = util.ReconVisualizer(model, test_x_batch, steps=200)
filter_visualizer = util.FilterVisualizer(model, steps=200)


#model.learning_rate_symbol.set_value(0.000005/10)
print('Trainig Model')
for x_batch, y_batch in train_iterator:        
    monitor.start()
    error = model.train(x_batch/2)
    monitor.stop(error) 
    recon_visualizer.run()   
    filter_visualizer.run()
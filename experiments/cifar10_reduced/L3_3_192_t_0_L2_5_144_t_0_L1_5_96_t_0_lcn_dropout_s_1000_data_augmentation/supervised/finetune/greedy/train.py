import os
import sys
sys.path.append('../../..')

import numpy

from fastor import util
from fastor.datasets import supervised_dataset

import checkpoints
from model import SupervisedModel

print('Start')

pid = os.getpid()
print('PID: {}'.format(pid))
f = open('pid', 'wb')
f.write(str(pid)+'\n')
f.close()

model = SupervisedModel('experiment', './', learning_rate=1e-2)
checkpoint = checkpoints.supervised_greedy
util.set_parameters_from_unsupervised_model(model, checkpoint)
monitor = util.Monitor(model)

# Loading CIFAR-10 dataset
print('Loading Data')
train_iterator = util.get_cifar_iterator_reduced('train',
                                    mode='random_uniform',
                                    batch_size=128,
                                    num_batches=100000,
                                    rescale=True,
                                    num_samples_per_class=1000, 
                                    which_split=0)

test_iterator = util.get_cifar_iterator('test',
                                    mode='random_uniform',
                                    batch_size=128,
                                    num_batches=100000,
                                    rescale=True)

normer = util.Normer2(filter_size=5, num_channels=3)
augmenter = util.DataAugmenter(2, (32, 32))

print('Training Model')
for x_batch, y_batch in train_iterator:
    x_batch = augmenter.run(x_batch)
    x_batch = normer.run(x_batch)
    y_batch = numpy.int64(numpy.argmax(y_batch, axis=1))
    monitor.start()
    log_prob, accuracy = model.train(x_batch, y_batch)
    monitor.stop(1-accuracy) # monitor takes error instead of accuracy    
    
    if monitor.test:
        monitor.start()
        x_test_batch, y_test_batch = test_iterator.next()
        x_test_batch = normer.run(x_test_batch)
        y_test_batch = numpy.int64(numpy.argmax(y_test_batch, axis=1))
        test_accuracy = model.eval(x_test_batch, y_test_batch)
        monitor.stop_test(1-test_accuracy)

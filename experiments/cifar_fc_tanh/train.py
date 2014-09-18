import sys

import pylearn2.datasets.cifar10 as cifar10

from fastor import util
from fastor.datasets import unsupervised_dataset

from model import Model

print 'Creating model...'
model = Model(sys.argv[1], sys.argv[2])
monitor = util.Monitor(model)

print 'Loading dataset...'
train_dataset = cifar10.CIFAR10('train', rescale=True)
train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=1000000)

print 'Start training...'
for batch in train_iterator:
        monitor.start()
        error = model.train(batch-1)
        monitor.stop(error)
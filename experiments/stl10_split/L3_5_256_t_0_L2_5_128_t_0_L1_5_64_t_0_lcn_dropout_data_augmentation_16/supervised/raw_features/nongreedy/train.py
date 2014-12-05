import argparse
import os
import sys
sys.path.append('../../..')

import numpy

from fastor import util
from fastor.datasets import supervised_dataset

import checkpoints
from model import SupervisedModel

print('Start')

parser = argparse.ArgumentParser(prog='train_raw_features_nongreedy', description='Script to train deconvolutional network in nongreedy fashion.')
parser.add_argument("-s", "--split", default='0', help='Training split of stl10 to use. (0-9)')
args = parser.parse_args()

train_split = int(args.split)
if train_split < 0 or train_split > 9:
    raise Exception("Training Split must be in range 0-9.")
print('Using STL10 training split: {}'.format(train_split))

pid = os.getpid()
print('PID: {}'.format(pid))
f = open('pid_'+str(train_split), 'wb')
f.write(str(pid)+'\n')
f.close()

model = SupervisedModel('experiment', './')
checkpoint = checkpoints.supervised_nongreedy
util.set_parameters_from_unsupervised_model(model, checkpoint)
monitor = util.Monitor(model, checkpoint_directory='checkpoints_'+str(train_split))

model.conv1.trainable = False
model.conv2.trainable = False
model.conv3.trainable = False
model._compile()

# Loading STL-10 dataset
print('Loading Data')
X_train = numpy.load('/data/stl10_matlab/train_splits/train_X_'+str(train_split)+'.npy')
y_train = numpy.load('/data/stl10_matlab/train_splits/train_y_'+str(train_split)+'.npy')
X_test = numpy.load('/data/stl10_matlab/test_X.npy')
y_test = numpy.load('/data/stl10_matlab/test_y.npy')

X_train = numpy.float32(X_train)
X_train /= 255.0
X_train *= 2.0

X_test = numpy.float32(X_test)
X_test /= 255.0
X_test *= 2.0

train_dataset = supervised_dataset.SupervisedDataset(X_train, y_train)
test_dataset = supervised_dataset.SupervisedDataset(X_test, y_test)
train_iterator = train_dataset.iterator(
    mode='random_uniform', batch_size=128, num_batches=100000)
test_iterator = test_dataset.iterator(
    mode='random_uniform', batch_size=128, num_batches=100000)

# Create object to local contrast normalize a batch.
# Note: Every batch must be normalized before use.
normer = util.Normer2(filter_size=5, num_channels=3)
augmenter = util.DataAugmenter(16, (96, 96))

print('Training Model')
for x_batch, y_batch in train_iterator:        
    x_batch = x_batch.transpose(1, 2, 3, 0) 
    x_batch = augmenter.run(x_batch)  
    x_batch = normer.run(x_batch)
    # y_batch = numpy.int64(numpy.argmax(y_batch, axis=1))
    monitor.start()
    log_prob, accuracy = model.train(x_batch, y_batch-1)
    monitor.stop(1-accuracy) # monitor takes error instead of accuracy    
    
    if monitor.test:
        monitor.start()
        x_test_batch, y_test_batch = test_iterator.next()
        x_test_batch = x_test_batch.transpose(1, 2, 3, 0)
        x_test_batch = normer.run(x_test_batch)
        # y_test_batch = numpy.int64(numpy.argmax(y_test_batch, axis=1))
        test_accuracy = model.eval(x_test_batch, y_test_batch-1)
        monitor.stop_test(1-test_accuracy)

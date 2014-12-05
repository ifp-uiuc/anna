import sys
import cPickle
import argparse

import numpy

from fastor import util
from fastor.datasets import supervised_dataset
from model import SupervisedModel

parser = argparse.ArgumentParser(prog='stl10_compute_train_test_error', description='Script to compute training and testing error of a specific checkpoint from a split of stl10.')
parser.add_argument("-s", "--split", default='0', help='Training split of stl10 to use. (0-9)')
parser.add_argument("checkpoint_file", help='Checkpoint file (.pkl) to load.')
args = parser.parse_args()

model = SupervisedModel('xxx', './')
model.fc4.dropout = 0.0
model._compile()
num_channels = model.conv1.filter_shape[0]
filter_size = model.conv1.filter_shape[1]


train_split = int(args.split)
if train_split < 0 or train_split > 9:
    raise Exception('Training split must be in range 0-9.')


checkpoint = args.checkpoint_file
util.load_checkpoint(model, checkpoint)

# Load STL-10 data
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
train_iterator = train_dataset.iterator(mode='sequential', batch_size=128)
test_iterator = test_dataset.iterator(mode='sequential', batch_size=128)

# Create object to local contrast normalize a batch.
# Note: Every batch must be normalized before use.
normer = util.Normer2(filter_size=filter_size, num_channels=num_channels)

train_accuracy = []
test_accuracy = []

# Compute accuracy on each training batch
print('\nTraining Accuray:')
i = 0
for x_batch, y_batch in train_iterator:
    #y_batch = numpy.int64(numpy.argmax(y_batch, axis=1))
    x_batch = x_batch.transpose(1, 2, 3, 0)
    x_batch = normer.run(x_batch)
    train_batch_accuracy = model.eval(x_batch, y_batch-1)
    print('Train Batch {} Accuracy: {}'.format(i, train_batch_accuracy))
    train_accuracy.append(train_batch_accuracy)
    i+=1

# Compute accuracy on each test batch
print('\nTesting Accuray:')
i = 0
for x_batch, y_batch in test_iterator:
    #y_batch = numpy.int64(numpy.argmax(y_batch, axis=1))
    x_batch = x_batch.transpose(1, 2, 3, 0)
    x_batch = normer.run(x_batch)
    test_batch_accuracy = model.eval(x_batch, y_batch-1) 
    print('Test Batch {} Accuracy: {}'.format(i, test_batch_accuracy))
    test_accuracy.append(test_batch_accuracy)
    i+=1

# Compute overall training and testing accuracy
train_accuracy = numpy.mean(numpy.hstack(train_accuracy))
test_accuracy = numpy.mean(numpy.hstack(test_accuracy))

print('\nOverall Training Accuracy {}'.format(train_accuracy))
print('Overall Testing Accuracy {}\n'.format(test_accuracy))

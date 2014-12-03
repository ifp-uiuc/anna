import argparse
import os
import sys
sys.path.append('../../..')

import numpy

from fastor import util
from fastor.datasets import supervised_dataset

from model import SupervisedModel

def orthogonalize(w):
    # Orthogonalize square matrices.
    # Or left orthogonalize overcomplete matrices.
    # Simply gets an SVD decomposition, and sets the singular values to ones.
    dim2, dim1 = w.shape
    dim = numpy.min((dim1, dim2))
    u, s, v = numpy.linalg.svd(w)
    S = numpy.zeros((dim2,dim1))
    s = s/s
    S[:dim,:dim] = numpy.diag(s)
    w = numpy.dot(u,numpy.dot(S,v))
    w = numpy.float32(w)
    return w

def conv_orthogonalize(w, k=1.0):
    # Reshape filters into a matrix
    channels, width, height, filters = w.shape
    w = w.reshape(channels*width*height, filters).transpose(1,0)

    # Orthogonalize the matrix
    w = orthogonalize(w)

    # Contruct 2D hamming window
    hamming1 = numpy.hamming(width)
    hamming2 = numpy.hamming(height)
    hamming = numpy.outer(hamming1, hamming2)

    # Use it to mask the input to w
    mask = numpy.tile(hamming[None,:,:], (channels,1,1))
    mask = mask.reshape(channels*width*height)*k
    m = numpy.diag(mask)
    w = numpy.dot(w, m)

    # Reshape the matrix into filters
    w = w.transpose(1,0)
    w =  w.reshape(channels, width, height, filters)
    w = numpy.float32(w)
    return w

print('Start')

parser = argparse.ArgumentParser(prog='train_raw_features_patches', description='Script to train deconvolutional network initialized with random patches.')
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
monitor = util.Monitor(model, checkpoint_directory='checkpoints_'+str(train_split))

model.conv1.trainable = False
model.conv2.trainable = False
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

# Grab batch for patch extraction.
x_batch, y_batch = train_iterator.next()
x_batch = x_batch.transpose(1, 2, 3, 0)
x_batch = normer.run(x_batch)
# Grab some patches to initialize weights.
patch_grabber = util.PatchGrabber(64, 5)
patches = patch_grabber.run(x_batch)*0.05
model.conv1.W.set_value(patches)
# Orthogonalize second layer weights.
W2 = model.conv2.W.get_value()
W2 = conv_orthogonalize(W2)
# Scale second layer weights.
s=5.0
model.conv2.W.set_value(W2*s)


print('Training Model')
for x_batch, y_batch in train_iterator:     
    x_batch = x_batch.transpose(1, 2, 3, 0)   
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

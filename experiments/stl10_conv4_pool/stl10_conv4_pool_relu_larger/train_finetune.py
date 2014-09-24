import sys
import numpy
from fastor import util
from fastor.datasets import supervised_dataset

from model import Model

print('Start')
model = Model(sys.argv[1], sys.argv[2])
checkpoint = sys.argv[3]
print('Loading from {}'.format(checkpoint))

util.load_checkpoint(model, checkpoint)
monitor = util.Monitor(model)

print('Loading Data')
data = numpy.load('/data/stl10_matlab/unsupervised.npy')
data = numpy.float32(data)
data /= 255.0
train_data = data[0:90000, :, :, :]
test_data = data[90000::, :, :, :]

train_dataset = supervised_dataset.SupervisedDataset(train_data)
test_dataset = supervised_dataset.SupervisedDataset(test_data)
train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=100000)
test_iterator = test_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=100000)

model.learning_rate_symbol.set_value(0.0001)
print('Trainig Model')
for x_batch, y_batch in train_iterator:        
    y_batch = numpy.int64(numpy.argmax(y_batch, axis=1))
    monitor.start()
    log_prob, accuracy = model.classify(x_batch/2, y_batch)
    monitor.stop(1-accuracy) # monitor takes error instead of accuracy    
    
    if monitor.test:
        monitor.start()
        x_test_batch, y_test_batch = test_iterator.next()
        y_test_batch = numpy.int64(numpy.argmax(y_test_batch, axis=1))
        test_accuracy = model.accuracy(x_test_batch/2, y_test_batch)
        monitor.stop_test(1-test_accuracy)

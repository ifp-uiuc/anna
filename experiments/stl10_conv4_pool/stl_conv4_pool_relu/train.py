import sys

import numpy
from fastor import util
from fastor.datasets import unsupervised_dataset

from model import Model


if __name__ == "__main__":
    print('Start')
    model = Model(sys.argv[1], sys.argv[2])
    monitor = util.Monitor(model)

    print('Load Data')
    data = numpy.load('/data/stl10_matlab/unsupervised.npy')
    data = numpy.float32(data)
    #data = data/255.0*2.0
    data /= 255.0
    data *= 2.0
    train_data = data[0:90000, :, :, :]
    test_data = data[90000::, :, :, :]
    train_dataset = unsupervised_dataset.UnsupervisedDataset(train_data)
    test_dataset = unsupervised_dataset.UnsupervisedDataset(test_data)
    train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=100000)
    test_iterator = test_dataset.iterator(mode='sequential', batch_size=128)
    # evaluator = util.Evaluator(model, test_dataset, steps=1000)

    test_batch = test_iterator.next()
    recon_visualizer = util.ReconVisualizer(model, test_batch)

    print('Training Model')
    for batch in train_iterator:
        batch = batch.transpose(1,2,3,0)
        monitor.start()
        error = model.train(batch/2)
        monitor.stop(error)
    # evaluator.run()
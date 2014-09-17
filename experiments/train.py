import numpy

from pylearn2.space import Conv2DSpace

import unsupervised_experiments.datasets.unsupervised_dataset as unsupervised_dataset
import unsupervised_experiments.model as Model
import unsupervised_experiments.util

if __name__ == "__main__":
    model = Model.Model('deconv-stl10-32x64-old', '/experiments/deconv/stl10-32x64')
    checkpoint = '/experiments/deconv/stl10-32x64/deconv-stl10-32x64-old-09m-13d-13h-48m-26s.pkl'
    util.load_checkpoint(model, checkpoint)
    monitor = util.Monitor(model)

    data = numpy.load('/data/stl10_matlab/unsupervised.npy')
    data = numpy.float32(data)
    data = data/255.0*2.0
    train_data = data[0:90000, :, :, :]
    test_data = data[90000::, :, :, :]

    train_dataset = unsupervised_dataset.UnsupervisedDataset(train_data)
    test_dataset = unsupervised_dataset.UnsupervisedDataset(test_data)
    train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=100000)

    # evaluator = util.Evaluator(model, test_dataset, steps=1000)

    for batch in train_iterator:
        batch = batch.transpose(1,2,3,0)
        monitor.start()
        error = model.train(batch)
        monitor.stop(error)
        # evaluator.run()
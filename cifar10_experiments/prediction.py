import pylearn2.datasets.cifar10 as cifar10
from pylearn2.space import Conv2DSpace

import model as Model
import util

if __name__ == "__main__":
    model = Model.Model('deconv-cifar10', '/experiments/deconv/cifar10/')
    util.load_checkpoint(model, '/experiments/cifar10/cifar10-09m-09d-01h-25m-08s.pkl')

    test_dataset = cifar10.CIFAR10(which_set='test',
                                   # rescale=True,
                                   axes=['c', 0, 1, 'b']
                                   )
    input_convspace = Conv2DSpace(shape=(32, 32), num_channels=3,
                              axes=('c', 0, 1, 'b'))
    data_specs = (input_convspace,'features')
    test_iterator = test_dataset.iterator(mode='sequential', batch_size=128, data_specs=data_specs)

    x = test_iterator.next()

    x_hat = model.prediction(x)
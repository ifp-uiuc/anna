import pylearn2.datasets.cifar10 as cifar10

import model as Model
import util

if __name__ == "__main__":
    model = Model.Model('deconv-cifar10', '/experiments/deconv/cifar10/')
    util.load_checkpoint(model, '/experiments/cifar10/cifar10-09m-09d-01h-25m-08s.pkl')

    test_dataset = cifar10.CIFAR10(which_set='test',
                                   rescale=True,
                                   axes=['c', 0, 1, 'b']
                                   )
    test_iterator = test_dataset.iterator(mode='sequential', batch_size=128, topo=True)

    x = test_iterator.next()

    x_hat = model.prediction(x)
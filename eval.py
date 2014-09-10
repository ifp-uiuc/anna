import pylearn2.datasets.cifar10 as cifar10
from pylearn2.space import Conv2DSpace

import model as Model
import util

if __name__ == "__main__":
    model = Model.Model('deconv-cifar10-fixed-144x144', '/experiments/deconv/cifar10-fixed-144x144/')
    util.load_checkpoint(model, '/experiments/deconv/cifar10-fixed-144x144/deconv-cifar10-fixed-144x144-09m-10d-10h-17m-53s.pkl')

    test_dataset = cifar10.CIFAR10(which_set='test',
                                   rescale=True,
                                   axes=['c', 0, 1, 'b']
                                   )

    evaluator = util.Evaluator(model, test_dataset, steps=1)

    evaluator.run()
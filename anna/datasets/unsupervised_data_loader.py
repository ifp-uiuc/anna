import os

import numpy


class UnsupervisedDataContainer(object):
    def __init__(self, X):
        self.X = X

    def next(self):
        pass


class UnsupervisedDataLoader(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        assert os.path.exists(self.dataset_path), \
            'Dataset directory %s does not exist!' % dataset_path

    def load(self):
        # Load unlabeled data matrix from disk
        X = numpy.load(os.path.join(self.dataset_path, 'X.npy'))
        # Initialize a data_container object and return it
        unsupervised_data_container = UnsupervisedDataContainer(X)
        return unsupervised_data_container

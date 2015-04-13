import os

import numpy


class SupervisedDataContainer(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def next(self):
        pass


class SupervisedDataLoader(object):
    def __init__(self, dataset_path):  # mode='train', fold=0):
        self.dataset_path = dataset_path

        # Check if dataset_path exists
        assert os.path.exists(self.dataset_path), \
            'Dataset directory does not exist!'

    def load(self, fold=0):
        fold_path = os.path.join(self.dataset_path, 'folds.npy')
        if os.path.exists(fold_path):
            supervised_data_container = self._load_with_folds(fold)
        else:
            supervised_data_container = self._load_without_folds()

        return supervised_data_container

    def _load_with_folds(self, fold):
        X = numpy.load(os.path.join(self.dataset_path, 'X.npy'))
        y = numpy.load(os.path.join(self.dataset_path, 'y.npy'))
        folds = numpy.load(os.path.join(self.dataset_path, 'folds.npy'))

        assert fold <= folds.max(), \
            'Fold number exceeds available number of folds. Please try again.'

        mask = (folds == fold)

        X = X[mask, :, :, :]
        y = y[mask]

        # Create supervised data container and return it
        supervised_data_container = SupervisedDataContainer(X, y)
        return supervised_data_container

    def _load_without_folds(self):
        X = numpy.load(os.path.join(self.dataset_path, 'X.npy'))
        y = numpy.load(os.path.join(self.dataset_path, 'y.npy'))

        # Create supervised data container and return it
        supervised_data_container = SupervisedDataContainer(X, y)
        return supervised_data_container


class SupervisedDataLoaderCrossVal(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        # Check if dataset_path exists
        assert os.path.exists(self.dataset_path), \
            'Dataset directory does not exist!'

    def load(self, mode='train', fold=0):
        assert mode == 'train' or mode == 'test', \
            'Please enter train or test mode!'

        fold_path = os.path.join(self.dataset_path, 'folds.npy')
        assert os.path.exists(fold_path), \
            'There is folds.npy in specified dataset directory.'

        supervised_data_container = self._load_with_folds(fold, mode)

        return supervised_data_container

    def _load_with_folds(self, fold, mode='train'):
        X = numpy.load(os.path.join(self.dataset_path, 'X.npy'))
        y = numpy.load(os.path.join(self.dataset_path, 'y.npy'))
        folds = numpy.load(os.path.join(self.dataset_path, 'folds.npy'))

        assert fold <= folds.max(), \
            'Fold number exceeds available number of folds. Please try again.'

        if mode == 'train':
            mask = (folds != fold)
        else:
            # mode = 'test'
            mask = (folds == fold)

        X = X[mask, :, :, :]
        y = y[mask]

        # Create supervised data container and return it
        supervised_data_container = SupervisedDataContainer(X, y)
        return supervised_data_container

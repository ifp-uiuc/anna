import numpy as np


#
# TODO(pkhorrami4) Consider datatype of input (convert to float32, etc.)
# TODO(pkhorrami4) Decide datatype for conversion from theano.config.floatX
# TODO(pkhorrami4) Add centering + rescaling of data as option
# TODO(pkhorrami4) Add verbosity flag option
#


class UnsupervisedDataset(object):
    # Class to construct an unsupervised dataset

    def __init__(self, X):  # , batch_size=None, iter_type=None):
        # print('Dataset loaded with size: {}'.format(X.shape))
        self.X = X
        self.n_samples = X.shape[0]
        self.n_channels = X.shape[1]
        self.height = X.shape[2]
        self.width = X.shape[3]

    def __iter__(self):
        return self.iter  # self.iterator()

    def iterator(self,
                 mode='sequential',
                 batch_size=None,
                 num_batches=None,
                 rng_seed=0):

        if(batch_size is not None):
            if(batch_size > self.n_samples):
                raise ValueError("batch_size is larger than number of samples")

        if(mode == 'sequential'):
            if batch_size is None:
                if num_batches is not None:
                    batch_size = int(self.n_samples / num_batches)
                else:
                    raise ValueError("need one of batch_size, num_batches"
                                     "for sequential batch iteration")

            elif batch_size is not None:
                if num_batches is not None:
                    max_num_batches = int(np.ceil(self.n_samples / batch_size))
                    if num_batches > max_num_batches:
                        raise ValueError("dataset of %d examples can only "
                                         "provide %d batches with batch_size "
                                         "%d, but %d batches were requested" %
                                         (self.n_samples, max_num_batches,
                                          batch_size, num_batches))
                else:
                    num_batches = int(self.n_samples / batch_size)

            self.iter = DatasetIteratorSequential(self.X, batch_size,
                                                  num_batches, self.n_samples)

        elif(mode == 'random_uniform'):
            if batch_size is None:
                raise ValueError("batch_size cannot be None for random "
                                 "uniform iteration")
            elif num_batches is None:
                raise ValueError("num_batches cannot be None for random "
                                 "uniform iteration")
            self.iter = DatasetIteratorRandomUniform(self.X,
                                                     batch_size,
                                                     num_batches,
                                                     self.n_samples,
                                                     rng_seed)

        elif(mode == 'random_uniform_no_rep'):
            if batch_size is None:
                if num_batches is not None:
                    # Floor Function chosen to ensure that uneven segment
                    # (i.e. "runt") is ignored
                    batch_size = int(np.floor(np.float32(self.n_samples) /
                                     num_batches))
                else:
                    raise ValueError("need one of batch_size, num_batches"
                                     "for sequential batch iteration")

            elif batch_size is not None:
                if num_batches is not None:
                    max_num_batches = int(np.ceil(self.n_samples / batch_size))
                    if num_batches > max_num_batches:
                        raise ValueError("dataset of %d examples can only "
                                         "provide %d batches with batch_size "
                                         " %d, but %d batches were requested" %
                                         (self.n_samples, max_num_batches,
                                          batch_size, num_batches))
                else:
                    # Floor Function chosen to ensure that uneven segment
                    # (i.e. "runt") is ignored
                    num_batches = int(np.floor(np.float32(self.n_samples) /
                                      batch_size))

            self.iter = DatasetIteratorRandomUniformNoRep(self.X,
                                                          batch_size,
                                                          num_batches,
                                                          self.n_samples,
                                                          rng_seed)

        return self.iter

    def get_num_samples(self):
        return self.n_samples

    def get_batch(self):
        return self.iter.next()


class BasicIterator(object):

    def __init__(self, X, batch_size=None, num_batches=None, num_samples=None):

        self.X = X
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_samples = num_samples
        # print('Batch Size: {}'.format(batch_size))
        # print('Num Batches: {}'.format(num_batches))
        # print('Num Samples: {}'.format(num_samples))

        # Batch Counter
        self.batch_count = 0
        # Sample Counter
        self.sample_count = 0
        # Batch Indices
        self.last = 0

    def reset(self):
        self.batch_count = 0
        self.sample_count = 0
        self.last = 0


class DatasetIteratorSequential(BasicIterator):

    #
    # Iterator that traverses the data by extracting sequential slices
    # of size (batch_size) from the data tensor
    #
    def __init__(self, X, batch_size=None, num_batches=None, num_samples=None):
        # print('Using Sequential Iterator')
        super(DatasetIteratorSequential, self).__init__(X,
                                                        batch_size,
                                                        num_batches,
                                                        num_samples)

    def __iter__(self):
        return self

    def next(self):
        # print('In Dataset_Iter_Seq_Next')
        if (self.batch_count >= self.num_batches or
           self.sample_count >= self.num_samples):
            raise StopIteration()
        # fixes the problem where (self._num_samples % self._batch_size) != 0
        elif (self.sample_count + self.batch_size) > self.num_samples:
            self.last = slice(self.sample_count, self.num_samples)
            self.sample_count = self.num_samples
            return self.X[self.last, :, :, :]
        else:
            self.last = slice(self.sample_count,
                              self.sample_count + self.batch_size)
            self.sample_count += self.batch_size
            self.batch_count += 1
            return self.X[self.last, :, :, :]


class DatasetIteratorRandomUniform(BasicIterator):

    #
    # Iterator that uniformly samples batches of size (batch_size)
    # from the data tensor (with replacement)
    #

    def __init__(self, X,
                 batch_size=None,
                 num_batches=None,
                 num_samples=None,
                 rng_seed=0):
        # print('Using Random Uniform Iterator (with replacement)')
        np.random.seed(rng_seed)
        super(DatasetIteratorRandomUniform, self).__init__(X, batch_size,
                                                           num_batches,
                                                           num_samples)

    def __iter__(self):
        return self

    def next(self):
        if self.batch_count >= self.num_batches:
            raise StopIteration()
        else:
            self.last = np.random.random_integers(low=0,
                                                  high=self.num_samples - 1,
                                                  size=(self.batch_size,))
            self.batch_count += 1
            return self.X[self.last, :, :, :]


class DatasetIteratorRandomUniformNoRep(BasicIterator):

    #
    # Iterator that uniformly samples batches of size (batch_size)
    # from the data tensor (without replacement)
    #
    def __init__(self, X,
                 batch_size=None,
                 num_batches=None,
                 num_samples=None,
                 rng_seed=0):
        # print('Using Random Uniform Iterator (w/o replacement)')
        np.random.seed(rng_seed)
        self.remain_samples = np.arange(num_samples)
        super(DatasetIteratorRandomUniformNoRep, self).__init__(X,
                                                                batch_size,
                                                                num_batches,
                                                                num_samples)

    def __iter__(self):
        return self

    def next(self):
        if self.batch_count >= self.num_batches:
            raise StopIteration()

        else:
            idx = np.random.permutation(
                np.arange(len(self.remain_samples)))[0:self.batch_size]
            self.last = self.remain_samples[idx]
            self.remain_samples = np.setdiff1d(self.remain_samples, self.last)
            self.batch_count += 1
            return self.X[self.last, :, :, :]


if __name__ == "__main__":
    # print('Making Datset Object')
    X = np.zeros((1024, 3, 5, 5))
    d = UnsupervisedDataset(X)
    d.iterator(batch_size=128)
    # d.iterator(mode='random_uniform', batch_size=128, num_batches = 1000,
    #           rng_seed=10)
    # d.iterator(mode='RAND_UNIF_nr', batch_size=128)
    for i, b in enumerate(d):
        print('Batch {} --- {}'.format(i+1, b.shape))

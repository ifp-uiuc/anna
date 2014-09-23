"""Utils for training neural networks.
"""
import os
import Image
from time import time
from datetime import datetime
import cPickle

import numpy
from pylearn2.space import Conv2DSpace, VectorSpace, CompositeSpace
from pylearn2.datasets import cifar10


def load_checkpoint(model, checkpoint_path):
    all_parameters = model.all_parameters_symbol
    f = open(checkpoint_path, 'rb')
    checkpoint = cPickle.load(f)
    f.close()
    
    [model_param.set_value(checkpoint_param) for model_param, checkpoint_param 
     in zip(all_parameters, checkpoint)]

def save_checkpoint(model):
    all_parameters = model.all_parameters_symbol
    checkpoint = [param.get_value() for param in all_parameters]
    tt = datetime.now()
    time_string = tt.strftime('%mm-%dd-%Hh-%Mm-%Ss')
    checkpoint_name = '%s-%s.pkl' % (model.name, time_string)
    print(model.path)
    checkpoint_path = os.path.join(model.path, 'checkpoints', checkpoint_name)

    print 'Saving model checkpoint to: %s' % checkpoint_path
    f = open(checkpoint_path, 'wb')
    cPickle.dump(checkpoint, f)
    f.close()

def rescale(data):
    data = data/2.0*255.0
    data[data > 255.0] = 255.0
    return data

class ReconVisualizer(object):
    def __init__(self, model, batch, steps=2000):
        self.model = model
        self.batch = batch
        self.count = 0
        self.steps = steps
        self.save_path = os.path.join(self.model.path, 'recon')
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def run(self):
        if self.count % self.steps == 0:
            self._save()
        self.count += 1
    
    def _save(self):
        tt = datetime.now()
        time_string = tt.strftime('%mm-%dd-%Hh-%Mm-%Ss')
        sub_path = os.path.join(self.save_path, time_string)
        
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        
        prediction = self.model.prediction(self.batch)
        for i in range(128):
            image = self.batch[:, :, :, i]
            image = image.transpose(1, 2, 0)

            recon = numpy.array(prediction[:, :, :, i])
            recon = recon.transpose(1, 2, 0)

            image_array = numpy.uint8(rescale(numpy.hstack((image, recon))))

            to_save = Image.fromarray(image_array)
            filename = 'recon-%02d.jpeg' % i
            filepath = os.path.join(sub_path, filename)
            to_save.save(filepath)

class Monitor(object):
    errors = []
    times = []
    big_errors = []
    big_times = []

    def __init__(self, model, step_number=0, best=1, short_steps=10, long_steps=50, save_steps=2000, test_steps=50):
        self.step_number = step_number
        self.best = best
        self.short_steps = short_steps
        self.long_steps = long_steps
        self.save_steps = save_steps
        self.model = model
        self.test = False
        self.test_steps = test_steps

        # Check if model.path exists, if not create it (with a checkpoint folder)
        if model.path and not os.path.exists(os.path.join(model.path, 'checkpoints')):
            os.makedirs(os.path.join(model.path, 'checkpoints'))            

        
    def start(self):
        self.tic = time()
    
    def stop_test(self, error):
        if self.test:
            self.toc = time()
            _time = self.toc - self.tic
            print '&%d, test error: %.5f, time: %.2f' % (self.step_number, error, _time)

    def stop(self, error):
        self.toc = time()
        _time = self.toc-self.tic
        self.errors.append(error)
        self.times.append(_time)
        self.big_errors.append(error)
        self.big_times.append(_time)
        if self.step_number % self.test_steps == 0:
            self.test = True
        else:
            self.test = False
        if self.step_number % self.save_steps == 0:
            save_checkpoint(self.model)
        if self.step_number % self.long_steps == 0:
            mean_error = numpy.mean(self.big_errors)
            mean_time = numpy.mean(self.big_times)
            print '*%d, train error: %.5f, time: %.2f' % (self.step_number, mean_error, mean_time)
            self.big_errors = []
            self.big_times = []
        if self.step_number % self.short_steps == 0:
            mean_error = numpy.mean(self.errors)
            mean_time = numpy.mean(self.times)
            print '%d, train error: %.5f, time: %.2f' % (self.step_number, mean_error, mean_time)
            self.errors = []
            self.times = []
        self.step_number += 1
        
class Evaluator(object):
    def __init__(self, model, dataset, steps=100):
        self.step_number = 1
        self.steps = steps
        self.model = model
        self.dataset = dataset
    
    def run(self):
        input_convspace = Conv2DSpace(shape=(32, 32), num_channels=3,
                                      axes=('c', 0, 1, 'b'))
        data_specs = (input_convspace,'features')
        iterator = self.dataset.iterator(mode='sequential', batch_size=128, data_specs=data_specs)
        errors = []
        if self.step_number % self.steps == 0:
            tic = time()
            for batch in iterator:
                error = self.model.eval(batch)
                errors.append(error)
            toc = time()
            _time = toc-tic
            mean_error = numpy.mean(errors)
            print '%d, test error: %.3f, time: %.2f' % (self.step_number, mean_error, _time)
        self.step_number += 1

def get_cifar_iterator(which_set, mode='sequential', batch_size=128, num_batches=None, center=False, rescale=True, axes=['c', 0, 1, 'b'], targets=False):
    dataset = cifar10.CIFAR10(which_set=which_set,
                              center=center,
                              rescale=rescale,
                              axes=axes
                              )
    input_convspace = Conv2DSpace(shape=(32, 32), num_channels=3,
                              axes=axes)
    #target_space = VectorSpace(dim=10)
    #data_specs = (input_convspace,'features')

    target_space = VectorSpace(dim=10)
    data_specs = (CompositeSpace((input_convspace, target_space)), ("features", "targets"))

    if num_batches:
        iterator = dataset.iterator(mode=mode, batch_size=batch_size, num_batches=num_batches, data_specs=data_specs)
    else:
        iterator = dataset.iterator(mode=mode, batch_size=batch_size, data_specs=data_specs)
    return iterator
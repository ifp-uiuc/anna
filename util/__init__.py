"""Utils for training neural networks.
"""
import os
import numpy
from time import time
from datetime import datetime
import cPickle

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
    checkpoint_path = os.path.join(model.path, checkpoint_name)

    print 'Saving model checkpoint to: %s' % checkpoint_path
    f = open(checkpoint_path, 'wb')
    cPickle.dump(checkpoint, f)
    f.close()

class Monitor(object):
    errors = []
    times = []
    big_errors = []
    big_times = []

    def __init__(self, model, step_number=0, best=1, short_steps=10, long_steps=50):
        self.step_number = step_number
        self.best = best
        self.short_steps = short_steps
        self.long_steps = long_steps
        self.model = model
        
    def start(self):
        self.tic = time()
    
    def stop(self, error):
        self.toc = time()
        _time = self.toc-self.tic
        self.errors.append(error)
        self.times.append(_time)
        self.big_errors.append(error)
        self.big_times.append(_time)
        if self.step_number % self.long_steps == 0:
            mean_error = numpy.mean(self.big_errors)
            mean_time = numpy.mean(self.big_times)
            print '*%d, train error: %.3f, time: %.2f' % (self.step_number, mean_error, mean_time)
            self.big_errors = []
            self.big_times = []
            if mean_error < self.best:
                self.best = mean_error
                save_checkpoint(self.model)
        if self.step_number % self.short_steps == 0:
            mean_error = numpy.mean(self.errors)
            mean_time = numpy.mean(self.times)
            print '%d, train error: %.3f, time: %.2f' % (self.step_number, mean_error, mean_time)
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
        iterator = self.dataset.iterator(mode='sequential', batch_size=128, topo=True)
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
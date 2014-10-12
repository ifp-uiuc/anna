"""Utils for training neural networks.
"""
import os
import Image
from time import time
from datetime import datetime
from copy import deepcopy
import cPickle

import numpy

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.space import Conv2DSpace, VectorSpace, CompositeSpace
from pylearn2.datasets import cifar10

def load_checkpoint(model, checkpoint_path):
    all_parameters = model.all_save_parameters_symbol
    f = open(checkpoint_path, 'rb')
    checkpoint = cPickle.load(f)
    f.close()
    
    [model_param.set_value(checkpoint_param) for model_param, checkpoint_param 
     in zip(all_parameters, checkpoint)]

def save_checkpoint(model):
    all_parameters = model.all_save_parameters_symbol
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
        self.batch = deepcopy(batch)
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

class NormReconVisualizer(object):
    def __init__(self, model, batch, steps=2000):
        self.model = model
        self.batch = deepcopy(batch)
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
            image = deepcopy(self.batch[:, :, :, i])
            image = image.transpose(1, 2, 0)
            image_min = image.min()
            image -= image_min
            image_max = image.max()
            image /= image_max
            image *= 255

            recon = numpy.array(prediction[:, :, :, i])
            recon = recon.transpose(1, 2, 0)
            recon2 = deepcopy(recon)*1.0

            recon_mask = (numpy.sum(recon==0.0, axis=2)<3)
            recon_mask = 255*(numpy.tile(recon_mask[:,:,None],(1,1,3)))

            recon -= image_min
            recon /= image_max
            recon *= 255

            recon2 -= recon2.min()
            recon2 /= recon2.max()
            recon2 *= 255

            image_array = numpy.uint8(numpy.hstack((image, recon, recon2, recon_mask)))

            to_save = Image.fromarray(image_array)
            filename = 'recon-%02d.jpeg' % i
            filepath = os.path.join(sub_path, filename)
            to_save.save(filepath)

class FilterVisualizer(object):
    def __init__(self, model, steps=2000):
        self.model = model
        self.count = 0
        self.steps = steps
        self.save_path = os.path.join(self.model.path, 'filters')
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def run(self):
        if self.count % self.steps == 0:
            self._save()
        self.count += 1
    
    def _save(self):
        tt = datetime.now()
        time_string = tt.strftime('%mm-%dd-%Hh-%Mm-%Ss')
        
        W = self.model.conv1.W.get_value()
        W = W.transpose(1,2,0,3)

        row_list = []
        img_list = []
        k = 0
        
        rows = W.shape[3]/16        
        bar = 0.5*numpy.ones((W.shape[0],1,3))
        for i in range(rows):
            row_list.append(bar)
            for j in range(16):
                W0 = W[:,:,:,k]
                W0 -= W0.min()
                W0 /= W0.max()
                #W0[:,:,0] -= W0[:,:,0].min()
                #W0[:,:,0] /= W0[:,:,0].max()
                #W0[:,:,1] -= W0[:,:,1].min()
                #W0[:,:,1] /= W0[:,:,1].max()
                #W0[:,:,2] -= W0[:,:,2].min()
                #W0[:,:,2] /= W0[:,:,2].max()
                row_list.append(W0)
                row_list.append(bar)
                k += 1
            row_image = numpy.hstack(row_list)
            row_list = []
            bar_h = 0.5*numpy.ones((1,row_image.shape[1],3))
            img_list.append(bar_h)
            img_list.append(row_image)            
        img_list.append(bar_h)    
        img_image = numpy.vstack(img_list)

        to_save = Image.fromarray(numpy.uint8(255*img_image))
        filename = 'filters_'+time_string+'.png'
        filepath = os.path.join(self.save_path, filename)
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

class Normer(object):
    def __init__(self, filter_size=7):
        
        # magic numbers that make things work for stl10
        self.filter_size = 7
        self.pad = self.filter_size/2 -1
        self.num_channels = 3
        self.num_filters = 16
        input = T.ftensor4(name='input')
        filter = T.ftensor4(name='filter')
        gpu_input = gpu_contiguous(input)
        gpu_filter = gpu_contiguous(filter)
        
        self.conv_func = theano.function([input, filter], FilterActs(pad=3)(gpu_input, gpu_filter))
        n = self.num_channels * self.filter_size * self.filter_size
        self.w = numpy.float32(numpy.ones((self.num_channels, self.filter_size, self.filter_size, self.num_filters)))/n
        
    def run(self, x_batch):
        mean_batch = self.conv_func(x_batch, self.w)
        mean_batch = numpy.tile(numpy.array(mean_batch[0,:,:,:])[None, :, :], (3, 1, 1, 1))
        diff_batch = x_batch - mean_batch
        std_batch = self.conv_func(diff_batch**2, self.w)
        std_batch = numpy.tile(numpy.array(std_batch[0,:,:,:])[None, :, :], (3, 1, 1, 1))
        norm_batch = diff_batch/(numpy.array(std_batch)**(1/2))
        return norm_batch


class PatchGrabber(object):
    def __init__(self, num_patches, patch_size, num_channels=3):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_channels = num_channels

    def run(self, x_batch):
        image_size = x_batch.shape[1]
        batch_size = x_batch.shape[-1]

        patches = numpy.zeros((self.num_channels, self.patch_size, self.patch_size, self.num_patches),dtype=numpy.float32)

        for i_patch in range(self.num_patches):
            x_start = numpy.random.randint(image_size-self.patch_size)
            y_start = numpy.random.randint(image_size-self.patch_size)
            image_id = numpy.random.randint(batch_size)
            x_slice = slice(x_start, x_start+self.patch_size)
            y_slice = slice(y_start, y_start+self.patch_size)
            patch = x_batch[:, x_slice, y_slice, image_id]
            patches[:, :, :, i_patch] = patch

        return patches

class WeightVisualizer(object):
    def __init__(self, model, model_layer, layer_name, steps=2000):
        self.model_layer = model_layer
        self.name = layer_name
        self.path = model.path

        self.count = 0
        self.steps = steps
    
    def run(self):
        if self.count % self.steps == 0:
            self._save()
        self.count += 1

    def _save(self):
        tt = datetime.now()
        time_string = tt.strftime('%mm-%dd-%Hh-%Mm-%Ss')
        
        W = self.model_layer.W.get_value()
        W -= W.min()
        W /= W.max()
        input_filters, width, height, output_filters = W.shape
        
        tall_bar = numpy.zeros((height, 1))
        output_filter = 0
        row_list = []
        image_list = []

        for output_filter in range(output_filters):
            row_list.append(tall_bar)
            for input_filter in range(input_filters):
                temp = W[input_filter, :, :, output_filter]
                row_list.append(temp)
                row_list.append(tall_bar)
            row_image = numpy.hstack(row_list)
            row_list = []
            long_bar = numpy.zeros((1, row_image.shape[1]))
            image_list.append(long_bar)
            image_list.append(row_image)
        image_list.append(long_bar)
        image_image = numpy.vstack(image_list)
        
        to_save = Image.fromarray(numpy.uint8(255*image_image))
        filename = os.path.join(self.path, '%s-%s.png' % (self.name, time_string))
        to_save.save(filename)

def set_parameters_from_unsupervised_model(model, checkpoint):
    
    f = open(checkpoint, 'rb')
    checkpoint_params = cPickle.load(f)
    f.close()
    
    checkpoint_params_flipped = checkpoint_params[::-1]
    
    model_params = model.all_save_parameters_symbol
    model_params_flipped = model_params[::-1]
    
    for i in range(len(checkpoint_params_flipped)):
        if (list(checkpoint_params_flipped[i].shape) != 
                list(model_params_flipped[i].shape.eval())):
            raise Exception('Size mismatch!')
         
        model_params_flipped[i].set_value(checkpoint_params_flipped[i])
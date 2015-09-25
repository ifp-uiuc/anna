"""Utils for training neural networks.
"""
import os
import Image
from time import time
from datetime import datetime
from copy import deepcopy
import cPickle

import numpy
import skimage.transform
from skimage import color

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs

from anna.layers import layers
from anna.datasets import supervised_dataset


def load_checkpoint(model, checkpoint_path):
    all_parameters = model.all_save_parameters_symbol
    f = open(checkpoint_path, 'rb')
    checkpoint = cPickle.load(f)
    f.close()

    [model_param.set_value(checkpoint_param)
     for model_param, checkpoint_param in zip(all_parameters, checkpoint)]


def save_checkpoint(model, checkpoint_directory_name):
    all_parameters = model.all_save_parameters_symbol
    checkpoint = [param.get_value() for param in all_parameters]
    tt = datetime.now()
    time_string = tt.strftime('%mm-%dd-%Hh-%Mm-%Ss')
    checkpoint_name = '%s-%s.pkl' % (model.name, time_string)
    # print(model.path)
    checkpoint_path = os.path.join(model.path, checkpoint_directory_name,
                                   checkpoint_name)

    print 'Saving model checkpoint to: %s' % checkpoint_path
    f = open(checkpoint_path, 'wb')
    cPickle.dump(checkpoint, f)
    f.close()


def rescale(data):
    data = data / 2.0 * 255.0
    data[data > 255.0] = 255.0
    return data


def color_augment_image(data):
    image = data.transpose(1, 2, 0)
    hsv = color.rgb2hsv(image)

    # Contrast 2
    s_factor1 = numpy.random.uniform(0.25, 4)
    s_factor2 = numpy.random.uniform(0.7, 1.4)
    s_factor3 = numpy.random.uniform(-0.1, 0.1)

    hsv[:, :, 1] = (hsv[:, :, 1] ** s_factor1) * s_factor2 + s_factor3

    v_factor1 = numpy.random.uniform(0.25, 4)
    v_factor2 = numpy.random.uniform(0.7, 1.4)
    v_factor3 = numpy.random.uniform(-0.1, 0.1)

    hsv[:, :, 2] = (hsv[:, :, 2] ** v_factor1) * v_factor2 + v_factor3

    # Color
    h_factor = numpy.random.uniform(-0.1, 0.1)
    hsv[:, :, 0] = hsv[:, :, 0] + h_factor

    hsv[hsv < 0] = 0.0
    hsv[hsv > 1] = 1.0

    rgb = color.hsv2rgb(hsv)

    data_out = rgb.transpose(2, 0, 1)
    return data_out


def gray_augment_image(data):
    image = data.transpose(1, 2, 0)

    v_factor1 = numpy.random.uniform(0.25, 4)
    v_factor2 = numpy.random.uniform(0.7, 1.4)
    v_factor3 = numpy.random.uniform(-0.1, 0.1)

    # print '(v1, v2, v3) = (%f, %f, %f)' % (v_factor1, v_factor2, v_factor3)

    image = (image ** v_factor1) * v_factor2 + v_factor3

    # image[image < 0] = 0.0
    # image[image > 1] = 1.0

    # Rescale to [0, 1] range
    image_min = image.min()
    image -= image_min
    image_max = image.max()
    image /= image_max

    data_out = image.transpose(2, 0, 1)
    return data_out


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
            recon2 = deepcopy(recon) * 1.0

            recon_mask = (numpy.sum(recon == 0.0, axis=2) < 3)
            # recon_mask = 255*(numpy.tile(recon_mask[:, :,None],(1,1,3)))
            recon_mask = 255 * (numpy.tile(recon_mask[:, :, None],
                                           (1, 1, image.shape[2])))

            recon -= image_min
            recon /= image_max
            recon *= 255

            recon2 -= recon2.min()
            recon2 /= recon2.max()
            recon2 *= 255

            image_array = numpy.uint8(numpy.hstack((image, recon, recon2,
                                                    recon_mask)))

            to_save = Image.fromarray(image_array)
            filename = 'recon-%02d.jpeg' % i
            filepath = os.path.join(sub_path, filename)
            to_save.save(filepath)


class NormReconVisualizerGrayscale(object):
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
        for i in range(self.batch.shape[3]):
            image = deepcopy(self.batch[:, :, :, i])
            image = image.transpose(1, 2, 0)
            image_min = image.min()
            image -= image_min
            image_max = image.max()
            image /= image_max
            image *= 255

            recon = numpy.array(prediction[:, :, :, i])
            recon = recon.transpose(1, 2, 0)
            recon2 = deepcopy(recon) * 1.0

            recon_mask = (numpy.sum(recon == 0.0, axis=2) < 3)
            recon_mask = 255 * (numpy.tile(recon_mask[:, :, None],
                                           (1, 1, image.shape[2])))

            recon -= image_min
            recon /= image_max
            recon *= 255

            recon2 -= recon2.min()
            recon2 /= recon2.max()
            recon2 *= 255

            image_array = numpy.uint8(numpy.hstack((image, recon, recon2,
                                                    recon_mask)))
            # Needed for grayscale images. If color, has no effect.
            image_array = numpy.tile(image_array, (1, 1, 3))

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
        W = W.transpose(1, 2, 0, 3)

        row_list = []
        img_list = []
        k = 0

        rows = W.shape[3] / 16
        bar = 0.5 * numpy.ones((W.shape[0], 1, 3))
        for i in range(rows):
            row_list.append(bar)
            for j in range(16):
                W0 = W[:, :, :, k]
                W0 -= W0.min()
                W0 /= W0.max()

                # W0[:, :,0] -= W0[:, :,0].min()
                # W0[:, :,0] /= W0[:, :,0].max()
                # W0[:, :,1] -= W0[:, :,1].min()
                # W0[:, :,1] /= W0[:, :,1].max()
                # W0[:, :,2] -= W0[:, :,2].min()
                # W0[:, :,2] /= W0[:, :,2].max()
                row_list.append(W0)
                row_list.append(bar)
                k += 1
            row_image = numpy.hstack(row_list)
            row_list = []
            bar_h = 0.5 * numpy.ones((1, row_image.shape[1], 3))
            img_list.append(bar_h)
            img_list.append(row_image)
        img_list.append(bar_h)
        img_image = numpy.vstack(img_list)

        to_save = Image.fromarray(numpy.uint8(255 * img_image))
        filename = 'filters_' + time_string + '.png'
        filepath = os.path.join(self.save_path, filename)
        to_save.save(filepath)


class FilterVisualizerGrayscale(object):
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
        W = W.transpose(1, 2, 0, 3)

        row_list = []
        img_list = []
        k = 0

        rows = W.shape[3] / 16
        bar = 0.5 * numpy.ones((W.shape[0], 1, 3))
        for i in range(rows):
            row_list.append(bar)
            for j in range(16):
                W0 = W[:, :, :, k]
                W0 -= W0.min()
                W0 /= W0.max()
                W0 = numpy.tile(W0, (1, 1, 3))

                # W0[:, :,0] -= W0[:, :,0].min()
                # W0[:, :,0] /= W0[:, :,0].max()
                # W0[:, :,1] -= W0[:, :,1].min()
                # W0[:, :,1] /= W0[:, :,1].max()
                # W0[:, :,2] -= W0[:, :,2].min()
                # W0[:, :,2] /= W0[:, :,2].max()
                row_list.append(W0)
                row_list.append(bar)
                k += 1
            row_image = numpy.hstack(row_list)
            row_list = []
            bar_h = 0.5 * numpy.ones((1, row_image.shape[1], 3))
            img_list.append(bar_h)
            img_list.append(row_image)
        img_list.append(bar_h)
        img_image = numpy.vstack(img_list)

        to_save = Image.fromarray(numpy.uint8(255 * img_image))
        filename = 'filters_' + time_string + '.png'
        filepath = os.path.join(self.save_path, filename)
        to_save.save(filepath)


class Monitor(object):
    errors = []
    times = []
    big_errors = []
    big_times = []

    def __init__(self, model,
                 step_number=0,
                 best=1,
                 short_steps=10,
                 long_steps=50,
                 save_steps=2000,
                 test_steps=50,
                 checkpoint_directory='checkpoints'):
        self.step_number = step_number
        self.best = best
        self.short_steps = short_steps
        self.long_steps = long_steps
        self.save_steps = save_steps
        self.model = model
        self.test = False
        self.test_steps = test_steps
        self.checkpoint_directory = checkpoint_directory

        # Check if model.path exists, if not create it
        # (with a checkpoint folder)
        if model.path and not os.path.exists(
                os.path.join(model.path, self.checkpoint_directory)):
            os.makedirs(os.path.join(model.path, self.checkpoint_directory))

    def start(self):
        self.tic = time()

    def stop_test(self, error):
        if self.test:
            self.toc = time()
            _time = self.toc - self.tic
            print '&%d, test error: %.5f, time: %.2f' % (self.step_number,
                                                         error, _time)

    def stop(self, error):
        self.toc = time()
        _time = self.toc - self.tic
        self.errors.append(error)
        self.times.append(_time)
        self.big_errors.append(error)
        self.big_times.append(_time)
        if self.step_number % self.test_steps == 0:
            self.test = True
        else:
            self.test = False
        if self.step_number % self.save_steps == 0:
            save_checkpoint(self.model, self.checkpoint_directory)
        if self.step_number % self.long_steps == 0:
            mean_error = numpy.mean(self.big_errors)
            mean_time = numpy.mean(self.big_times)
            print '*%d, train error: %.5f, time: %.2f' % (self.step_number,
                                                          mean_error,
                                                          mean_time)
            self.big_errors = []
            self.big_times = []
        if self.step_number % self.short_steps == 0:
            mean_error = numpy.mean(self.errors)
            mean_time = numpy.mean(self.times)
            print '%d, train error: %.5f, time: %.2f' % (self.step_number,
                                                         mean_error, mean_time)
            self.errors = []
            self.times = []
        self.step_number += 1


class Normer2(object):
    def __init__(self, filter_size=7, num_channels=3):

        # magic numbers that make things work for stl10
        self.filter_size = filter_size
        self.pad = self.filter_size / 2  # -1
        self.num_channels = num_channels
        self.num_filters = 16
        input = T.ftensor4(name='input')
        filter = T.ftensor4(name='filter')
        gpu_input = gpu_contiguous(input)
        gpu_filter = gpu_contiguous(filter)

        self.conv_func = theano.function([input, filter],
                                         FilterActs(pad=self.pad)(gpu_input,
                                                                  gpu_filter))
        n = self.num_channels * self.filter_size * self.filter_size
        self.w = numpy.float32(numpy.ones((self.num_channels, self.filter_size,
                                           self.filter_size,
                                           self.num_filters))) / n

    def run(self, x_batch):
        mean_batch = self.conv_func(x_batch, self.w)
        mean_batch = numpy.tile(numpy.array(
            mean_batch[0, :, :, :])[None, :, :],
            (self.num_channels, 1, 1, 1))
        diff_batch = x_batch - mean_batch
        std_batch = self.conv_func(diff_batch ** 2, self.w)
        std_batch = numpy.tile(numpy.array(std_batch[0, :, :, :])[None, :, :],
                               (self.num_channels, 1, 1, 1))
        norm_batch = diff_batch / (numpy.array(std_batch) ** (1 / 2))
        return norm_batch


class Normer3(object):
    def __init__(self, filter_size=7, num_channels=3):
        self.filter_size = filter_size
        self.pad = self.filter_size / 2
        self.num_channels = num_channels
        n = self.filter_size*self.filter_size*self.num_channels
        self.w = numpy.float32(numpy.ones(
                               (1, self.num_channels, self.filter_size,
                                self.filter_size))) / n

        input = T.ftensor4(name='input')
        filter = T.ftensor4(name='filter')
        gpu_input = gpu_contiguous(input)
        gpu_filter = gpu_contiguous(filter)
        self.conv_func = theano.function([input, filter],
                                         theano.sandbox.cuda.dnn.dnn_conv(
                                         gpu_input,
                                         gpu_filter,
                                         border_mode=(self.pad, self.pad)))

    def run(self, x_batch):
        mean_batch = self.conv_func(x_batch, self.w)
        mean_batch = numpy.tile(mean_batch, (1, self.num_channels, 1, 1))
        diff_batch = x_batch - mean_batch
        std_batch = self.conv_func((diff_batch)**2, self.w)
        std_batch = numpy.tile(std_batch, (1, self.num_channels, 1, 1))
        norm_batch = diff_batch / (numpy.array(std_batch) ** (1 / 2))
        return norm_batch


class PatchGrabber(object):
    def __init__(self, num_patches, patch_size, num_channels=3):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_channels = num_channels

    def run(self, x_batch):
        image_size = x_batch.shape[1]
        batch_size = x_batch.shape[-1]

        patches = numpy.zeros((self.num_channels, self.patch_size,
                               self.patch_size, self.num_patches),
                              dtype=numpy.float32)

        for i_patch in range(self.num_patches):
            x_start = numpy.random.randint(image_size - self.patch_size)
            y_start = numpy.random.randint(image_size - self.patch_size)
            image_id = numpy.random.randint(batch_size)
            x_slice = slice(x_start, x_start + self.patch_size)
            y_slice = slice(y_start, y_start + self.patch_size)
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

        to_save = Image.fromarray(numpy.uint8(255 * image_image))
        filename = os.path.join(self.path, '%s-%s.png' % (self.name,
                                                          time_string))
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


class Evaluator(object):
    def __init__(self, model, data_container, checkpoint,
                 preprocessor_module_list):
        self.model = model
        self.data_container = data_container
        self.checkpoint = checkpoint
        self.preprocessor = Preprocessor(preprocessor_module_list)
        self.batch_size = model.batch

        # Load parameters from checkpoint
        load_checkpoint(self.model, self.checkpoint)
        self._switch_off_dropout_flags()

    def run(self):
        predictions = self._get_predictions()

        # Compute accuracy
        accuracy = (100.0 * numpy.sum(predictions == self.data_container.y)
                    ) / len(self.data_container.y)

        return accuracy

    def set_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint
        load_checkpoint(self.model, self.checkpoint)
        self._switch_off_dropout_flags()

    def set_preprocessor(self, preprocessor_module_list):
        self.preprocessor = Preprocessor(preproessor_module_list)

    def _switch_off_dropout_flags(self):
        # Switch off dropout flag (if present) in every layer
        all_layers = layers.all_layers(self.model.output)
        for layer in all_layers:
            if hasattr(layer, 'dropout'):
                layer.dropout = 0.0
        # Re-compile the model
        self.model._compile()

    def _get_iterator(self):
        dataset = supervised_dataset.SupervisedDataset(self.data_container.X,
                                                       self.data_container.y)
        iterator = dataset.iterator(mode='sequential',
                                    batch_size=self.batch_size)
        return iterator

    def _get_predictions(self):
        iterator = self._get_iterator()

        # Compute accuracy on each training batch
        predictions = []
        for x_batch, y_batch in iterator:
            x_batch = self.preprocessor.run(x_batch)
            batch_pred = self.model.prediction(x_batch)
            batch_pred = numpy.argmax(batch_pred, axis=1)
            predictions.append(batch_pred)

        # Classify last training batch
        num_samples, num_channels, height, width = self.data_container.X.shape
        last_batch_start_ind = numpy.floor(num_samples /
                                           self.batch_size) * self.batch_size
        last_batch_start_ind = int(last_batch_start_ind)
        last_batch = self.data_container.X[last_batch_start_ind:, :, :, :]

        dummy_batch = numpy.zeros((self.batch_size, num_channels, height,
                                   width),
                                  dtype=numpy.float32)
        dummy_batch[0:last_batch.shape[0], :, :, :] = last_batch
        dummy_batch = self.preprocessor.run(dummy_batch)
        batch_pred = self.model.prediction(dummy_batch)
        batch_pred = batch_pred[0:last_batch.shape[0], :]
        batch_pred = numpy.argmax(batch_pred, axis=1)
        predictions.append(batch_pred)

        # Get all predictions
        predictions = numpy.hstack(predictions)
        # print predictions.shape

        return predictions


class Preprocessor(object):
    def __init__(self, module_list):
        self.module_list = module_list

    def run(self, batch):
        for module in self.module_list:
            batch = module.run(batch)
        return batch


class DataAugmenter(object):
    def __init__(self, amount_pad, window_shape,
                 flip=True,
                 color_on=False,
                 gray_on=False):
        self.amount_pad = amount_pad
        self.window_shape = window_shape
        self.flip = flip
        self.color_on = color_on
        self.gray_on = gray_on
        if len(window_shape) != 2:
            raise ValueError("window_shape should be length 2")

    def run(self, x_batch):
        pad_seq = ((0, 0), (self.amount_pad, self.amount_pad),
                   (self.amount_pad, self.amount_pad), (0, 0))
        x_batch_pad = numpy.pad(x_batch, pad_seq, mode='constant')
        x_batch_pad_aug = self._random_window_and_flip(x_batch_pad)
        if self.color_on:
            x_batch_out = self._color_augment(x_batch_pad_aug)
        elif self.gray_on:
            x_batch_out = self._gray_augment(x_batch_pad_aug)
        else:
            x_batch_out = x_batch_pad_aug
        return x_batch_out

    def _random_window_and_flip(self, x_batch_pad):
        num_channels, _, _, num_images = x_batch_pad.shape
        crop_batch_shape = (num_channels, self.window_shape[0],
                            self.window_shape[1], num_images)
        x_batch_crop = numpy.empty(crop_batch_shape,
                                   dtype=x_batch_pad.dtype)

        for i in range(num_images):
            sample = x_batch_pad[:, :, :, i]

            if self.flip:
                flip_rv = numpy.random.randint(0, 2)
                if flip_rv == 1:
                    sample = sample[:, :, ::-1]

            width_start = numpy.random.randint(0, self.amount_pad)
            height_start = numpy.random.randint(0, self.amount_pad)
            sample = sample[:,
                            height_start:(height_start+self.window_shape[0]),
                            width_start:(width_start+self.window_shape[1])]
            x_batch_crop[:, :, :, i] = sample

        return x_batch_crop

    def _color_augment(self, x_batch):
        out_batch = numpy.zeros(x_batch.shape, dtype=x_batch.dtype)
        __, __, __, num_samples = x_batch.shape

        for i in range(num_samples):
            out_batch[:, :, :, i] = color_augment_image(x_batch[:, :, :, i])

        out_batch *= 2
        return out_batch

    def _gray_augment(self, x_batch):
        out_batch = numpy.zeros(x_batch.shape, dtype=x_batch.dtype)
        __, __, __, num_samples = x_batch.shape

        for i in range(num_samples):
            out_batch[:, :, :, i] = gray_augment_image(x_batch[:, :, :, i])

        out_batch *= 2
        return out_batch


class Crop(object):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # Get input keypoints (only need center)
        input_width, input_height = self.input_size
        self.input_center = numpy.array([input_width / 2, input_height / 2, 1])

        # Get output keypoints
        output_width, output_height = self.output_size
        self.corner_1 = numpy.array([0, 0, 1])
        self.corner_2 = numpy.array([0, output_height, 1])
        self.corner_3 = numpy.array([output_width, 0, 1])
        self.corner_4 = numpy.array([output_width, output_height, 1])
        self.center = numpy.array([output_width / 2, output_height / 2, 1])

        self.transform = skimage.transform.AffineTransform(scale=(1.0, 1.0))

    def get(self, image):
        """Takes an image as an ndarray, and returns a cropped image as an
        ndarray of dtype float32"""

        num_channels = image.shape[0]

        current_corner_1 = numpy.dot(self.transform.params, self.corner_1)
        current_corner_2 = numpy.dot(self.transform.params, self.corner_2)
        current_corner_3 = numpy.dot(self.transform.params, self.corner_3)
        current_corner_4 = numpy.dot(self.transform.params, self.corner_4)
        current_center = numpy.dot(self.transform.params, self.center)

        matrix = self.transform.params

        output = numpy.empty(
            (num_channels, self.output_size[0], self.output_size[1]),
            dtype=numpy.float32)

        for channel in range(num_channels):
            output[channel, :, :] = skimage.transform._warps_cy._warp_fast(
                image=image[channel, :, :],
                H=matrix,
                output_shape=self.output_size)
        return numpy.float32(output)

    def scale(self, scale):
        self.transform += skimage.transform.AffineTransform(
            scale=(scale, scale))

    def rotate(self, angle):
        self.transform += skimage.transform.AffineTransform(
            rotation=numpy.deg2rad(angle))

    def translate(self, x, y):
        self.transform += skimage.transform.AffineTransform(translation=(x, y))

    def centered(self):
        current_center = numpy.dot(self.transform.params, self.center)
        shift = self.input_center - current_center
        self.transform += skimage.transform.AffineTransform(
            translation=shift[0:2])

    def show(self, image):
        current_corner_1 = numpy.dot(self.transform.params, self.corner_1)
        current_corner_2 = numpy.dot(self.transform.params, self.corner_2)
        current_corner_3 = numpy.dot(self.transform.params, self.corner_3)
        current_corner_4 = numpy.dot(self.transform.params, self.corner_4)
        current_center = numpy.dot(self.transform.params, self.center)

        pyplot.imshow(image)
        pyplot.plot(current_corner_1[0], current_corner_1[1], 'r.')
        pyplot.plot(current_corner_2[0], current_corner_2[1], 'r.')
        pyplot.plot(current_corner_3[0], current_corner_3[1], 'r.')
        pyplot.plot(current_corner_4[0], current_corner_4[1], 'r.')
        pyplot.plot(current_center[0], current_center[1], 'b.')
        pyplot.show()


class DataAugmenter2(object):
    def __init__(self, crop_shape, flip=True, scale=True, rotate=True,
                 color_on=False, gray_on=False, kernel='cudnn'):
        """"""
        self.crop_shape = crop_shape
        self.flip = flip
        self.scale = scale
        self.rotate = rotate
        self.color_on = color_on
        self.gray_on = gray_on
        self.kernel = kernel
        if len(crop_shape) != 2:
            raise ValueError("window_shape should be length 2")
        if kernel != 'cudnn' and kernel != 'cuda_convnet':
            raise ValueError("kernel must be cudnn or cuda_convnet")

    def run(self, batch):
        """Applies random crops to each image in a batch.

        Args:
          batch: 4D ndarray with shape (batch_size, channels, width, height)

        Returns:
          batch_out: 4D ndarray with shape (batch_size, channels,
          crop_shape[0], crop_shape[1])
        """

        if self.kernel == 'cuda_convnet':
            # Transpose to cudnn batch shape (to be switch back later)
            batch = batch.transpose(3, 0, 1, 2)

        batch_size, channels, width, height = batch.shape
        out_shape = (batch_size, channels,
                     self.crop_shape[0], self.crop_shape[1])
        batch_out = numpy.empty(out_shape, dtype=numpy.float32)

        for sample_index in range(batch_size):
            sample = batch[sample_index, :, :, :]

            if self.rotate:
                angle = (numpy.random.rand() - 0.5) * 10
            else:
                angle = 0.0

            if self.scale:
                scale = numpy.random.rand() * 0.7 + 0.7
            else:
                scale = 1.0

            diff = (width-scale*self.crop_shape[0])
            translation_x = numpy.random.rand() * diff - diff / 2
            translation_y = numpy.random.rand() * diff - diff / 2

            if self.flip:
                flip_rv = numpy.random.randint(0, 2)
                if flip_rv == 1:
                    sample = sample[:, :, ::-1]

            crop = Crop((width, height), self.crop_shape)
            crop.rotate(angle)
            crop.scale(scale)
            crop.centered()
            crop.translate(translation_x, translation_y)
            output = crop.get(sample)

            batch_out[sample_index, :, :, :] = output

        if self.color_on:
            x_batch_out = self._color_augment(batch_out)
        elif self.gray_on:
            x_batch_out = self._gray_augment(batch_out)
        else:
            x_batch_out = batch_out

        if self.kernel == 'cuda_convnet':
            x_batch_out = x_batch_out.transpose(1, 2, 3, 0)

        return x_batch_out

    def _color_augment(self, x_batch):
        out_batch = numpy.zeros(x_batch.shape, dtype=x_batch.dtype)
        num_samples, __, __, __ = x_batch.shape

        for i in range(num_samples):
            out_batch[i, :, :, :] = color_augment_image(x_batch[i, :, :, :])

        out_batch *= 2
        return out_batch

    def _gray_augment(self, x_batch):
        out_batch = numpy.zeros(x_batch.shape, dtype=x_batch.dtype)
        num_samples, __, __, __ = x_batch.shape

        for i in range(num_samples):
            out_batch[i, :, :, :] = gray_augment_image(x_batch[i, :, :, :])

        out_batch *= 2
        return out_batch


class Annealer(object):
    def __init__(self, model, steps_per_epoch, func=None):
        self.model = model
        self.steps_per_epoch = steps_per_epoch
        if func is None:
            self.func = self.exp_decay
        else:
            self.func = func

        self.step_count = 0
        self.epoch_count = 0
        self.init_learning_rate = self.model.learning_rate_symbol.get_value()
        self.current_learning_rate = self.init_learning_rate

    def run(self):
        self.step_count += 1
        if (self.step_count % self.steps_per_epoch) == 0:
            self.epoch_count += 1
            # Compute new learning rate
            new_learning_rate = self.func(self.init_learning_rate,
                                          self.epoch_count)
            # Set model's learning rate to new learning rate
            self.set_learning_rate(self.model, new_learning_rate)

    def set_learning_rate(self, model, new_learning_rate):
        print 'Learning rate before: ', model.learning_rate_symbol.get_value()
        model.learning_rate_symbol.set_value(numpy.float32(new_learning_rate))
        self.current_learning_rate = new_learning_rate
        print 'Learning rate now: ', model.learning_rate_symbol.get_value()

    def get_current_learning_rate(self):
        return self.current_learning_rate

    def exp_decay(self, init_learning_rate, epoch_count):
        return init_learning_rate * (0.1)**(epoch_count)

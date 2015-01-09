import os
from copy import deepcopy
import cPickle
import argparse

import Image
import numpy
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from anna import util
from anna.datasets import unsupervised_dataset
# from model import Model
# from model_2layer import Model
from model import UnsupervisedModel
from anna.layers import layers


class ZeilerPlotter(object):
    def __init__(self, model, model_layer, pool_layer, dataset, normer):
        self.model = model
        self.W = model_layer.W.get_value()
        self.model_layer = model_layer
        self.dataset = dataset
        self.normer = normer
        max_act_func = theano.function([model.input.output()],
                                       T.max(pool_layer.output(), axis=(1, 2)))
        self.max_act_func = max_act_func

    def run(self, image_index, filter_index):
        image = self.dataset.X[image_index, :, :, :]
        batch = numpy.tile(image[:, :, :, None], (1, 1, 1, 128))
        batch = self.normer.run(batch)

        W_masked = numpy.zeros(self.W.shape, dtype=numpy.float32)
        W_masked[:, :, :, filter_index] = self.W[:, :, :, filter_index]
        self.model_layer.W.set_value(W_masked)

        max_activations = numpy.array(
            self.max_act_func(batch))[filter_index, 0]
        print max_activations

        zeiler_batch = model.prediction(batch)
        # print zeiler_batch.shape
        self.model_layer.W.set_value(self.W)
        return zeiler_batch[:, :, :, 0]


class ZeilerMaxPlotter(ZeilerPlotter):
    def __init__(self, model, model_layer, pool_layer, dataset, normer):
        insert_mask_layer(model, model_layer)
        super(ZeilerMaxPlotter, self).__init__(model,
                                               model_layer,
                                               pool_layer,
                                               dataset,
                                               normer)

    def run(self, image_index, filter_index, max_val):
        model.max_mask_layer.max_val.set_value(max_val)
        return super(ZeilerMaxPlotter, self).run(image_index, filter_index)


class MaxMaskLayer(object):
    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.max_val = theano.shared(0.0)
        self.params = []
        self.trainable = False  # NEW

    def get_output_shape(self):
        return self.input_layer.get_output_shape()

    def output(self, input=None, *args, **kwargs):
        if input is None:
            input = self.input_layer.output(*args, **kwargs)

        return input * T.eq(input, self.max_val)


def get_patches(zeiler_image, max_image):
    nonzeros = numpy.sum(zeiler_image == 0, axis=2)
    row_nonzeros = numpy.where(numpy.sum(nonzeros < 3, axis=1))[0]
    row_start = row_nonzeros[0]
    row_end = row_nonzeros[-1]
    col_nonzeros = numpy.where(numpy.sum(nonzeros < 3, axis=0))[0]
    col_start = col_nonzeros[0]
    col_end = col_nonzeros[-1]
    row_slice = slice(row_start, row_end)
    col_slice = slice(col_start, col_end)

    max_patch = max_image[row_slice, col_slice]
    filter_patch = zeiler_image[row_slice, col_slice]
    return max_patch, filter_patch


def zero_pad(image, size):
    diff = size - numpy.array(image.shape)
    left = diff/2
    right = diff - left
    return numpy.pad(image, zip(left, right), mode='constant')


def make_image_from_list(patches, num_rows, num_cols):
    count = 0
    row_list = []
    image_list = []
    size = [19, 19, 3]
    for row in range(num_rows):
        for col in range(num_cols):
            row_list.append(zero_pad(patches[count], size))
            count += 1
        row_image = numpy.hstack(row_list)
        row_list = []
        image_list.append(row_image)
    image_image = numpy.vstack(image_list)
    return image_image


def insert_mask_layer(model, model_layer):
    all_layers = layers.all_layers(model.output)
    all_layers = all_layers[0:-1]

    next_layer = [layer for layer in all_layers
                  if layer.input_layer == model_layer][0]

    model.max_mask_layer = MaxMaskLayer(model_layer)
    next_layer.input_layer = model.max_mask_layer

    model._compile()


def get_max_activations_and_images(model, model_layer, dataset, normer):

    max_act_func = theano.function([model.input.output()],
                                   T.max(model_layer.output(), axis=(1, 2)))
    iterator = dataset.iterator(mode='sequential', batch_size=128)
    acts_list = []

    for batch in iterator:
        batch = batch.transpose(1, 2, 3, 0)
        batch = normer.run(batch)
        max_acts_test = max_act_func(batch)
        acts_list.append(max_acts_test)

    acts_array = numpy.hstack(acts_list)
    maximum_activation_values = numpy.max(acts_array, axis=1)
    maximum_image_indices = numpy.argmax(acts_array, axis=1)

    return maximum_activation_values, maximum_image_indices


def get_activations(model, model_layer, dataset, normer):

    max_act_func = theano.function([model.input.output()],
                                   T.max(model_layer.output(), axis=(1, 2)))
    iterator = dataset.iterator(mode='sequential', batch_size=128)
    acts_list = []

    for batch in iterator:
        batch = batch.transpose(1, 2, 3, 0)
        batch = normer.run(batch)
        max_acts_test = max_act_func(batch)
        acts_list.append(max_acts_test)

    acts_array = numpy.hstack(acts_list)
    return acts_array


def plot_top_activation(zeiler_plotter,
                        output_path,
                        best_acts,
                        best_index,
                        num_filters):

    output_path = os.path.join(output_path, 'top1')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Make a top 1 zeiler plot
    maximum_activation_values = numpy.max(acts_array, axis=1)
    maximum_image_indices = numpy.argmax(acts_array, axis=1)

    image_patches = []
    filter_patches = []
    for i in range(num_filters):
        print i
        # num_filters = maximum_activation_values.shape[0]
        # W = model.conv1.W.get_value()

        # for i in range(num_filters):
        # i = numpy.random.randint(num_filters)
        # filter_ = W[:,:,:,i].transpose(1,2,0)
        # filter_ -= filter_.min()
        # filter_ /= filter_.max()

        image_index = maximum_image_indices[i]
        max_val = maximum_activation_values[i]

        max_image = test_dataset.X[image_index, :, :, :]
        batch = numpy.tile(max_image[:, :, :, None], (1, 1, 1, 128))
        max_image = normer.run(batch)[:, :, :, 0].transpose(1, 2, 0)
        max_image -= max_image.min()
        max_image /= max_image.max()

        # zeiler_image = numpy.array(zeiler_plotter.run(image_index, i))
        zeiler_image = numpy.array(zeiler_plotter.run(image_index, i,
                                                      numpy.float32(max_val)))
        zeiler_image = zeiler_image.transpose(1, 2, 0)

        zeiler_image2 = deepcopy(zeiler_image)
        zeiler_image2 -= zeiler_image2.min()
        zeiler_image2 /= zeiler_image2.max()

        # plt.subplot(2,2,1)
        # plt.imshow(max_image)
        # plt.subplot(2,2,2)
        # plt.imshow(0.2 * max_image + 0.9 * zeiler_image2)
        # plt.colorbar()
        # plt.subplot(2,2,3)
        # plt.imshow(zeiler_image2)
        image_patch, filter_patch = get_patches(zeiler_image, max_image)
        # filter_patch -= filter_patch.min()
        # filter_patch /= filter_patch.max()
        image_patches.append(image_patch)
        filter_patches.append(filter_patch)
        # plt.suptitle('Filter: {} -- Image: {} -- Max Value: {}'
        # .format(i, image_index, maximum_activation_values[i]))
        # plt.show()

    image_image = make_image_from_list(image_patches, num_filters/16, 16)
    filter_image = make_image_from_list(filter_patches, num_filters/16, 16)
    filter_image -= filter_image.min()
    filter_image /= filter_image.max()

    to_save = Image.fromarray(numpy.uint8(255*image_image))
    filename = 'all_patches.jpeg'
    filepath = os.path.join(output_path, filename)
    to_save.save(filepath)

    to_save = Image.fromarray(numpy.uint8(255*filter_image))
    filename = 'all_recons.jpeg'
    filepath = os.path.join(output_path, filename)
    to_save.save(filepath)
    # plt.subplot(2,1,1)
    # plt.imshow(image_image)
    # plt.subplot(2,1,2)
    # plt.imshow(filter_image)


def plot_top_25_activations(zeiler_plotter,
                            output_path,
                            best_acts,
                            best_index,
                            num_filters):

    output_path = os.path.join(output_path, 'top25')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Make a top n zeiler plot
    for i in range(num_filters):
        n = 25
        image_patches = []
        filter_patches = []
        top10_acts = best_acts[i, -n::][::-1]
        top10_index = best_index[i, -n::][::-1]

        print i

        for top_n in range(n):
            image_index = top10_index[top_n]
            max_val = top10_acts[top_n]
            print max_val

            # Given an image_index, get an image
            # max_image = test_dataset.X[image_index,:,:,:]
            max_image = zeiler_plotter.dataset.X[image_index, :, :, :]
            batch = numpy.tile(max_image[:, :, :, None], (1, 1, 1, 128))
            # max_image = normer.run(batch)[:,:,:,0].transpose(1,2,0)
            max_image = zeiler_plotter.normer.run(
                batch)[:, :, :, 0].transpose(1, 2, 0)
            max_image -= max_image.min()
            max_image /= max_image.max()

            # Given an image_index, and a max_val, get a recon
            zeiler_image = numpy.array(zeiler_plotter.run(
                image_index, i, numpy.float32(max_val)))
            zeiler_image = zeiler_image.transpose(1, 2, 0)

            zeiler_image2 = deepcopy(zeiler_image)
            zeiler_image2 -= zeiler_image2.min()
            zeiler_image2 /= zeiler_image2.max()

            # Grab the patches
            try:
                image_patch, filter_patch = get_patches(zeiler_image,
                                                        max_image)
            except:
                print 'oops'
                image_patch = numpy.zeros((19, 19, 3))
                filter_patch = numpy.zeros((19, 19, 3))
            filter_patch -= filter_patch.min()
            filter_patch /= filter_patch.max()
            image_patches.append(image_patch)
            filter_patches.append(filter_patch)

        image_image = make_image_from_list(image_patches, 5, 5)
        filter_image = make_image_from_list(filter_patches, 5, 5)
        filter_image -= filter_image.min()
        filter_image /= filter_image.max()

        to_save = Image.fromarray(numpy.uint8(255*image_image))
        filename = 'image-%02d.jpeg' % i
        filepath = os.path.join(output_path, filename)
        to_save.save(filepath)

        to_save = Image.fromarray(numpy.uint8(255*filter_image))
        filename = 'recon-%02d.jpeg' % i
        filepath = os.path.join(output_path, filename)
        to_save.save(filepath)

if __name__ == "__main__":

    # Note: This script can only visualize the inner-most encoding
    #       layer of the neural network.

    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(
        prog='zeiler_plotter',
        description='Script to make Zeiler plots.')
    parser.add_argument('weight_layer', help='Layer to extract weight matrix.')
    parser.add_argument('feature_layer', help='Layer whose features will be '
                        'masked and backpropogated to input space.')
    parser.add_argument('checkpoint', help='Path to checkpoint File.')
    parser.add_argument('output_path', help='Path to folder to save results.')
    args = parser.parse_args()

    checkpoint = args.checkpoint
    output_path = args.output_path
    weight_layer = args.weight_layer
    feature_layer = args.feature_layer
    print 'Weight Layer: {}'.format(weight_layer)
    print 'Feature Layer: {}'.format(feature_layer)

    model_file_path = os.path.join(os.getcwd(), 'model.py')

    # Check if model.py file exists
    if not os.path.exists(model_file_path):
        raise Exception('model.py file does not exist!')
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint):
        raise Exception('Checkpoint file does not exist!')
    # Check if output directory exists, if not construct it
    if not os.path.exists(output_path):
        print('Output directory does not exist. Creating it now...')
        os.makedirs(output_path)

    print('Loading Model')
    model = UnsupervisedModel('xxx', './')
    print('Loading Checkpoint')
    util.load_checkpoint(model, checkpoint)

    weight_layer = eval('model.' + weight_layer)
    feature_layer = eval('model.' + feature_layer)

    print('Loading Data')
    data = numpy.load('/data/stl10_matlab/unsupervised.npy')
    data = numpy.float32(data)
    data /= 255.0
    data *= 2.0
    train_data = data[0:90000, :, :, :]
    test_data = data[90000::, :, :, :]
    train_dataset = unsupervised_dataset.UnsupervisedDataset(train_data)
    test_dataset = unsupervised_dataset.UnsupervisedDataset(test_data)
    # test_x_batch = test_x_batch.transpose(1, 2, 3, 0)
    normer = util.Normer(filter_size=7)
    print('Done')

    print('Computing top activations for each filter.')
    acts_array = get_activations(model, weight_layer, test_dataset, normer)
    best_acts = numpy.sort(acts_array, axis=1)
    best_index = numpy.argsort(acts_array, axis=1)
    print acts_array.shape
    print numpy.max(acts_array, axis=1)

    num_filters = acts_array.shape[0]
    zeiler_plotter = ZeilerMaxPlotter(model, weight_layer, feature_layer,
                                      test_dataset, normer)

    print('Extracting top 25 activations for each filter.')
    plot_top_25_activations(zeiler_plotter, output_path, best_acts, best_index,
                            num_filters)
    print('Extracting top activation for each filter.')
    plot_top_activation(zeiler_plotter, output_path, best_acts, best_index,
                        num_filters)

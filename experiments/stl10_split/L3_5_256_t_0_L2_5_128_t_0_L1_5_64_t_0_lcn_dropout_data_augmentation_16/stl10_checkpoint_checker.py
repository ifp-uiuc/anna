import sys
import os
import argparse

import numpy

from model import SupervisedModel
from fastor import util
from fastor.datasets import supervised_dataset

def load_stl10_data(train_split):

    # Load STL-10 data
    print 'Loading STL-10 Training Data'
    X_train = numpy.load('/data/stl10_matlab/train_splits/train_X_'+str(train_split)+'.npy')
    y_train = numpy.load('/data/stl10_matlab/train_splits/train_y_'+str(train_split)+'.npy')

    print 'Loading STL-10 Testing Data'
    X_test = numpy.load('/data/stl10_matlab/test_X.npy')
    y_test = numpy.load('/data/stl10_matlab/test_y.npy')

    X_train = numpy.float32(X_train)
    X_train /= 255.0
    X_train *= 2.0

    X_test = numpy.float32(X_test)
    X_test /= 255.0
    X_test *= 2.0

    train_dataset = supervised_dataset.SupervisedDataset(X_train, y_train)
    test_dataset = supervised_dataset.SupervisedDataset(X_test, y_test)
    train_iterator = train_dataset.iterator(mode='sequential', batch_size=128)
    test_iterator = test_dataset.iterator(mode='sequential', batch_size=128)
    
    return train_iterator, test_iterator


def compute_overall_accuracy(model, normer, mode, iterator):

    accuracy_list = []
    # Compute accuracy on each batch
    i = 0
    for x_batch, y_batch in iterator:
        #y_batch = numpy.int64(numpy.argmax(y_batch, axis=1))
        x_batch = x_batch.transpose(1, 2, 3, 0)
        x_batch = normer.run(x_batch)
        batch_accuracy = model.eval(x_batch, y_batch-1)
        print('{} Batch {} Accuracy: {}'.format(mode, i, batch_accuracy))
        accuracy_list.append(batch_accuracy)
        i+=1

    # Compute overall accuracy
    overall_accuracy = numpy.mean(numpy.hstack(accuracy_list))
    print('\nOverall {} Accuracy: {}\n'.format(mode, overall_accuracy))
    return overall_accuracy
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='stl10_checkpoint_checker', description='Script to select best performing checkpoint on STL10.')
    parser.add_argument("-s", "--split", default='0', help='Training split of stl10 to use. (0-9)')
    parser.add_argument("checkpoint_dir", help='Folder containing all .pkl checkpoint files.')
    args = parser.parse_args()

    train_split = int(args.split)
    if train_split < 0 or train_split > 9:
        raise Exception('Train split must be in range 0-9.')

    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        raise Exception('Checkpoint directory does not exist.')
    checkpoint_list = sorted(os.listdir(checkpoint_dir))

    model = SupervisedModel('xxx', './')
    model.fc4.dropout = 0.0
    model._compile()
    num_channels = model.conv1.filter_shape[0]
    filter_size = model.conv1.filter_shape[1]
    print 'Using model trained on split '+str(train_split)+'\n'

    
    # Get iterators for stl10 train and test sets
    train_iterator, test_iterator = load_stl10_data(train_split)
    
    # Create object to local contrast normalize a batch.
    # Note: Every batch must be normalized before use.
    normer = util.Normer2(filter_size=filter_size, num_channels=num_channels)
    
    train_accuracies = []
    test_accuracies = []

    for i, checkpoint_file in enumerate(checkpoint_list):
        print 'Loading Checkpoint %s' % checkpoint_file
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        util.load_checkpoint(model, checkpoint_path)
    
        #print 'Compute Train Accuracy'
        #train_accuracies.append(compute_overall_accuracy(model, normer, 'train', train_iterator))
        print 'Compute Test Accuracy'
        test_accuracies.append(compute_overall_accuracy(model, normer, 'test', test_iterator))
        print '\n'
   
        train_iterator.reset()
        test_iterator.reset()     
    #print train_accuracies
    #print test_accuracies

    max_test_accuracy = numpy.max(test_accuracies)
    max_index = numpy.argmax(test_accuracies)

    max_checkpoint = checkpoint_list[max_index]

    print 'Max Test Accuracy: %.2f' % (max_test_accuracy*100)
    print 'Max Checkpoint: %s' % max_checkpoint

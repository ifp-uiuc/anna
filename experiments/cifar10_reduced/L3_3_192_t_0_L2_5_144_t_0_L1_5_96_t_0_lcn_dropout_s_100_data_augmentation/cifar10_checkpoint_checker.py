import sys
import os

import numpy

from model import SupervisedModel
from fastor import util

def compute_overall_accuracy(model, normer, which_set=None):

    if not which_set:
        print('Please specify train or test set of cifar10')
        sys.exit(0)

    if which_set == 'train':
        #print('Training!!')
        num_samples = 50000
    elif which_set == 'test':
        #print('Testing!!')
        num_samples = 10000
    else:
        print('Invalid mode. mode = train or mode = test')
        sys.exit(0)


    iterator = util.get_cifar_iterator_reduced(which_set, 
                                              mode='even_sequential', 
                                              batch_size=128,                                       
                                              rescale=True,
                                              num_samples_per_class=100,
                                              which_split=0)

    accuracy = numpy.zeros((num_samples/128))
    i = 0
    for x_batch, y_batch in iterator: 
        x_batch = normer.run(x_batch)               
        y_batch = numpy.int64(numpy.argmax(y_batch, axis=1))
        accuracy[i] = model.eval(x_batch, y_batch)
        #print('Mini-batch {}: {}'.format(i+1,accuracy[i]))
        i+=1

    print('Overall {} Accuracy: {}'.format(mode.upper(), numpy.mean(accuracy)))
    return numpy.mean(accuracy)


if __name__ == "__main__":
    checkpoint_dir = sys.argv[1]
    if not os.path.exists(checkpoint_dir):
        raise Exception('Checkpoint directory does not exist.')
    checkpoint_list = sorted(os.listdir(checkpoint_dir))

    model = SupervisedModel('xxx', './')
    num_channels = model.conv1.filter_shape[0]
    filter_size = model.conv1.filter_shape[1]
    normer = util.Normer2(filter_size=filter_size, num_channels=num_channels)
    #train_accuracies = []
    test_accuracies = []

    for i, checkpoint_file in enumerate(checkpoint_list):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        util.load_checkpoint(model, checkpoint_path)
        
        print 'Checkpoint %s' % checkpoint_file
        #print 'Compute Train Accuracy'
        #train_accuracies.append(compute_overall_accuracy(model, normer, 'train'))
        print 'Compute Test Accuracy'
        test_accuracies.append(compute_overall_accuracy(model, normer, 'test'))
        print '\n'
        
    #print train_accuracies
    #print test_accuracies

    max_test_accuracy = numpy.max(test_accuracies)
    max_index = numpy.argmax(test_accuracies)

    max_checkpoint = checkpoint_list[max_index]

    print 'Max Test Accuracy: %.2f' % (max_test_accuracy*100)
    print 'Max Checkpoint: %s' % max_checkpoint

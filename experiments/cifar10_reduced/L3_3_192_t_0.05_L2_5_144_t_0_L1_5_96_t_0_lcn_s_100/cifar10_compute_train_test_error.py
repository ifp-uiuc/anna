import sys
import numpy

from model import SupervisedModel
from fastor import util



def compute_overall_accuracy(model, normer, mode=None):

    if not mode:
        print('Please specify train or test set of cifar10')
        sys.exit(0)

    if mode == 'train':
        print('Training!!')
        num_samples = 50000
    elif mode == 'test':
        print('Testing!!')
        num_samples = 10000
    else:
        print('Invalid mode. mode = train or mode = test')
        sys.exit(0)


    iterator = util.get_cifar_iterator_reduced(mode, 
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
        print('Mini-batch {}: {}'.format(i+1,accuracy[i]))
        i+=1

    print('Overall {} Accuracy: {}'.format(mode.upper(), numpy.mean(accuracy)))

    
if __name__ == "__main__":

    print('Loading Model Checkpoint')
    model = SupervisedModel('xxx', './')
    checkpoint = sys.argv[1]
    util.load_checkpoint(model, checkpoint)
    normer = util.Normer2(filter_size=5, num_channels=3)

    print('Computing Accuracies')
    compute_overall_accuracy(model, normer, 'train')
    compute_overall_accuracy(model, normer, 'test')

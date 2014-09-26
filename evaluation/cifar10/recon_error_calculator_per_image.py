import sys
import numpy

from masci_model import Model
#from model import Model
from fastor import util

from matplotlib import pyplot


def compute_recon_error_per_image(model, mode=None):

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


    iterator = util.get_cifar_iterator(mode, 
                                      mode='even_sequential', 
                                      batch_size=128,                                       
                                      rescale=True)

    #recon_error = numpy.zeros((num_samples/128))
    recon_list = []
    i = 0
    for x_batch, y_batch in iterator:                
        y_batch = numpy.int64(numpy.argmax(y_batch, axis=1))
        #recon_error[i] = model.eval(x_batch/2)
        image_costs = model.eval_images(x_batch/2)
        batch_costs = numpy.mean(image_costs)
        recon_list.append(image_costs)
        print('Mini-batch {}: {}'.format(i+1,batch_costs))
        i+=1

    recons = numpy.hstack(recon_list)
    recons_sorted = numpy.sort(recons)
    pyplot.plot(recons_sorted)
    pyplot.show()
    # print('Overall {} Recon Error: {}'.format(mode.upper(), numpy.mean(recon_error)))

    
if __name__ == "__main__":

    print('Loading Model Checkpoint')
    model = Model('xxx', './')
    checkpoint = sys.argv[1]
    util.load_checkpoint(model, checkpoint)

    print('Computing Recon Error')
    compute_overall_recon_error_per_image(model, 'train')
    compute_overall_recon_error_per_image(model, 'test')
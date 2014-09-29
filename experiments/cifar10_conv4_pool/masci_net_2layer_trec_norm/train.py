import sys

import numpy
import theano

from fastor import util

from model import Model

def orthogonalize(w):
    # Orthogonalize square matrices.
    # Or left orthogonalize overcomplete matrices.
    # Simply gets an SVD decomposition, and sets the singular values to ones.
    dim2, dim1 = w.shape
    dim = numpy.min((dim1, dim2))
    u, s, v = numpy.linalg.svd(w)
    S = numpy.zeros((dim2,dim1))
    s = s/s
    S[:dim,:dim] = numpy.diag(s)
    w = numpy.dot(u,numpy.dot(S,v))
    w = numpy.float32(w)
    return w

def conv_orthogonalize(w, k=1.0):
    # Reshape filters into a matrix
    channels, width, height, filters = w.shape
    w = w.reshape(channels*width*height, filters).transpose(1,0)

    # Orthogonalize the matrix
    w = orthogonalize(w)

    # Contruct 2D hamming window
    hamming1 = numpy.hamming(width)
    hamming2 = numpy.hamming(height)
    hamming = numpy.outer(hamming1, hamming2)

    # Use it to mask the input to w
    mask = numpy.tile(hamming[None,:,:], (channels,1,1))
    mask = mask.reshape(channels*width*height)*k
    m = numpy.diag(mask)
    w = numpy.dot(w, m)

    # Reshape the matrix into filters
    w = w.transpose(1,0)
    w =  w.reshape(channels, width, height, filters)
    w = numpy.float32(w)
    return w


print('Start')
model = Model(sys.argv[1], sys.argv[2])
#model = Model('gah', '/home/paine1/Desktop/')
#checkpoint = '/experiments/cifar_conv4_pool_relu_right/checkpoints/cifar_conv4_pool_relu_right-09m-23d-00h-37m-40s.pkl'
#util.load_checkpoint(model, checkpoint)
monitor = util.Monitor(model)

print('Loading Data')
train_iterator = util.get_cifar_iterator('train', 
                                    mode='random_uniform', 
                                    batch_size=128, 
                                    num_batches=100000,                                     
                                    rescale=True)

test_iterator = util.get_cifar_iterator('test', 
                                    mode='sequential', 
                                    batch_size=128,                                      
                                    rescale=True)

filter_visualizer = util.FilterVisualizer(model, steps=40)
normer = util.Normer(filter_size=5)

# Grab some patches to initialize weights
patch_grabber = util.PatchGrabber(96, 5)
x_batch, y_batch = train_iterator.next()
x_batch = normer.run(x_batch)
patches = patch_grabber.run(x_batch) * 0.05
#patches = conv_orthogonalize(patches)
W = model.conv1.W.get_value()
print W.shape
print patches.shape
model.conv1.W.set_value(patches)
W2 = model.conv2.W.get_value()
W2 = conv_orthogonalize(W2) * 7.5
model.conv2.W.set_value(W2)
#patch_grabber_pool1 = util.PatchGrabber(144, 5, num_channels=96)
#pool1_func = theano.function([model.input.output()], model.pool1.output())
#x_batch, y_batch = train_iterator.next()
#x_batch = normer.run(x_batch)
#batch_pool1 = pool1_func(x_batch)
#patches_pool1 = patch_grabber_pool1.run(batch_pool1) * 0.01
#model.conv2.W.set_value(patches_pool1)
#conv2_func = theano.function([model.input.output()], model.conv2.output())
#conv1_func = theano.function([model.input.output()], model.conv1.output())


# Grad test data to give to NormReconVisualizer
test_x_batch, test_y_batch = test_iterator.next()
test_x_batch = normer.run(test_x_batch)
recon_visualizer = util.NormReconVisualizer(model, test_x_batch, steps=50)

#model.learning_rate_symbol.set_value(0.000005/10)
print('Trainig Model')
count = 0
for x_batch, y_batch in train_iterator:        
    monitor.start()
    x_batch = normer.run(x_batch)
    error = model.train(x_batch)
    monitor.stop(error) 
    recon_visualizer.run()   
    filter_visualizer.run()
    if count*1.0 / 100000 == 1:
        print 'increasing learning rate...'
        model.learning_rate_symbol.set_value(0.001)
    count += 1
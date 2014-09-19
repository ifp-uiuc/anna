import numpy
from fastor import util

import pylearn2.datasets.cifar10 as cifar10

def orthogonalize(w):
    # Orthogonalize square matrices.
    # Or left orthogonalize overcomplete matrices.
    # Simply gets an SVD decomposition, and sets the singular values to ones.
    dim2, dim1 = w.shape
    short = numpy.min((dim2, dim1))
    u, s, v = numpy.linalg.svd(w)
    S = numpy.zeros((dim2,dim1))
    s = s/s
    S[:short,:short] = numpy.diag(s)
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


model = Model('deconv-stl10-32x64', '/experiments/deconv/stl10-32x64-ortho')
#util.load_checkpoint(model, checkpoint)
monitor = util.Monitor(model)
print('Done Making Monitor')
# w1 = model.conv1.W.get_value()
# w2 = model.conv2.W.get_value()

k = 1.05

# w1 = conv_orthogonalize(w1,k=k)
# w2 = conv_orthogonalize(w2,k=k)
# # Initialize the layers with orthogonal weights
# model.conv1.W.set_value(w1)
# model.conv2.W.set_value(w2)
# model.deconv3.W.set_value(w2)
# model.output.W.set_value(w1)
print('1')
batch = test_iterator.next()
batch = batch.transpose(1,2,3,0)
batch_hat = model.prediction(batch)
print 'k: %0.2f, error: %0.4f' % (k, model.eval(batch))

alpha = numpy.float32(1.0)
model.conv1.alpha.set_value(alpha)
model.conv2.alpha.set_value(alpha)
model.deconv3.alpha.set_value(alpha)
model.output.alpha.set_value(alpha)

print('2')
model.learning_rate_symbol.set_value(0.01)
count = 1
for batch in train_iterator:
    # if count % 100 == 0:
    #     alpha += numpy.float32(0.1)
    #     model.conv1.alpha.set_value(alpha)
    #     model.conv2.alpha.set_value(alpha)
    #     model.deconv3.alpha.set_value(alpha)
    #     model.output.alpha.set_value(alpha)
    #     print 'alpha set to %0.3f' % alpha
    batch = batch.transpose(1,2,3,0)
    monitor.start()
    error = model.train(batch)
    monitor.stop(error)
    count += 1
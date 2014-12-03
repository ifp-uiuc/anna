import numpy

from fastor.layers import layers, cc_layers
import fastor.models

class UnsupervisedModel(fastor.models.UnsupervisedModel):        
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 96, 96)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(5*5*3) # was = 0.25   
    winit2 = k/numpy.sqrt(5*5*64)
    binit = 0.0
    
    def trec(x):
      return x*(x > 0.0)

    nonlinearity = trec

    conv1 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        input, 
        n_filters=64,
        filter_size=5,
        weights_std=winit1,
        nonlinearity=nonlinearity,
        pad=2)
    pool1 = cc_layers.CudaConvnetPooling2DLayer(conv1, 2, stride=2)
    conv2 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        pool1, 
        n_filters=128,
        filter_size=5,
        weights_std=winit2,
        nonlinearity=nonlinearity,
        pad=2)
    pool2 = cc_layers.CudaConvnetPooling2DLayer(conv2, 2, stride=2)
    unpool3 = cc_layers.CudaConvnetUnpooling2DLayer(pool2, pool2)
    deconv3 = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        unpool3, conv2, nonlinearity=layers.identity)
    unpool4 = cc_layers.CudaConvnetUnpooling2DLayer(deconv3, pool1)
    output = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        unpool4, conv1, nonlinearity=layers.identity)    


class SupervisedModel(fastor.models.SupervisedModel):    
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 96, 96)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(5*5*3) # was = 0.25  
    winit2 = k/numpy.sqrt(5*5*96)
    binit = 0.0
    
    def trec(x):
      return x*(x > 0.0)

    nonlinearity = trec

    conv1 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        input, 
        n_filters=64,
        filter_size=5,
        weights_std=winit1,
        nonlinearity=nonlinearity,
        pad=2)
    pool1 = cc_layers.CudaConvnetPooling2DLayer(conv1, 2, stride=2)
    conv2 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        pool1, 
        n_filters=128,
        filter_size=5,
        weights_std=winit2,
        nonlinearity=nonlinearity,
        pad=2)
    pool2 = cc_layers.CudaConvnetPooling2DLayer(conv2, 2, stride=2)

    winitD1 = k/numpy.sqrt(numpy.prod(pool2.get_output_shape()))
    winitD2 = k/numpy.sqrt(512)

    pool2_shuffle = cc_layers.ShuffleC01BToBC01Layer(pool2)    
    fc3 = layers.DenseLayer(
        pool2_shuffle,
        n_outputs = 512,
        weights_std=winitD1,
        init_bias_value=1.0,
        nonlinearity=layers.rectify,
        dropout=0.0)
    output = layers.DenseLayer(
        fc3,
        n_outputs=10,
        weights_std=winitD2,
        init_bias_value=0.0,
        nonlinearity=layers.softmax)

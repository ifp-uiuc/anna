import numpy

from fastor.layers import layers, cc_layers
import fastor.models

class UnsupervisedModel(fastor.models.UnsupervisedModel):        
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 32, 32)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(5*5*3) # was = 0.25 
    winit2 = k/numpy.sqrt(5*5*96)  
    winit3 = k/numpy.sqrt(5*5*144)  
    binit = 0.0
    
    def trec0(x):
      return x*(x > 0.0)

    def trec05(x):
      return x*(x > 0.05)

    nonlinearity0 = trec0
    nonlinearity05 = trec05

    conv1 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        input, 
        n_filters=96,
        filter_size=5,
        weights_std=winit1,
        nonlinearity=nonlinearity0,
        pad=2)
    pool1 = cc_layers.CudaConvnetPooling2DLayer(conv1, 2, stride=2)
    conv2 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        pool1, 
        n_filters=144,
        filter_size=5,
        weights_std=winit2,
        nonlinearity=nonlinearity0,
        pad=2)
    pool2 = cc_layers.CudaConvnetPooling2DLayer(conv2, 2, stride=2)    
    conv3 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        pool2, 
        n_filters=192,
        filter_size=3,
        weights_std=winit3,
        nonlinearity=nonlinearity05,
        pad=1)    
    deconv3 = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        conv3, conv3, nonlinearity=layers.identity)
    unpool4 = cc_layers.CudaConvnetUnpooling2DLayer(deconv3, pool2)
    deconv4  = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        unpool4, conv2, nonlinearity=layers.identity)
    unpool5 = cc_layers.CudaConvnetUnpooling2DLayer(deconv4, pool1)
    output = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        unpool5, conv1, nonlinearity=layers.identity)    


class SupervisedModel(fastor.models.SupervisedModel):    
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 32, 32)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(5*5*3) # was = 0.25  
    winit2 = k/numpy.sqrt(5*5*96)  
    winit3 = k/numpy.sqrt(5*5*144)  
    binit = 0.0
    
    def trec0(x):
      return x*(x > 0.0)

    def trec05(x):
      return x*(x > 0.05)

    nonlinearity0 = trec0
    nonlinearity05 = trec05

    conv1 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        input, 
        n_filters=96,
        filter_size=5,
        weights_std=winit1,
        nonlinearity=nonlinearity0,
        pad=2)
    pool1 = cc_layers.CudaConvnetPooling2DLayer(conv1, 2, stride=2)
    conv2 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        pool1, 
        n_filters=144,
        filter_size=5,
        weights_std=winit2,
        nonlinearity=nonlinearity0,
        pad=2)
    pool2 = cc_layers.CudaConvnetPooling2DLayer(conv2, 2, stride=2)
    conv3 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        pool2, 
        n_filters=192,
        filter_size=3,
        weights_std=winit3,
        nonlinearity=nonlinearity0,
        pad=1)   

    winitD1 = k/numpy.sqrt(numpy.prod(conv3.get_output_shape()))
    winitD2 = k/numpy.sqrt(300)

    conv3_shuffle = cc_layers.ShuffleC01BToBC01Layer(conv3)    
    fc4 = layers.DenseLayer(
        conv3_shuffle,
        n_outputs = 300,
        weights_std=winitD1,
        init_bias_value=1.0,
        nonlinearity=layers.rectify,
        dropout=0.0)
    output = layers.DenseLayer(
        fc4,
        n_outputs=10,
        weights_std=winitD2,
        init_bias_value=0.0,
        nonlinearity=layers.softmax)

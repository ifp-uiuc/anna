import numpy

from fastor.layers import layers, cc_layers
import fastor.models

class UnsupervisedModel(fastor.models.UnsupervisedModel):        
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 96, 96)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(3*3*3) # was = 0.25   
    binit = 0.0
    
    def trec(x):
      return x*(x > 0.0)

    nonlinearity = trec

    conv1 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        input, 
        n_filters=96,
        filter_size=3,
        weights_std=winit1,
        nonlinearity=nonlinearity,
        pad=1)
    pool1 = cc_layers.CudaConvnetPooling2DLayer(conv1, 2, stride=2)
    unpool2 = cc_layers.CudaConvnetUnpooling2DLayer(pool1, pool1)
    output = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        unpool2, conv1, nonlinearity=layers.identity)    


class SupervisedModel(fastor.models.SupervisedModel):    
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 96, 96)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(3*3*3) # was = 0.25  
    binit = 0.0
    
    def trec(x):
      return x*(x > 0.0)

    nonlinearity = trec

    conv1 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        input, 
        n_filters=96,
        filter_size=3,
        weights_std=winit1,
        nonlinearity=nonlinearity,
        pad=1)
    pool1 = cc_layers.CudaConvnetPooling2DLayer(conv1, 2, stride=2)

    winitD1 = k/numpy.sqrt(numpy.prod(pool1.get_output_shape()))
    winitD2 = k/numpy.sqrt(300)

    pool1_shuffle = cc_layers.ShuffleC01BToBC01Layer(pool1)    
    fc2 = layers.DenseLayer(
        pool1_shuffle,
        n_outputs = 300,
        weights_std=winitD1,
        init_bias_value=1.0,
        nonlinearity=layers.rectify,
        dropout=0.0)
    output = layers.DenseLayer(
        fc2,
        n_outputs=10,
        weights_std=winitD2,
        init_bias_value=0.0,
        nonlinearity=layers.softmax)
import numpy

from fastor.layers import layers, cc_layers
import fastor.models

class UnsupervisedModel(fastor.models.UnsupervisedModel):        
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 32, 32)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(7*7*3) # was = 0.25   
    binit = 0.0
    
    def trec(x):
      return x*(x > 0.0)

    nonlinearity = trec

    conv1 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        input, 
        n_filters=96,
        filter_size=7,
        weights_std=winit1,
        nonlinearity=nonlinearity,
        pad=3)
    output = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        conv1, conv1, nonlinearity=layers.identity)    


class SupervisedModel(fastor.models.SupervisedModel):    
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 32, 32)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(7*7*3) # was = 0.25  
    binit = 0.0
    
    def trec(x):
      return x*(x > 0.0)

    nonlinearity = trec

    conv1 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        input, 
        n_filters=96,
        filter_size=7,
        weights_std=winit1,
        nonlinearity=nonlinearity,
        pad=3)
    winitD1 = k/numpy.sqrt(numpy.prod(conv1.get_output_shape()))
    winitD2 = k/numpy.sqrt(300)

    conv1_shuffle = cc_layers.ShuffleC01BToBC01Layer(conv1)    
    fc2 = layers.DenseLayer(
        conv1_shuffle,
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

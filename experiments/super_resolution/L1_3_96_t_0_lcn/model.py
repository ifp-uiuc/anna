import numpy

from fastor.layers import layers, cc_layers
import fastor.models

class RegressionModel(fastor.models.RegressionModel):        
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 96, 96)
    y = cc_layers.CudaConvnetInput2DLayer(batch, 3, 96, 96)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(3*3*3) # was = 0.25   
    winit2 = k/numpy.sqrt(3*3*96) # was = 0.25   
    binit = 0.0
    
    def trec(x):
      return x*(x > 0.0)

    nonlinearity = trec
    # nonlinearity = layers.identity

    conv1 = cc_layers.CudaConvnetConv2DLayer(
        input, 
        n_filters=96,
        filter_size=3,
        weights_std=winit1,
        init_bias_value=binit,
        nonlinearity=nonlinearity,
        pad=1)
    conv2 = cc_layers.CudaConvnetConv2DLayer(
        conv1, 
        n_filters=96,
        filter_size=3,
        weights_std=winit1,
        init_bias_value=binit,
        nonlinearity=nonlinearity,
        pad=1)
    deconv3 = cc_layers.CudaConvnetDeconvUntied2DLayer(
        conv2, conv2, nonlinearity=layers.identity)
    output = cc_layers.CudaConvnetDeconvUntied2DLayer(
        deconv3, conv1, nonlinearity=layers.identity)

class RegressionModel2(fastor.models.RegressionModel):        
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 96, 96)
    y = cc_layers.CudaConvnetInput2DLayer(batch, 3, 96, 96)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    nn = 48
    winit1 = k/numpy.sqrt(3*3*3) # was = 0.25   
    winit2 = k/numpy.sqrt(3*3*nn) # was = 0.25   
    winit3 = k/numpy.sqrt(3*3*nn) # was = 0.25   
    winit4 = k/numpy.sqrt(3*3*nn) # was = 0.25   
    winit5 = k/numpy.sqrt(3*3*nn) # was = 0.25   
    binit = 1.0
    
    def trec(x):
      return x*(x > 0.0)

    nonlinearity = trec
    # nonlinearity = layers.identity

    conv1 = cc_layers.CudaConvnetConv2DLayer(
        input, 
        n_filters=nn,
        filter_size=3,
        weights_std=winit1,
        init_bias_value=binit,
        nonlinearity=nonlinearity,
        pad=1)
    print conv1.get_output_shape()
    conv2 = cc_layers.CudaConvnetConv2DLayer(
        conv1, 
        n_filters=nn,
        filter_size=3,
        weights_std=winit2,
        init_bias_value=binit,
        nonlinearity=nonlinearity,
        pad=1)
    print conv2.get_output_shape()
    conv3 = cc_layers.CudaConvnetConv2DLayer(
        conv2, 
        n_filters=nn,
        filter_size=3,
        weights_std=winit3,
        init_bias_value=binit,
        nonlinearity=nonlinearity,
        pad=1)
    print conv3.get_output_shape()
    conv4 = cc_layers.CudaConvnetConv2DLayer(
        conv3, 
        n_filters=nn,
        filter_size=3,
        weights_std=winit4,
        init_bias_value=binit,
        nonlinearity=nonlinearity,
        pad=1)
    print conv4.get_output_shape()
    output = cc_layers.CudaConvnetConv2DLayer(
        conv4, 
        n_filters=16,
        filter_size=3,
        weights_std=winit5,
        init_bias_value=binit,
        nonlinearity=layers.identity,
        pad=1)
    print output.get_output_shape()
import numpy

from fastor.layers import layers, cc_layers
import fastor.models

class UnsupervisedModel(fastor.models.UnsupervisedModel):        
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 96, 96)
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(3*3*3) # was = 0.25 
    winit2 = k/numpy.sqrt(3*3*96)
    winit3 = k/numpy.sqrt(3*3*128)
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
    lcn1 = cc_layers.LcnLayer(pool1, 
        filter_size=3, 
        num_channels=96, 
        num_filters=96)
    conv2 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        lcn1, 
        n_filters=128,
        filter_size=3,
        weights_std=winit2,
        nonlinearity=nonlinearity,
        pad=1)
    pool2 = cc_layers.CudaConvnetPooling2DLayer(conv2, 2, stride=2)
    lcn2 = cc_layers.LcnLayer(pool2, 
        filter_size=3, 
        num_channels=128, 
        num_filters=128)
    conv3 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        lcn2, 
        n_filters=512,
        filter_size=3,
        weights_std=winit3,
        nonlinearity=nonlinearity,
        pad=1)
    pool3 = cc_layers.CudaConvnetPooling2DLayer(conv3, 2, stride=2)
    unpool4 = cc_layers.CudaConvnetUnpooling2DLayer(pool3, pool3)
    deconv4 = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        unpool4, conv3, nonlinearity=layers.identity)        
    unpool5 = cc_layers.CudaConvnetUnpooling2DLayer(deconv4, pool2)
    deconv5 = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        unpool5, conv2, nonlinearity=layers.identity)
    unpool6 = cc_layers.CudaConvnetUnpooling2DLayer(deconv5, pool1)
    output = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        unpool6, conv1, nonlinearity=layers.identity)


class SupervisedModel(fastor.models.SupervisedModel):
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 96, 96)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(3*3*3) # was = 0.25 
    winit2 = k/numpy.sqrt(3*3*96)
    winit3 = k/numpy.sqrt(3*3*128)
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
    lcn1 = cc_layers.LcnLayer(pool1, 
        filter_size=3, 
        num_channels=96, 
        num_filters=96)
    conv2 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        pool1, 
        n_filters=128,
        filter_size=3,
        weights_std=winit2,
        nonlinearity=nonlinearity,
        pad=1)
    pool2 = cc_layers.CudaConvnetPooling2DLayer(conv2, 2, stride=2)
    lcn2 = cc_layers.LcnLayer(pool2, 
        filter_size=3, 
        num_channels=128, 
        num_filters=128)
    conv3 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        lcn2, 
        n_filters=512,
        filter_size=3,
        weights_std=winit3,
        nonlinearity=nonlinearity,
        pad=1)
    pool3 = cc_layers.CudaConvnetPooling2DLayer(conv3, 2, stride=2)

    winitD1 = k/numpy.sqrt(numpy.prod(pool3.get_output_shape()))
    winitD2 = k/numpy.sqrt(300)

    pool3_shuffle = cc_layers.ShuffleC01BToBC01Layer(pool3)    
    fc4 = layers.DenseLayer(
        pool3_shuffle,
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
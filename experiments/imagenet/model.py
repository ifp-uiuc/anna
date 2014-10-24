import numpy

from fastor.layers import layers, cc_layers
import fastor.models

class UnsupervisedModel(fastor.models.UnsupervisedModel):        
    batch = 128
    input = cc_layers.CudaConvnetInput2DLayer(batch, 3, 224, 224)

    num_filters0 = 3
    num_filters1 = 96
    num_filters2 = 256
    num_filters3 = 512
    num_filters4 = 1024
    num_filters5 = 512

    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(3*3*num_filters0) # was = 0.25 
    winit2 = k/numpy.sqrt(3*3*num_filters1)
    winit3 = k/numpy.sqrt(3*3*num_filters2)
    winit4 = k/numpy.sqrt(3*3*num_filters3) 
    winit5 = k/numpy.sqrt(3*3*num_filters4)
    binit = 0.0

    def trec(x):
      return x*(x > 0.0)

    nonlinearity = trec

    conv1 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        input, 
        n_filters=num_filters1,
        filter_size=7,
        weights_std=winit1,
        nonlinearity=nonlinearity,
        stride=2,
        pad=1)
    pool1 = cc_layers.CudaConvnetPooling2DLayer(conv1, 3, stride=2)
    conv2 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        pool1, 
        n_filters=num_filters2,
        filter_size=5,
        weights_std=winit2,
        nonlinearity=nonlinearity,
        stride=2,
        pad=1)
    pool2 = cc_layers.CudaConvnetPooling2DLayer(conv2, 3, stride=2)
    conv3 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        pool2, 
        n_filters=num_filters3,
        filter_size=3,
        weights_std=winit3,
        nonlinearity=nonlinearity,
        pad=1)
    conv4 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        conv3, 
        n_filters=num_filters4,
        filter_size=3,
        weights_std=winit4,
        nonlinearity=nonlinearity,
        pad=1)
    conv5 = cc_layers.CudaConvnetConv2DNoBiasLayer(
        conv4, 
        n_filters=num_filters5,
        filter_size=3,
        weights_std=winit5,
        nonlinearity=nonlinearity,
        pad=1)
    pool5 = cc_layers.CudaConvnetPooling2DLayer(conv5, 3, stride=2)
    unpool6 = cc_layers.CudaConvnetUnpooling2DLayer(pool5, pool5)
    deconv6 = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        unpool6, conv5, nonlinearity=layers.identity)
    deconv7 = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        deconv6, conv4, nonlinearity=layers.identity)
    deconv8 = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        deconv7, conv3, nonlinearity=layers.identity)
    unpool9 = cc_layers.CudaConvnetUnpooling2DLayer(deconv8, pool2)
    deconv9 = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        unpool9, conv2, nonlinearity=layers.identity)
    unpool10 = cc_layers.CudaConvnetUnpooling2DLayer(deconv9, pool1)
    output = cc_layers.CudaConvnetDeconv2DNoBiasLayer(
        unpool10, conv1, nonlinearity=layers.identity)

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
        n_filters=384,
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
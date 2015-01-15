# Getting started
This part of the documentation will give you an overview of the main parts of Anna, and then walk you through the steps required to run your first experiment. 

Anna has five main parts:
+ layers
+ models
+ datasets
+ util
+ scripts

The core neural network functionality is all in the `layers` and `models`. Everything else is just there to help you run experiments.

So let's start by using the layers to make a model.

Anna has basic `model` classes that perform a lot of the core functionality for you. These models have three basic methods:
+ `predict`
+ `eval`
+ `train`

But these base model classes don't specify any model structure. To do that you need to create a new model that inherits from a base class, and implement the model structure using anna's `layers`.

```
from anna.layers import layers, cc_layers
from anna.models import SupervisedModel

class MyModel(SupervisedModel):
    def __init__(self):
        self.input = cc_layers.Input2D(minibatch_size=128,
                                       num_channels=3,
                                       width=32,
                                       height=32)
        self.conv1 = cc_layers.Conv2D(self.input,
                                      num_filters=96)
        self.output = layers.Dense(self.conv1, nonlinearity=layers.softmax)

model = MyModel()
```

Notice about we imported two modules from layers:
+ layers
+ cc_layers

`layers` is a collection of layers built using Theano's own kernels, including the ones used for 2D image convolution.
`cc_layers` is a collection of layers built using kernels originally from `cuda_convnet`, which were designed specifically for 2D image convolution. 

Note: Theano has recently adopted cuDNN's convolutional kernels, and so you might want to ignore cc_layers in the near future.

After we create the model, we instantiate it. This takes a while, because this is when are actually having Theano compile efficient c code to perform the computations for our neural network.

Now that we have created a model, we can give it data. Now, because Anna can be used to generate neural networks with many different input and output shapes, we have to know what the input and output dimensions should be.

This can be done by running:
model.input_shape()
model.output_shape()
# TODO(tpaine) add these functions.

x_batch = numpy.zeros(model.input_shape(), model.input_dtype())
y_batch = numpy.zeros(model.output_shape(), model.output_dtype())

MINIBATCH_SIZE = 'minibatch_size'
WIDTH = 'width'
HEIGHT = 'height'
CHANNELS = 'num_channels'
(MINIBATCH_SIZE, WIDTH, HEIGHT, CHANNELS)

It is up you, the user, to ensure that the data fed into the neural network are the right shape and data type.

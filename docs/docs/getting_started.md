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


## How to Write a Model

Anna has basic `model` classes that perform a lot of the core functionality for you. These models have three basic methods:
+ `predict`
+ `eval`
+ `train`

But these base model classes don't specify any model structure. To do that you need to create a new model that inherits from a base class, and implement the model structure using anna's `layers`.

``` python
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


## What About Data?

Now that we have created a model, we can give it data. Since Anna can be used to generate neural networks with many different input and output shapes, we have to know what the input and output dimensions should be.

This can be done by running:

``` python
model.input_shape()
model.output_shape()
```
## TODO(tpaine) add these functions.

With these attributes, we can create a minibatch of data the can be fed into 
our neural network in the following manner:

``` python
x_batch = numpy.zeros(model.input_shape(), model.input_dtype())
y_batch = numpy.zeros(model.output_shape(), model.output_dtype())
```

MINIBATCH_SIZE = 'minibatch_size'  
CHANNELS = 'num_channels'  
WIDTH = 'width'  
HEIGHT = 'height'  
(MINIBATCH_SIZE, CHANNELS, WIDTH, HEIGHT)  

It is up you, the user, to ensure that the data fed into the neural network are the right shape and data type.

## How to Make a Dataset

Now that we know how to create a minibatch, we will show you how to randomly 
generate multiple minibatches using a dataset and an iterator. If you are 
already familiar with pylearn2, you can use one of the pre-packaged iterators 
to generate minibatches. Otherwise, we will show you how to make your own 
dataset from scratch.

Before we start, we will need a few libraries: 

``` python
import numpy
from anna.datasets.supervised_dataset import SupervisedDataset
```  

For the purposes of this example, let's consider a simple case where the 
training data (`X_train`) contains 10000 random 32x32 pixel images generated from a 
uniform distribution and the labels (`y_train`) are randomly selected integers ranging 
between 0 and 9. We house all of the samples in a `SupervisedDataset` object called `train_dataset`, and randomly sample minibatches of size 128 using an iterator: `train_iterator`.

``` python
X_train = numpy.random.rand(10000, 3, 32, 32)
y_train = numpy.random.randint(0, 10, 10000)    
    
train_dataset = SupervisedDataset(X_train, y_train)

train_iterator = train_dataset.iterator(
	mode='random_uniform', batch_size=128, num_batches=100000)
```

Now, if you want a minibatch of size 128, all you need is the
following code:

``` python
x_batch, y_batch = train_iterator.next()
```

We also create a test dataset in a similar fashion to the training dataset. This
test dataset will contain 1000 random 32x32 color images (`X_test`) and labels
(`y_test`). 

``` python
X_test = numpy.random.rand(1000, 3, 32, 32)
y_test = numpy.random.randint(0, 10, 1000)

test_dataset = SupervisedDataset(X_test, y_test)

test_iterator = test_dataset.iterator(
	mode='random_uniform', batch_size=128, num_batches=10000)
```

## How to Train the Network

Now that we can iterate over our dataset, we can train our model using the 
following code:

``` python
print('Training Model')
for x_batch, y_batch in train_iterator:      
    log_prob, accuracy = model.train(x_batch, y_batch)
    print('Log Prob: %f --- Accuracy: %f' % (log_prob, accuracy))
```

                              

## Incorporating Utils

## Talk about using the monitor


## Full Experiment File

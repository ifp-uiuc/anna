# Welcome to Anna!

Anna is a neural network library, written in Python, for human beings.

There are many other neural network libraries, which excel in different areas. We've used many of them and love bits of each. But we wanted something a little bit simplier. We want it to be simple to create a neural networks, train them, and use them. We also want it to be easy to do common things done is research right now.

Here's how you would create a simple neural network, and train it on random data:

```
import numpy

from anna.models import SupervisedModel
from anna.layers import layers, cc_layers

# Setup the model
class Model(SupervisedModel):
    def __init__(self):
        self.input = cc_layers.Input2D(mb_size=128,
                                       n_features=3,
                                       width=32,
                                       height=32)
        self.conv1 = cc_layers.Conv2D(self.input,
                                      n_filters=96,
                                      filter_size=3,
                                      weights_std=0.01)
        self.conv2 = cc_layers.Conv2D(self.conv1,
                                      n_filters=96,
                                      filter_size=3,
                                      weights_std=0.01)
        self.conv2_shuffle = cc_layers.ShuffleCCtoStd(self.conv2)
        self.output = layers.Dense(self.conv2_shuffle)

# Initialize the model
model = Model()

# Generate boring data
x_data = numpy.random.zeros((128, 3, 32, 32))
y_data = numpy.random.zeros((128, 10))

# Run the model on the data
model.predict(x_data)
# Train the model on the data
model.train(x_data, y_data)
# Evaluate the model on the data
model.eval(x_data, y_data)
```

Anna's goal is to take all the work out of using Theano to make neural networks, by providing simple building blocks to make creating models easy, and simple model templates to help with prediction, training, and evaluation.

## Features

+ We should think about our features
+ Obvious ones we may or may not use
+ Checkpointing
+ Loading from checkpoint
+ Simple datasets
+ Built on Theano for easy extension

## User guide
This section of the documentation helps you install Anna, walks you through your first Anna experiment, and focuses on special topics for getting the most out of Anna.

+ Installation
+ Wall-through
### Special topics

## Contribute

+ Issue Tracker: github.com/ifp-uiuc/anna/issues
+ Source Code: github.com/ifp-uiuc/anna

# Support

If you are having issues, please let us know.
We have a mailing list located at: TDB

# License

The project is licensed under the BSD license.

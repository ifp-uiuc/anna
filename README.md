anna - artifical neural networks, anytime 
================

This is a code repository to efficiently train a deconvolutional neural network with rectified linear units.

This code uses [theano][theano], [pylearn2][pylearn2], [cuda-convnet][ccnet] and is heavily based on Sander Dieleman's [kaggle galaxy repo][galaxy].

It also currently relies on a change to `pylearn2.sandbox.cuda_convnet.pool.py` that defines a `grad` method for the `MaxPoolGrad` class, which can be useful in some cases...

## Layout of the code
There are currently 3 main modules:
+ datasets - generic dataset classes
+ layers - layer definitions
+ util - utilities for training, evaluating, and saving/loading checkpoints  

[theano]:https://github.com/Theano/Theano
[pylearn2]:https://github.com/lisa-lab/pylearn2
[ccnet]:http://code.google.com/p/cuda-convnet/
[galaxy]:https://github.com/benanne/kaggle-galaxies

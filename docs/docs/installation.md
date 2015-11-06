# Installation

This guide describes how to install anna on Ubuntu 12.04. 

## Depenendencies

In order to install and use anna properly, you will first need the following dependencies:

+ CUDA

+ Python

+ theano

+ pylearn2


## Installing CUDA

TBD

## Installing Python

We highly recommend using the [anaconda](https://store.continuum.io/cshop/anaconda/) scientific python distribution, mainly because it is easy to install and already includes several useful packages (numpy, scipy, matplotlib, ipython etc.)

Downloading and installing anaconda should be more or less straightforward. Just click "Download Anaconda" and follow the instructions to install it in your home directory. Towards the end, the installer will ask if you would like to include anaconda in your PATH environment variable. Say yes and verify this by opening your .bashrc file in your home directory. There should be a line that looks like this:

    export PATH=/home/your_user_name/anaconda/bin:$PATH

Apply the changes by typing ` source ~/.bashrc ` in the terminal. 

Now if you open python from the terminal, you will see something like this:

    Python 2.7.6 |Anaconda 1.9.1 (64-bit)| (default, Jan 17 2014, 10:13:17)
    [GCC 4.1.2 20080704 (Red Hat 4.1.2-54)] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>>

## Installing theano

[Theano](http://deeplearning.net/software/theano/) is a python library that is used for designing and optimizating symbolic mathematical expressions. These symbolic expressions form the backbone of the anna library. As such, it is absolutely essential to have a working version of theano before installing anna. (Give more detail on importance of theano?)

While there appear to be mulitple ways to install theano, we recommend cloning the git repository to get the most up-to-date version:

    $ git clone git://github.com/Theano/Theano.git
    $ cd ./Theano
    $ sudo python setup.py develop
    
The second command allows you to bypass the step where you must add theano to the PYTHONPATH and makes updating easier.

To check if the installation was successful, simply open up python and type:

    >>> import theano

For further reference, please go [here](http://deeplearning.net/software/theano/install.html#install).


## Installing pylearn2

[pylearn2](http://deeplearning.net/software/pylearn2/), like anna, is a machine learning library that uses theano. Installing 
pylearn2 will give anna access to the cuda kernels, originally designed by Alex Krizhevsky, that define the behavior of several of the network layers.

To install, simply clone the git repository:

    $ git clone git://github.com/lisa-lab/pylearn2.git

Make sure to add pylearn2 to your PYTHONPATH environment variable in your ~/.bashrc file so that python can locate it:
    
    export PYTHONPATH=$PYTHONPATH:/path/to/pylearn2/

Then, type ` source ~/.bashrc ` in the terminal to apply the changes.

(Or talk about running  python setup.py develop in top-level pylearn2 directory?, might accomplish the same thing and will also compile the Cython extensions required for e.g. pylearn2.train_extensions.window_flip)


To check if the installation was successful, simply open up python and type:

    >>> import pylearn2

## Installing anna

Once you have installed all of the dependencies, you will be ready to install anna. You can do this by checking out the latest version from github using the following terminal command:

    $ git clone https://github.com/ifp-uiuc/anna.git
    
Now, add anna to your PYTHONPATH environment variable in your ~/.bashrc file so that python knows where to look:

    export PYTHONPATH=$PYTHONPATH:/path_to_anna/anna

Apply the changes by typing ` source ~/.bashrc ` in the terminal and you're all done! Now just open python and type:

    >>> import anna
    
If no error pops up, you will have a working copy of anna! Congratulations!



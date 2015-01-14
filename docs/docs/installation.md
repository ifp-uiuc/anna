# Installation

This guide describes how to install anna on Ubuntu 12.04. 

## Depenendencies

In order to install and use anna properly, you will first need the following dependencies:

+ cuda

+ python

+ theano

+ pylearn2


## Installing cuda

## Installing python

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

[Theano](http://deeplearning.net/software/theano/) is a python library that is used for designing and optimizating symbolic mathematical expressions. (Give more detail on importance of theano?)

While there appear to be mulitple ways to install theano, we recommend cloning the git repository to get the most up-to-date version:

    $ git clone git://github.com/Theano/Theano.git
    $ python setup.py develop
    
The second command 

To check if the installation was successful, simply open up python and type:

    >>> import theano

For further reference, please go [here](http://deeplearning.net/software/theano/install.html#install).


## Installing pylearn2

[pylearn2](http://deeplearning.net/software/pylearn2/) 

To install, simply clone the git repository:

    $ git clone git://github.com/lisa-lab/pylearn2.git

Then, add pylearn2 to your PYTHONPATH environment variable in your ~/.bashrc file so that python can locate it:
    
    export $PYTHONPATH=$PYTHONPATH:/path_to_anna/anna

Type the following in the terminal to apply the changes:

    $ source ~/.bashrc
    
To check if the installation was successful, simply open up python and type:

    >>> import pylearn2

## Getting anna from git

Once you have installed all the dependencies, you are ready to install anna. You can do this by checking out the latest version from github using the following terminal command:

    $ git clone https://github.com/ifp-uiuc/anna.git
    
Now, add anna to your PYTHONPATH environment variable in your ~/.bashrc file so that python knows where to look:

    export $PYTHONPATH=$PYTHONPATH:/path_to_anna/anna

And then type the following in the terminal to apply the changes.
    
    $ source ~/.bashrc

You're all done! Now if you open python and type:

    >>> import anna
    
You will have a working copy of anna. Congratulations!



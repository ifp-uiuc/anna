# How to train a model

Now to you have written your first model, you are now ready to train it.
This process can be broken up into three simple steps:

1. Load a model 

2. Load the data

3. Run the training loop

We will go through these steps one at a time. Before we start, we 
will need some libraries: 

    import numpy

    from model import MyModel
    from anna.datasets import supervised_dataset


## Loading a model

Once you've specified your model's structure in model.py, loading it 
is very simple.

    model_name = 'test_simple_model'
    model_path = './'
    model = MyModel(model_name, model_path, learning_rate=1e-2)


## Loading the data

Now that you have our model, you will need some data. For the purposes 
of this example, let's consider a simple case where the data (X) is a 
bunch of random images generated from a uniform distribution and the 
labels (y) are randomly selected integers ranging between 0 and 9.

    X = numpy.random.rand(1000, 3, 32, 32)
    y = numpy.random.randint(0, 10, (1000, 1))    
    
    
    X_train = X[0:900, :, :, :]
    X_test = X[900:, :, :, :]
    y_train = y[0:900]
    y_test = y[900:]
    

    train_dataset = supervised_dataset.SupervisedDataset(X_train, y_train)
    test_dataset = supervised_dataset.SupervisedDataset(X_test, y_test)
    train_iterator = train_dataset.iterator(
        mode='random_uniform', batch_size=128, num_batches=10000)
    test_iterator = test_dataset.iterator(
        mode='random_uniform', batch_size=128, num_batches=10000)



## Running the training loop

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:41:49 2020

@author: Mauri
"""

"""
mnist_data
~~~~~~~~~~~~
A library to load the MNIST image data.  load_data_wrapper is the
function called by our neural network code.
"""

#### Libraries
import pickle
import gzip
import numpy as np

def data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The training_data is returned as a tuple with two entries.
    The first entry contains the actual training images.  
    The second entry in the training_data tuple is a numpy ndarray
    containing 50,000 entries.  
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_data, val_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (train_data, val_data, test_data)

def data_loader():
    """Return a tuple containing (training_data, validation_data,
    test_data). Based on ``data, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, training_data is a list containing 50,000
    2-tuples (x, y).
    validation_data and test_data are lists containing 10,000
    2-tuples (x, y)."""
    tr_d, va_d, te_d = data()
    train_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    train_results = [vector_result(y) for y in tr_d[1]]
    train_data = zip(train_inputs, train_results)
    val_in = [np.reshape(x, (784, 1)) for x in va_d[0]]
    val_data = zip(val_in, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (train_data, val_data, test_data)

def vector_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
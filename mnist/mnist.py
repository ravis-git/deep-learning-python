# Shallow Net in Keras
## Build a shallow neural network to classify MNIST data

# set seed for reproducability
import numpy as np
np.random.seed(42)

# load dependencies
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# load data
(x_train, y_train), (X_test, y_test) = mnist.load_data()

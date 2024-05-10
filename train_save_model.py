import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Input, RandomRotation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import numpy as np
import random
import matplotlib.pyplot as plt

def get_model():
  """
  Creates a sequential neural network model for MNIST handwritten digit classification.

  This function defines a sequential model architecture that achieved high test accuracy 
  on the MNIST dataset in the previous assignment. The model consists of three fully-connected 
  dense layers with sigmoid and softmax activations for hidden and output layers, respectively.

  Returns:
    keras.Sequential: The compiled neural network model.
  """
  model = Sequential([
      Input(shape=(784,)),
      Dense(units=256, activation='sigmoid'),
      Dense(units=128, activation='sigmoid'),
      Dense(units=10, activation='softmax')
  ])
  return model


def train_model():
  """
  Trains a neural network model for MNIST handwritten digit classification.

  This function retrieves the pre-defined model architecture, compiles it with an appropriate 
  loss function (Sparse Categorical Crossentropy) and accuracy metric, loads the MNIST dataset, 
  performs data preprocessing (flattening and normalization), trains the model on the training data 
  with a 20% validation split for 10 epochs, evaluates the model performance on the test data, 
  and finally saves the trained model for later use.

  """
  # Get the neural network model
  model = get_model()

  # Compile the model with loss and metrics
  model.compile(loss=SparseCategoricalCrossentropy(), metrics=['acc'])

  # Load the MNIST dataset
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # Preprocess the data (flattening and normalization)
  x_train = np.array(x_train, dtype='float32').reshape(60000, -1) / 255.0
  x_test = np.array(x_test, dtype='float32').reshape(10000, -1) / 255.0

  # Train the model
  model.fit(x_train, y_train, validation_split=0.2, epochs=10)

  # Evaluate the model on test data
  model.evaluate(x_test, y_test)

  # Save the trained model for later use
  model.save('MNIST_model.keras')


if __name__ == "__main__":
  train_model()

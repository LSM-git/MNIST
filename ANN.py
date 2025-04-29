# Main file for the computations of the neural network

import os, random, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

'''
Network to implement:

28 x 28 inputs 
1 hidden layer
10 outputs
Classify digits from 0-9
'''

n = [28*28, 128, 10]
EPOCHS = 250
TRAINING_DIRECTORY = "training"
PARAM_FILE = "parameters.pkl"

class Parameters:
    def __init__(self):
      # Define the layers and initialise weights and biases 
      self.W1 = np.random.rand(n[1], n[0])
      self.W2 = np.random.rand(n[2], n[1])
      self.b1 = np.random.rand(n[1], 1)
      self.b2 = np.random.rand(n[2], 1)

    def get_parameters(self):
        return self.W1, self.W2, self.b1, self.b2

def sigmoid(x):
    '''
    Activation function for hidden layer.
    Keeps the values between 0 and 1.
    Args:
      x: A 28x28-dimensional vector containing the computations of the input layer
    Returns: The result of applying the sigmoid function to x component-wise 
    '''
    return 1/(1+np.exp(-x))

def softmax(z: np.array) -> np.array:
    '''
    Softmax classification function for the output layer
    Args:
      z: A 128-dimensional vector containing the computations of the hidden layer
    Returs: A 10 dimensional vector where each component is between 0 and 1
    '''
    expVector = np.exp(z)
    return expVector / np.sum(expVector)

def imageToVector(image_path: str):
    '''
    Transforms a black and white image to a numpy array (vector)
    Args:
      image_path (str): Path to the image to transform
    Returns: A 28x28 dimensional vector containing the pixel values of the image.
    '''
    return np.resize(np.array(Image.open(image_path))/255, (n[0],1))


def feedforward(X, params):
    '''
    Feedforward function. 
    Args:
      X: Column vector containing the pixel values of a 28x28 image
      params: Parameter object containing the weights and biases of the model
    Returns: A prediction as a column vector and the resulting matrix of the sigmoid function applied to the hidden layer input.
    '''
    W1, W2, b1, b2 = params.get_parameters()

    T = W1 @ X + b1
    V = sigmoid(T)
    Z = W2 @ V + b2
    return softmax(Z), V

def costFunction(y: np.array, y_hat: np.array):
    '''
    Multiclass cross entropy function that determines the error.
    The objective of machine learning is to find the parameters that minimize this function.
    It is the sum of the components of -y*log(y_hat)
    Args:
      y: Desired output
      y_hat: The value obtained from the model
    '''
    return -(1/n[2])*np.sum(y*np.log(y_hat))

def backpropagation(y, y_hat, V, W2, X):
    '''
    Backpropagation function that calculates changes in parameters to minimize error function.
    Args:
      y: Desired output
      y_hat: Obtained output from the model
      V: Hidden layer matrix
      W2: Weight matrix for second layer
      X: Input vector
    Returns: The changes in the error function with respect to W2, W1, b2 and b1
    '''
    # Outer layer
    # Softmax + cost function differentials simplify to
    y_hat = y_hat.reshape(-1,1)
    y = y.reshape(-1,1)
    dC_dZ = y_hat - y

    dC_dW2 = dC_dZ @ V.T # 10 x 128 matrix
    dC_db2 = dC_dZ # 10 dim vector
    
    # Hidden layer 
    dC_dT = (W2.T @ dC_dZ) * (V * (np.ones((n[1], 1)) - V)) # 10 x 128 matrix

    dC_dW1 = dC_dT @ X.T # 128 x 784 matrix
    dC_db1 = dC_dT # 128 dim vector

    return dC_dW2, dC_dW1, dC_db2, dC_db1


def train(params: Parameters):
    '''
    Train function for model.
    '''
    global n
    W1, W2, b1, b2 = params.get_parameters()

    alpha = 0.01
    costs = []
    for e in range(EPOCHS):
        num = e%10
        print(f"Starting epoch: {e}")

        # Choose a random digit image
        path = random.choice(os.listdir(f"./{TRAINING_DIRECTORY}/{e%10}"))
        X = imageToVector(f"./{TRAINING_DIRECTORY}/{num}/{path}")

        y_hat, v = feedforward(X, params)
        y = np.array([0]*num + [1] + [0]*(9-num))

        error = costFunction(y, y_hat)
        costs.append(error)

        dC_dW2, dC_dW1, dC_db2, dC_db1 = backpropagation(y,y_hat, v, W2, X)

        W2 -= alpha * dC_dW2
        b2 -= alpha * dC_db2

        W1 -= alpha * dC_dW1
        b1 -= alpha * dC_db1
    return costs   

def main():
    if '--no-training' in sys.argv:
        sys.argv.remove('--no-training')
        # Load parameters
        with open(PARAM_FILE, 'rb') as param_file:
            params =pickle.load(param_file)
    else:
      params = Parameters()
      costs = train(params)

      # Serialize parameters
      with open(PARAM_FILE, 'wb') as param_file:
          pickle.dump(params, param_file)

      # Show error graph
      ax = plt.gca()
      ax.set_xlim([0, EPOCHS])
      ax.set_ylim([-0.2, 6])

      xs = [x for x in range(EPOCHS)]
      plt.plot(xs, costs)
      plt.show()

    for image in sys.argv[1:]:
      print(feedforward(imageToVector(image), params)[0])

if __name__ == '__main__':
    main()

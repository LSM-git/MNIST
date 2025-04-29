# Introduction
Hi! Neural networks have always interested me so I decided to learn how they work by writting one from scratch using numpy.
The network recognizes handdrawn digits from a 28x28 image.
This readme explains how the network works but its a MNIST classifier is a classic in the artificial intelligence area and there are many examples on the internet. 

# Training

For each epoch the model selects a random image from a training dataset for its training. 
This way the model is trained on every digit. 
Therefore it is necesary a dataset with the following structure:

```
train/
  ├── 0/
  ├── 1/
  ├── 2/
  ├── 3/
  ├── 4/
  ├── 5/
  ├── 6/
  ├── 7/
  ├── 8/
  └── 9/
```
Such dataset can be found on the following kaggle url: https://www.kaggle.com/datasets/shreyasi2002/corrupted-mnist

Further details of the implementation are discussed below.
At the end of the training, the program provides a graph showing the error vs the epochs. 

> **Note**  
> To skip training, run the program with the `--no-training` option to use the weights and biases stored in the `parameters.pkl` file.  
> This file contains the weights and biases obtained during my training.


# Network Topology
This neural network has 28x28 input neurons, 10 output neurons and a single hidden layer with 128 neurons. 
I chose a single hidden layer to simplify the calculations.
The images don't have a lot of information so further layers won't be necessary.

# Feedforward
With this network topology the feedforward is fairly simple. 
The first layer is obtained from the following calculation:

Then the sigmoid function is applied to restrain the values and prevent a linear model.
Once the sigmoid values are obtained the second layer is calculated with the following calculation:

Finally the softmax function is applied to Z to obtain the feedforward result.
Where SF: R^128 --> [0,1]^10.
The feedforward function also returs the V matrix for backpropagation.

# Backpropagation

# Conclusion
I had a lot of fun doing this project and taught how supervised machine learning models work, which is something I've always been curious about.
I had a lot of problems using matrix operations to optimize the model but it was worth it.
It is amazing to see how the model lears and is able to recognize digits. 
Altought the project is not unique I would still recommend it if you are interested in machine learning.

# Introduction
Hi! Neural networks have always interested me so I decided to learn how they work by writting one from scratch using numpy.
The network recognizes handdrawn digits from a 28x28 image.
This readme explains how to run the model and explains how the model works.

I had a lot of fun doing this project. It taught how supervised machine learning models work, which is something I've always been curious about.
It is amazing to see how the model learns and is able to recognize digits. 
Altought the project is not unique I would still recommend it if you are interested in machine learning.

# Training

For each epoch the model selects a random image from a training dataset for its training. 
This dataset contains image with digits from 0-9.
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
The training dataset I used can be found on the following kaggle url: https://www.kaggle.com/datasets/alexanderyyy/mnist-patched-2022.

For the execution of the model I provided a poetry pyproject.toml file to handle dependencies.
I also provided a `setup.py` script which automatically downloads and organises the files needed for training.

Further details of the implementation are discussed below.
At the end of the training, the program provides a graph showing the error vs the epochs. 

> [!NOTE]
> In case you want to train the model with just 5 epochs the model correctly classifies most images.

> [!NOTE]
> To skip training, run the program with the `--no-training` option to use the weights and biases stored in the `parameters.pkl` file. This file contains the weights and biases I obtained from 10 epochs of training.


# Network Topology
This neural network has 28x28 input neurons, 10 output neurons and a single hidden layer with 128 neurons. 
I chose a single hidden layer to simplify the calculations.
The images don't have a lot of information so further layers won't be necessary.
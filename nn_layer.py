"""
This scripts demonstrates how the Layers work!

"""

import numpy as np
from graph.layers import *

def test_sgd():
    inputs, weights, bias = Input(), Input(trainable=True), Input(trainable=True)
    random.seed(1)
    np.random.seed(1)

    x = np.array([[-1., -2.], [-1, -2],  [-1, -2]])
    x = np.array([[-1., -2., 2], [-1, -2, 2],  [-1, -2, 2]])

    w = np.random.rand(3,2)
    b = np.random.rand(2)

    ideal_output = np.array(
        [[0, 1],
        [0, 1],
        [0, 1]])

    f = Linear(inputs, weights, bias)
    g = Sigmoid(f)

    weights2, bias2 = Input(True), Input(True)
    w2 = np.random.rand(2,2)
    b2 = np.random.rand(2)

    f2 = Linear(g, weights2, bias2)
    g2 = Sigmoid(f2)
    cost = MSE(g2)

    feed_dict = {inputs: x, weights: w, bias: b, weights2: w2, bias2: b2}

    train_SGD(feed_dict, ideal_output, 10, learning_rate=0.1)
    print(x.shape)
    print(ideal_output.shape)

if __name__ == "__main__":
    test_sgd()




# Congratulations on making it to the end of this lab! Building a neural network from scratch is no small task. You should be proud!
# MiniFlow has the makings of becoming a powerful deep learning tool. It is entirely possible to classify something like the MNIST database with MiniFlow. MiniFlow only uses one training input at the moment and the biggest step remaining is iterating over many training inputs.
# I'll leave it as an exercise for you to finish MiniFlow from here.
# In the next lessons, you'll work with TensorFlow and then Keras, a high level framework on top of TensorFlow. Now that you've built a neural network from scratch, you should be in great shape!


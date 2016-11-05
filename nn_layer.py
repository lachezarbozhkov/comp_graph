"""
This scripts demonstrates how the Layers work!

"""

import numpy as np
from graph.layers import *

def test_sgd():
    inputs, weights, bias = Input(), Input(), Input()

    x = np.array([[-1., -2.], [-1, -2]])
    w = np.array([[random.random(), random.random()], [random.random(), random.random()]])
    b = np.array([random.random(), random.random()])
    ideal_output = np.array(
        [[1.23394576e-04, 9.82013790e-01],
        [1.23394576e-04, 9.82013790e-01]])

    f = Linear(inputs, weights, bias)
    g = Sigmoid(f)

    weights2, bias2 = Input(), Input()
    w2 = np.array([[random.random(), random.random()], [random.random(), random.random()]])
    b2 = np.array([random.random(), random.random()])

    f2 = Linear(g, weights2, bias2)
    cost = MSE(f2)

    feed_dict = {inputs: x, weights: w, bias: b, weights2: w2, bias2: b2}

    train_SGD(feed_dict, ideal_output, [weights, bias, weights2, bias2], 20, learning_rate=0.1)

if __name__ == "__main__":
    test_sgd()




# Congratulations on making it to the end of this lab! Building a neural network from scratch is no small task. You should be proud!
# MiniFlow has the makings of becoming a powerful deep learning tool. It is entirely possible to classify something like the MNIST database with MiniFlow. MiniFlow only uses one training input at the moment and the biggest step remaining is iterating over many training inputs.
# I'll leave it as an exercise for you to finish MiniFlow from here.
# In the next lessons, you'll work with TensorFlow and then Keras, a high level framework on top of TensorFlow. Now that you've built a neural network from scratch, you should be in great shape!


"""
This scripts demonstrates how the new Layers work!

"""

import numpy as np
from graph.layers import *



def test_layer():
    inputs, weights, bias = Input(), Input(), Input()

    f = Linear(inputs, weights, bias)

    x = np.array([[-1., -2.], [-1, -2]])
    w = np.array([[2., -3], [2., -3]])
    b = np.array([-3., -5])

    feed_dict = {inputs: x, weights: w, bias: b}

    graph = topological_sort(feed_dict, None)
    output = forward_pass(f, graph)

    """
    Output should be:
    [[-9., 4.],
    [-9., 4.]]
    """
    print("Output should be:\n", np.array([[-9., 4.], [-9., 4.]]))
    print(output)


def test_sigmoid():
    inputs, weights, bias = Input(), Input(), Input()

    f = Linear(inputs, weights, bias)
    g = Sigmoid(f)

    x = np.array([[-1., -2.], [-1, -2]])
    w = np.array([[2., -3], [2., -3]])
    b = np.array([-3., -5])

    feed_dict = {inputs: x, weights: w, bias: b}

    graph = topological_sort(feed_dict)
    output = forward_pass(g, graph)

    """
    Output should be:
    [[  1.23394576e-04   9.82013790e-01]
    [  1.23394576e-04   9.82013790e-01]]
    """
    print("Output should be:\n", np.array([[1.23394576e-04, 9.82013790e-01],
                                        [ 1.23394576e-04, 9.82013790e-01]]))
    print(output)


def test_MSE():
    
    inputs, weights, bias = Input(), Input(), Input()

    f = Linear(inputs, weights, bias)
    g = Sigmoid(f)

    x = np.array([[-1., -2.], [-1, -2]])
    w = np.array([[2., -3], [2., -3]])
    b = np.array([-3., -5])

    feed_dict = {inputs: x, weights: w, bias: b}

    graph = topological_sort(feed_dict)
    output = forward_pass(g, graph)

    ideal_output = np.array(
        [[1.23394576e-04, 9.82013790e-01],
        [1.23394576e-04, 9.82013790e-01]])

    cost = MSE(output) 

    """
    Output should be on the order of 1e-22.
    """
    print("Output should be on the order of 1e-22.")
    print(cost)


def test_backprop():
    inputs, weights, bias = Input(), Input(), Input()

    x = np.array([[-1., -2.], [-1, -2]])
    w = np.array([[2., -3], [2., -3]])
    b = np.array([-3., -5])
    ideal_output = np.array(
        [[1.23394576e-04, 9.82013790e-01],
        [1.23394576e-04, 9.82013790e-01]])

    f = Linear(inputs, weights, bias)
    g = Sigmoid(f)
    cost = MSE(g)

    feed_dict = {inputs: x, weights: w, bias: b}
    gradients = forward_and_backward(feed_dict, ideal_output, [weights, bias])

    """
    You should see a list of gradients on the weights and bias that looks like:
    [array([[  6.08973702e-08,   6.93800843e-02],
        [  1.21794740e-07,   1.38760169e-01]]), array([ -6.08973702e-08,  -6.93800843e-02])]
    """
    print(np.array([[  6.08973702e-08,   6.93800843e-02], [  1.21794740e-07,   1.38760169e-01]]), np.array([ -6.08973702e-08,  -6.93800843e-02]))
    print(gradients)

if __name__ == "__main__":
    # simple layer
    # test_layer()

    # # activation
    # test_sigmoid()

    # test_MSE()

    test_backprop()



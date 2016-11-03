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

    graph = topological_sort(feed_dict)
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




if __name__ == "__main__":
    # simple layer
    test_layer()

    # activation
    test_sigmoid()

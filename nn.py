"""
This script builds and runs a graph.

"""

from graph import *


def test_base_classes():
    x, y = Input(), Input()

    f = Add(x, y)
    f2 = Add(f, y)
    f_many = Add(x, y, f, f2)
    f_mul = Mul(x, y)

    feed_dict = {x: 10, y: 5}

    sorted_neurons = topological_sort(feed_dict)

    output = forward_pass(f, sorted_neurons)
    print("{} + {} = {} (according to graph)".format(feed_dict[x], feed_dict[y], output))

    output = forward_pass(f2, sorted_neurons)
    print("({} + {}) + {} = {} (according to graph)".format(feed_dict[x], feed_dict[y], feed_dict[y], output))

    output = forward_pass(f_many, sorted_neurons)
    print("result for many: 10 + 5 + 15 + 20 =", output)

    output = forward_pass(f_mul, sorted_neurons)
    print("{} * {} = {} (according to graph)".format(feed_dict[x], feed_dict[y], output))


def test_linear():
    x, y, z = Input(), Input(), Input()
    inputs = [x, y, z]

    weight_x, weight_y, weight_z = Input(), Input(), Input()
    weights = [weight_x, weight_y, weight_z]

    bias = Input()

    f = Linear(inputs, weights, bias)

    feed_dict = {
        x: 6,
        y: 14,
        z: 3,
        weight_x: 0.5,
        weight_y: 0.25,
        weight_z: 1.4,
        bias: 2
    }

    graph = topological_sort(feed_dict)
    output = forward_pass(f, graph)

    print("with this example should be 12.7 ==", output, output == 12.7)

test_base_classes()
test_linear()

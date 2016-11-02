"""
This script builds and runs a graph.

"""

from graph import *

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


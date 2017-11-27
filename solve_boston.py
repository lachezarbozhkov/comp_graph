from graph.layers import *
import numpy as np
from sklearn.datasets import load_boston

np.random.seed(1)

boston = load_boston()
labels = boston.target.reshape(-1,1)
features = boston.data

inputs, weights, bias = Input(), Input(trainable=True), Input(trainable=True)
f = Linear(inputs, weights, bias)
cost = MSE(f)
# g = Sigmoid(f)
# weights2, bias2 = Input(True), Input(True)
# f2 = Linear(g, weights2, bias2)
# # g2 = Sigmoid(f2)
# cost = MSE(f2)

x = features
w = np.random.rand(x.shape[1], 1)
b = np.random.rand(1)

# w2 = np.random.rand(w.shape[1], 1)
# b2 = np.random.rand(1)

feed_dict = {inputs: x, weights: w, bias: b}

# feed_dict = {inputs: x, weights: w, bias: b,  weights2: w2, bias2: b2}
print(x)
print(w)
print(b)
print(labels)

train_SGD(feed_dict, labels, 10, learning_rate=0.1)
print(x)
print(w)
print(b)
print(labels)

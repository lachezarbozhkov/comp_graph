from graph.layers import * 

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer

iris = datasets.load_iris()
x = iris.data
y = iris.target

np.random.seed(0)

labelBinarizer = LabelBinarizer()
y = labelBinarizer.fit_transform(y)
random.seed(1)


inputs, weights, bias = Input(name="x"), Input(trainable=True, name="weights 1"), Input(trainable=True, name="bias 1")
f = Linear(inputs, weights, bias, "Logit 1")
g = Sigmoid(f, name="Sigmoid 1")


weights2, bias2 = Input(trainable=True, name="weights 2"), Input(trainable=True, name="bias 2")
f2 = Linear(g, weights2, bias2, "Logit 2")
g2 = Sigmoid(f2, name="Sigmoid 2")
cost = MSE(g2)

# weights3, bias3 = Input(trainable=True, name="weights 3"), Input(trainable=True, name="bias 3")
# f3 = Linear(g2, weights3, bias3, "Logit 3")
# g3 = Sigmoid(f3, name="Sigmoid 3")
# cost = MSE(g3)


input_shape = x.shape[1]
w = np.random.rand(input_shape, input_shape)
b = np.random.rand(input_shape)

w2 = np.random.rand(w.shape[1], 3)
b2 = np.random.rand(3)

# w3 = np.random.rand(w2.shape[1], 3)
# b3 = np.random.rand(3)

feed_dict = {inputs: x, weights: w, bias: b, weights2: w2, bias2: b2}

ideal_output = y

train_SGD(feed_dict, ideal_output, 1000, learning_rate=0.01)
correct = np.argmax(g2.value, axis=1) == np.argmax(ideal_output, axis=1)
print("accuracy:", correct.mean())

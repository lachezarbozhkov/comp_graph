from functools import reduce
import operator


class Neuron:
    def __init__(self, inbound_neurons=[]):
        # Neurons from which this Node receives values
        self.inbound_neurons = inbound_neurons
        # Neurons to which this Node passes values
        self.outbound_neurons = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_neurons` and
        store the result in self.value.
        """
        raise NotImplemented

    def backward(self):
        """
        Backward propagation.

        Compute the gradient of the current node with respect
        to the input neurons. The gradient of the loss with respect
        to the current neuron should already be computed in the `gradients`
        attribute of the output neurons.
        """
        raise NotImplemented


class Input(Neuron):
    def __init__(self):
        # an Input neuron has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Neuron.__init__(self)

    # NOTE: Input node is the only node where the value
    # is passed as an argument to forward().
    #
    # All other neuron implementations should get the value
    # of the previous neurons from self.inbound_neurons
    #
    # Example:
    # val0 = self.inbound_neurons[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value:
            self.value = value


class Add(Neuron):
    def __init__(self, *args):
        Neuron.__init__(self, list(args))

    def forward(self):
        """
        Set the value of this neuron to the sum of it's inbound_nodes.
        """
        self.value = sum([neuron.value for neuron in self.inbound_neurons])


class Mul(Neuron):
    def __init__(self, *args):
        Neuron.__init__(self, list(args))

    def forward(self):
        """
        Set the value of this neuron to the product of it's inbound_nodes.
        """
        self.value = reduce(operator.mul, [neuron.value for neuron in self.inbound_neurons])


class Linear(Neuron):
    def __init__(self, inputs, weights, bias):
        Neuron.__init__(self, inputs)
        self.weights = weights
        self.bias = bias

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        """
        self.value = sum([x.value * w.value for x, w in zip(self.inbound_neurons, self.weights)])
        self.value += self.bias.value


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_neurons = [n for n in feed_dict.keys()]

    G = {}
    neurons = [n for n in input_neurons]
    while len(neurons) > 0:
        n = neurons.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_neurons:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            neurons.append(m)

    L = []
    S = set(input_neurons)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_neurons:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_neuron, sorted_neurons):
    """
    Performs a forward pass through a list of sorted neurons.

    Arguments:

        `output_neuron`: A neuron in the graph, should be the output neuron (have no outgoing edges).
        `sorted_neurons`: a topologically sorted list of neurons.

    Returns the output neuron's value
    """

    for n in sorted_neurons:
        n.forward()

    return output_neuron.value
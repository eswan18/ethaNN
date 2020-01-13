import abc

import numpy as np

from activations import activation_from_name_or_function


class Layer(abc.ABC):
    @abc.abstractmethod
    def transform(sample):
        '''Transform an input.'''

    def show_neurons(self):
        # Iterate over the rows of the weights matrix, showing that each one is
        # a "neuron".
        s = ''
        for i, vector in enumerate(self.weights):
            if i != 0:
                s += '--------------------\n'
            s += f'Neuron #{i}:\n'
            s += f'weights: {vector}\n'
            s += f'bias: {self.biases[i]}\n'
            s += f'activation: {self.activation}\n'
        return s

    def __repr__(self):
        class_name = type(self).__name__
        n_in, n_out, act = self.n_input, self.n_output, self.activation
        return (f'{class_name}(n_input={n_in}, n_output={n_out}, '
                f'activation={act.__name__})')

class DenseLayer(Layer):
    def __init__(self, activation, dim=None, n_input=None, n_output=None):
        # The user needs to either pass in dim or *both* n_input and n_output.
        if dim is None and (n_input is None or n_output is None):
            msg = ('Layer init method requires either dim or *both* n_input '
                   'and n_output to be specified')
            raise ValueError(msg)
        # If the user paseed in dim, unpack it so we know n_input and n_output.
        if dim is not None:
            n_output, n_input = dim
        self.n_output = n_output
        self.n_input = n_input
        # Initialize w and b with values in [0, 1).
        self.weights = np.random.rand(n_output, n_input)
        self.biases = np.random.rand(n_output)
        # Outsource to the activations module the heavy lifting of parsing the
        # user's specified activation.
        self.activation = activation_from_name_or_function(activation)

    def transform(sample):
        '''Transform and input.'''
        raw = np.matmul(self.weights, self.sample) + self.biases
        transformed = self.activation(raw)
        return transformed

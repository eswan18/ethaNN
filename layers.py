import abc

import numpy as np

from activations import activation_from_name_or_function


class Layer(abc.ABC):
    @abc.abstractmethod
    def predict(sample):
        '''Output a prediction, given an input.'''

class Dense(Layer):
    def __init__(self, activation, dim=None, n_input=None, n_output=None):
        # The user needs to either pass in dim or *both* n_input and n_output.
        if dim is None and (n_input is None or n_output is None):
            msg = ('Layer init method requires either dim or *both* n_input '
                   'and n_output to be specified')
            raise ValueError(msg)
        # If the user paseed in dim, unpack it so we know n_input and n_output.
        if dim is not None:
            n_output, n_input = dim
        # Initialize w and b with values in [0, 1).
        self.weights = np.random.rand(n_output, n_input)
        self.biases = np.random.rand(n_output)
        # Outsource to the activations module the heavy lifting of parsing the
        # user's specified activation.
        self.activation = activation_from_name_or_function(activation)

    def predict(sample):
        '''Output a prediction, given an input.'''
        raw = np.matmul(self.weights, self.sample) + self.biases
        prediction = self.activation(raw)
        return prediction

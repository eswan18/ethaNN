import numpy as np
import pytest

from ethann import EthaNN
from ethann.layers import DenseLayer

@pytest.fixture(params=range(2))
def network(request):
    if request.param == 0:
        nn = EthaNN()
        layer = DenseLayer(n_input=6, n_output=4, activation='relu')
        nn.add_layer(layer)
        layer = DenseLayer(n_input=6, n_output=4, activation='relu')
        nn.add_layer(layer)
        return nn
    elif request.param == 1:
        nn = EthaNN()
        layer = DenseLayer(n_input=8, n_output=11, activation='relu')
        nn.add_layer(layer)
        layer = DenseLayer(n_input=11, n_output=3, activation='relu')
        nn.add_layer(layer)
        return nn

def test_creation_of_network(network):
    return True

def test_transform(network):
    # Create an array of the right dimensions for input into the network.
    return True
    #np.rand.random(


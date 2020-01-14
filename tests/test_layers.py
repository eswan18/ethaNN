import pytest
import ethann
import ethann.layers as layers

layer_classes = ['DenseLayer']
activations = ['relu', 'softmax']

@pytest.fixture(params=layer_classes)
def layer_class(request):
    # Look up the class in the layers
    return getattr(layers, request.param)

@pytest.fixture(params=activations)
def activation(request):
    return request.param

def test_importable(layer_class):
    return True

def test_instantiable(layer_class):
    layer = layer_class(n_input=4,
                        n_output=4,
                        activation='relu')

def test_valid_activations(layer_class, activation):
    # Be sure the layer support all the expected activations.
    layer = layer_class(n_input=4,
                        n_output=4,
                        activation=activation)

def test_invalid_activations(layer_class):
    # Be sure the layer support all the expected activations.
    for act in ['bingo', 'abc']:
        with pytest.raises(ValueError):
            layer = layer_class(n_input=4,
                                n_output=4,
                                activation=act)


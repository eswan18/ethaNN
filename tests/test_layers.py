import pytest
import ethann
import ethann.layers as layers

layer_classes = ['DenseLayer']

@pytest.fixture(params=layer_classes)
def layer_class(request):
    # Look up the class in the layers
    return getattr(layers, request.param)

def test_importable(layer_class):
    return True

def test_instantiable(layer_class):
    layer = layer_class(n_input=4,
                        n_output=4,
                        activation='relu')

import pytest
import ethann
import ethann.layers as l

layer_classes = ['DenseLayer']
@pytest.fixture(params=layer_classes)
def layer_class(request):
    eval(f'from ethann.layers import {request.param}')
    eval(f'x = {request.param}')
    return x


def test_dense():
    return True

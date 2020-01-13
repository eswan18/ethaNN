from ethann import EthaNN
from layers import DenseLayer

nn = EthaNN()
layer = DenseLayer(n_input=6, n_output=4, activation='relu')
print(layer.show_neurons())
nn.add_layer(layer)
layer = DenseLayer(n_input=6, n_output=4, activation='relu')
nn.add_layer(layer)
print(repr(nn))

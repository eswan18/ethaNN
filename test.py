from ethann import EthaNN
from layers import Dense

nn = EthaNN()
layer = Dense(n_input=6, n_output=4, activation='relu')
print(layer.show_neurons())
nn.add_layer(layer)

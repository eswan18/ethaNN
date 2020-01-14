import copy

class EthaNN:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        # Always store a copy of the input to avoid having multiple copies of
        # the same object in the layers list.
        copied = copy.copy(layer)
        self.layers.append(layer)

    def add_layers(self, layers):
        for layer in layers:
            # Always store a copy of the input to avoid having multiple copies
            # of the same object in the layers list.
            copied = copy.copy(layer)
            self.layers.append(layer)

    def transform(self, sample):
        # Run the input through each layer successively.
        current = sample
        for layer in self.layers:
            current = layer.transform(current)
        final = current
        return final

    def __repr__(self):
        class_name = type(self).__name__
        layer_repr = ',\n\t'.join([repr(layer) for layer in self.layers])
        return class_name + '(\n\t' + layer_repr + '\n)'


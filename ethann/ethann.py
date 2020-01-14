class EthaNN:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_layers(self, layers):
        self.layers += layers

    def transform(self, sample):
        raise NotImplementedError

    def __repr__(self):
        class_name = type(self).__name__
        layer_repr = ',\n\t'.join([repr(layer) for layer in self.layers])
        return class_name + '(\n\t' + layer_repr + '\n)'


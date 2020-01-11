class EthaNN:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_layers(self, layers):
        self.layers += layers

    def predict(self, sample):
        raise NotImplementedError

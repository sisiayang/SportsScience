from keras import layers

class CNN_with_mask(layers.Layer):
    def __init__(self, filters, kernel_size, strides, name):
        super().__init__()
        self.con_layer = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', name=name)

    def call(self, input, training=None, mask=None):
        output = self.con_layer(input)
        return output

    def compute_mask(self, inputs, mask=None):
        return mask
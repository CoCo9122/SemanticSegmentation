import tensorflow as tf
import tensorflow_addons as tfa

from .layers import *

class VGG16:
    """
    VGG16.

    parameter:
      input_shape: (int, int, int)
        Input size and channels.
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape

        self.model = self.create_model()

    def call(self, inputs, ):

        x = inputs

        for i in range(2):
            x = Conv2D(x, 64, (3, 3), (1, 1))
            x = ReLU(x)
        x = Maxpooling2D(x, [2, 2], (2, 2))

        for i in range(2):
            x = Conv2D(x, 128, (3, 3), (1, 1))
            x = ReLU(x)
        x = Maxpooling2D(x, [2, 2], (2, 2))

        for i in range(3):
            x = Conv2D(x, 256, (3, 3), (1, 1))
            x = ReLU(x)
        x = Maxpooling2D(x, [2, 2], (2, 2))

        for i in range(3):
            x = Conv2D(x, 512, (3, 3), (1, 1))
            x = ReLU(x)
        x = Maxpooling2D(x, [2, 2], (2, 2))

        for i in range(3):
            x = Conv2D(x, 512, (3, 3), (1, 1))
            x = ReLU(x)
        x = Maxpooling2D(x, [2, 2], (2, 2))

        x = Conv2D(x, 4096, (7, 7), (1, 1))
        x = ReLU(x)
        x = Dropout(x, 0.5)
        x = Conv2D(x, 4096, (1, 1), (1, 1))
        x = ReLU(x)
        self.outputs = Dropout(x, 0.5)

        return self.outputs
    
    def create_model(self,):
        self.inputs = tf.keras.Input(shape=self.input_shape)
        return tf.keras.Model(inputs=[self.inputs], outputs=self.call(self.inputs), name='VGG-16')

    def get_model(self, ):
        return self.model

    def get_input(self, ):
        return self.inputs

    def get_output(self, ):
        return self.outputs
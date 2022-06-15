import tensorflow as tf
import numpy as np

class Sigmoid (tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

    def function(self, x):
        return 1/(1 + np.exp(-x))

    def call(self, inputs):
        return self.function(inputs)

    def get_config(self, ):
        return super(Sigmoid, self).get_config()
        
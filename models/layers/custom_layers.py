import tensorflow as tf
import numpy as np

class CustomSigmoid (tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomSigmoid, self).__init__(**kwargs)

    def function(self, x):
        return 1/(1 + tf.exp(-x))

    def call(self, inputs):
        return self.function(inputs)

    def get_config(self, ):
        return super(CustomSigmoid, self).get_config()
        
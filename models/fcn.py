import tensorflow as tf
import tensorflow_addons as tfa

from .vgg16 import VGG16
from .layers import *

class FCN32s:
    def __init__(self, input_shape, class_num):
        self.input_shape = input_shape
        self.class_num = class_num
    
    def call(self, inputs):
        x = Conv2D(inputs, self.class_num, (1, 1), (1, 1))
        x = Conv2DTranspose(x, self.class_num, (64, 64), (32, 32))
        x = Sigmoid(x)
        return x

    def get_model(self,):
        vgg16 = VGG16(self.input_shape)
        return tf.keras.Model(inputs=[vgg16.get_input()], outputs=self.call(vgg16.get_output()), name='FCN-32s')

import tensorflow as tf
import tensorflow_addons as tfa

import custom_layers

def Add(inputs, ):
    """
    Layer that adds a list of inputs.

    parameter:
      inputs: tf.keras.layer and list
        The layer input.
    """
    return tf.keras.layers.Add()(inputs)

def AtrousConv2D(inputs, c, ks, s, dr, p='same', ki = 'he_normal', ub = False):
    """
    A simple Conv2D layer.

    parameter:
      inputs: tf.keras.layers
        The layer input.
      c: int
        Output channels.
      ks: int or tuple(2 dimensions)
        It is kernel_size.
      s: int or tuple(2 dimensions)
        strides.
      dr: int or tuple(2 dimensions)
        Dilation (Atrous) rate.
      p: string
        Default padding is 'same'.
      ki: string
        default kernel initializer is 'he_normal'.
      ub: boolean
        If you use use bias, ub = Ture.

    return:
      AtrousConv2D:
        tf.keras.Layer
    """

    return tf.keras.layers.Conv2D(filters=c, kernel_size=ks, strides=s, dilation_rate=dr, padding=p, kernel_initializer=ki, use_bias=ub)(inputs)

def BatchNormalization(inputs, ):
    """
    Layer that normalizes its inputs.

    paramater:
      inputs: tf.keras.layers
        The layer input.
    """
    return tf.keras.layers.BatchNormalization()(inputs)

def Concatenate(inputs, axis):
    """
    Layer that concatenates a list of inputs.

    papameter:
      inputs: tf.keras.layer and Array
        The layer input.
      axis: int
        Axis along which to concatenate.
    """
    return tf.keras.layers.concatenate(inputs, axis)

def Conv2D(inputs, c, ks, s, p='same', ki = 'he_normal', ub = False):
    """
    A simple Conv2D layer.

    parameter:
      inputs: tf.keras.layers
        The layer input.
      c: int
        Output channels.
      ks: int or tuple(2 dimensions)
        It is kernel_size.
      s: int or tuple(2 dimensions)
        strides.
      p: string
        Default padding is 'same'.
      ki: string
        default kernel initializer is 'he_normal'.
      ub: boolean
        If you use use bias, ub = Ture.

    return:
      Conv2D:
        tf.keras.Layer
    """
    return tf.keras.layers.Conv2D(filters=c, kernel_size=ks, strides=s, padding=p, kernel_initializer=ki, use_bias=ub)(inputs)

def Dropout(inputs, r):
    """
    Applies Dropout to the input.
      
      parameter:
        inputs: tf.keras.layers
          The layer input.
        r: float
          Float between 0 and 1. Fraction of the input units to drop.
    """
    return tf.keras.layers.Dropout(r)(inputs)

def Maxpooling2D(inputs, ps, s, p='same'):
    """
    Max pooling operation for 2D spatial data.

    parameter:
      inputs:tf.keras.layer
        The layer input.
      ps: int or tuple(2 dimensions)
        integer or tuple of 2 integers, window size over which to take the maximum.
      s: int
        Integer, tuple of 2 integers, or None. Strides values. Specifies how far the pooling window moves for each pooling step. If None, it will default to pool_size.
      p: string
        One of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
    """
    return tf.keras.layers.MaxPool2D(pool_size=ps, strides=s, padding=p)(inputs)

def ReLU(inputs, ):
    """
    Rectified Linear Unit activation function.
    
    paramater:
      inputs: tf.keras.layers
        The layer input.
    """
    return tf.keras.layers.ReLU()(inputs)

def Sigmoid(inputs, ):
    """
    Sigmoid activation function.

    paramater:
      inputs: tf.keras.layers
        The layer input.
    """
    return custom_layers.Sigmoid()(inputs)
import tensorflow as tf
import tensorflow_addons as tfa

def DiceCoefficient(true, pred, smooth=1):
    true_flatten = tf.reshape(true, [-1])
    pred_flatten = tf.reshape(pred, [-1])
    return (2*tf.reduce_sum(true_flatten*pred_flatten) + smooth) / (tf.reduce_sum(true_flatten) + tf.reduce_sum(pred_flatten) + smooth)

def IoU(true, pred):
    i = tf.reduce_sum(pred*true, axis=(1, 2))
    u = tf.reduce_sum(pred+true, axis=(1, 2)) - i
    return tf.reduce_mean(i/u)
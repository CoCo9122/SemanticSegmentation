import tensorflow as tf
import tensorflow_addons as tfa

from .metrics import DiceCoefficient 

def BinaryCrossEntropy(true, pred, from_logits=False):
    return tf.keras.losses.binary_crossentropy(true, pred, from_logits=from_logits)

def DiceLoss(true, pred):
    return 1 - DiceCoefficient(true, pred)

def BceDiceLoss(true, pred):
    return BinaryCrossEntropy(true, pred) + DiceLoss(true, pred)


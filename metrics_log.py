import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from loss import *

class MetricsLog:
    def __init__(self, metrics, training=True):
        if training: s='Train' 
        else: s='Test'
        self.metrics = {
            '{} Recall'.format(s): [tf.keras.metrics.Recall(), None],
            '{} Precision'.format(s): [tf.keras.metrics.Precision(), None],
            '{} Dice Coefficient'.format(s): [tf.keras.metrics.Mean(), DiceCoefficient],
            '{} IoU'.format(s): [tf.keras.metrics.Mean(), IoU]
        }

        self.metrics_logs = {
            '{} Recall'.format(s): [],
            '{} Precision'.format(s): [],
            '{} Dice Coefficient'.format(s): [],
            '{} IoU'.format(s): [],
        }

        self.metrics.update(metrics)
        self.metrics_logs.update({k:[] for k in metrics.keys()})


    def log(self, true, pred):
        for k, v in self.metrics.items():
            if v[-1] == None:
                v[0].update_state(true, pred)
            else:
                v[0].update_state(v[-1](true, pred))
        self.result()
        self.reset_states()
    
    def result(self, ):
        for k, v in self.metrics.items():
            self.metrics_logs[k].append(np.array(v[0].result()))

    def reset_states(self, ):
        for k, v in self.metrics.items():
            v[0].reset_states()

    def get_log(self, ):
        return self.metrics_logs
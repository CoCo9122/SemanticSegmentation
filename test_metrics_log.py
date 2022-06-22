import unittest
import tensorflow as tf
import tensorflow_addons as tfa

from metrics_log import *
from loss import *

class TestMetricsLog(unittest.TestCase):

    def test_metricstreinlog(self, ):
        metrics = {
            'Train Bce Dice Loss': [tf.keras.metrics.Mean(), BceDiceLoss],
            'Dice Loss': [tf.keras.metrics.Mean(), DiceLoss],
            'BinaryCrossEntropy': [tf.keras.metrics.Mean(), BinaryCrossEntropy],
        }
        train_log = MetricsTrainLog(metrics)
        self.assertEqual(
            {
                'Train Recall': [],
                'Train Precision': [],
                'Train Dice Coefficient': [],
                'Train IoU': [],
                'Train Bce Dice Loss': [],
                'Dice Loss': [],
                'BinaryCrossEntropy': []
            },
            train_log.get_log()
        )
        input_shape = (4, 28, 28, 3)
        true = tf.ones(input_shape)
        pred = tf.random.uniform(input_shape)
        train_log.log(true, pred)
        self.assertEqual(
            [1, 1, 1, 1, 1, 1, 1],
            [len(v) for v in train_log.get_log().values()]
        )
        true = tf.ones(input_shape)
        pred = tf.ones(input_shape)
        train_log.log(true, pred)
        self.assertEqual(
            [1, 1, 1, 1, 0, 0, 0],
            [v[-1] for v in train_log.get_log().values()]
        )

if __name__ == '__main__':
    unittest.main()
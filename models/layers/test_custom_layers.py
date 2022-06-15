import unittest
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from custom_layers import *

class TestCustomLayers(unittest.TestCase):

    def test_sigmoid(self, ):
        x = tf.constant([-20, -1.0, 0.0, 1.0, 20], dtype = tf.float32)
        y = CustomSigmoid()(x)
        self.assertEqual(np.array([2.0611537e-09, 2.6894143e-01, 5.0000000e-01, 7.3105860e-01, 1.0000000e+00]).all(), y.all())

if __name__ == '__main__':
    unittest.main()

import unittest
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import layers

class TestLayers (unittest.TestCase):

    def test_add(self, ):
        input_shape = (2, 3, 4)
        x1 = tf.random.normal(input_shape)
        x2 = tf.random.normal(input_shape)
        y = layers.Add([x1, x2])
        self.assertEqual(input_shape, y.shape)

    def test_atrousconv2d(self, ):
        input_shape = (4, 28, 28, 3)
        x = tf.random.normal(input_shape)
        y = layers.AtrousConv2D(x, 6, (3, 3), 1, 2)
        self.assertEqual((4, 28, 28, 6), y.shape)

    def test_batchnormalization(self, ):
        input_shape = (4, 28, 28, 3)
        x = tf.random.normal(input_shape)
        y = layers.BatchNormalization(x)
        self.assertEqual((4, 28, 28, 3), y.shape)

    def test_concatenate(self, ):
        x1 = np.arange(20).reshape(2, 2, 5)
        x2 = np.arange(20, 30).reshape(2, 1, 5)
        y = layers.Concatenate([x1, x2], 1)
        self.assertEqual((2, 3, 5), y.shape)

    def test_conv2d(self, ):
        input_shape = (4, 28, 28, 3)
        x = tf.random.normal(input_shape)
        y = layers.Conv2D(x, 32, (3, 3), 2)
        self.assertEqual((4, 14, 14, 32), y.shape)

    def test_dropout(self, ):
        input_shape = (4, 28, 28, 3)
        x = tf.random.normal(input_shape)
        y = layers.Dropout(x, 0.3)
        self.assertEqual((4, 28, 28, 3), y.shape)

    def test_maxpooling2d(self, ):
        x = tf.constant([[1., 2., 3., 4.],
                        [5., 6., 7., 8.],
                        [9., 10., 11., 12.],
                        [13., 14., 15., 16.]])
        x = tf.reshape(x, [1, 4, 4, 1])
        y = layers.Maxpooling2D(x, (2, 2), 2)
        self.assertEqual((1, 2, 2, 1), y.shape)
        self.assertEqual(np.array([[[[6.0], [8.0]], [[14.], [16.]]]]).all(), y.numpy().all())

    def test_relu(self, ):
        x = np.array([-3.0, -1.0, 0.0, 2.0]).reshape(2, 2)
        y = layers.ReLU(x)
        self.assertEqual((2, 2,), y.shape)
        self.assertEqual(np.array([[0.0, 0.0], [0.0, 2.0]]).all(), y.numpy().all())

    def test_sigmoid(self, ):
        x = tf.constant([-20, -1.0, 0.0, 1.0, 20], dtype = tf.float32)
        y = layers.Sigmoid(x)
        self.assertEqual(np.array([2.0611537e-09, 2.6894143e-01, 5.0000000e-01, 7.3105860e-01, 1.0000000e+00]).all(), y.all())

if __name__ == '__main__':
    unittest.main()
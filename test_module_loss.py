import unittest
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from loss import *

class TestLoss(unittest.TestCase):

    def test_binarycrossentropy(self, ):
        y_true = [0, 1, 0, 0]
        y_pred = [-18.6, 0.51, 2.94, -12.8]
        print(BinaryCrossEntropy(y_true, y_pred, from_logits=True).numpy())
        self.assertAlmostEqual(0.865458, BinaryCrossEntropy(y_true, y_pred, from_logits=True).numpy())

    def test_diceloss(self,):
        y_true = [0., 1., 0., 0.]
        y_pred = [1., 0.9, 0.1, 0.5]
        print(DiceLoss(y_true, y_pred).numpy())
        self.assertAlmostEqual(0.3777778, DiceLoss(y_true, y_pred).numpy())
        y_true = [0., 1., 0., 0.]
        y_pred = [0., 1., 0., 0.]
        print(DiceLoss(y_true, y_pred).numpy())
        self.assertEqual(0, DiceLoss(y_true, y_pred).numpy())

    def test_bcediceloss(self ):
        y_true = [0., 1., 0., 0.]
        y_pred = [1., 0.9, 0.1, 0.5]
        print(BceDiceLoss(y_true, y_pred).numpy())
        self.assertAlmostEqual(4.4370546, BceDiceLoss(y_true, y_pred).numpy())
        y_true = [0., 1., 0., 0.]
        y_pred = [0., 1., 0., 0.]
        print(BceDiceLoss(y_true, y_pred).numpy())
        self.assertEqual(0, BceDiceLoss(y_true, y_pred).numpy())

    def test_dicecoefficient(self,):
        y_true = [0., 1., 0., 0.]
        y_pred = [1., 0.9, 0.1, 0.5]
        print(DiceCoefficient(y_true, y_pred).numpy())
        self.assertAlmostEqual(0.6222222, DiceCoefficient(y_true, y_pred).numpy())
        y_true = [0., 1., 0., 0.]
        y_pred = [0., 1., 0., 0.]
        print(DiceCoefficient(y_true, y_pred).numpy())
        self.assertEqual(1, DiceCoefficient(y_true, y_pred).numpy())

    def test_iou(self, ):
        input_shape = (4, 28, 28, 3)
        y_true = tf.ones(input_shape)
        y_pred = tf.random.uniform(input_shape)
        print(IoU(y_true, y_pred).numpy())
        self.assertLessEqual(0, IoU(y_true, y_pred).numpy())
        y_true = tf.ones(input_shape)
        y_pred = tf.ones(input_shape)
        print(IoU(y_true, y_pred).numpy())
        self.assertAlmostEqual(1.0, IoU(y_true, y_pred).numpy())


if __name__ == '__main__':
    unittest.main()
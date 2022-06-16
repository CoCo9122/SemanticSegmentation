import unittest
import tensorflow as tf
import tensorflow_addons as tfa

from models import *

class TestModels (unittest.TestCase):

    def test_vgg16(self, ):
        network = VGG16((128, 128, 3))
        model = network.get_model()
        model.summary()
        input = model.get_config()["layers"][0]["config"]["batch_input_shape"]
        self.assertEqual((None, 128, 128, 3), input)

        network = VGG16((64, 64, 1))
        model = network.get_model()
        model.summary()
        input = model.get_config()["layers"][0]["config"]["batch_input_shape"]
        self.assertEqual((None, 64, 64, 1), input)

    def test_fcn32s(self, ):
        network = FCN32s((128, 128, 3), 12)
        model = network.get_model()
        model.summary()
        input = model.get_config()["layers"][0]["config"]["batch_input_shape"]
        self.assertEqual((None, 128, 128, 3), input)

        network = FCN32s((256, 256, 3), 32)
        model = network.get_model()
        model.summary()
        input = model.get_config()["layers"][0]["config"]["batch_input_shape"]
        self.assertEqual((None, 256, 256, 3), input)


if __name__ == '__main__':
    unittest.main()
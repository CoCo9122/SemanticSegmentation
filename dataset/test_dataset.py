import unittest
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import LoadDataSet_Oxford_Pet

class TestDataSet(unittest.TestCase):

    def test_loaddatasetoxfordpet(self, ):
        def display(display_list):
            plt.figure(figsize=(15, 15))
            self.assertEqual(display_list[0].shape, [128, 128, 3])
            self.assertEqual(display_list[1].shape, [128, 128, 1])
            self.assertLessEqual(display_list[0].numpy().max(), 1)
            self.assertLessEqual(display_list[1].numpy().max(), 1 )
            title = ['Input Image', 'Ground Truth']
            for i in range(len(display_list)):
                plt.subplot(1, len(display_list), i+1)
                plt.title(title[i])
                plt.imshow((display_list[i]))
                plt.axis('off')
            plt.show()

        test = LoadDataSet_Oxford_Pet(1000, 32)
        train_batch = test.get_train_batch()
        test_batch = test.get_test_batch()
        for images, masks in train_batch.take(2):
            sample_image, sample_mask = images[0], masks[0]
            display([sample_image, sample_mask])
        for images, masks in test_batch.take(2):
            sample_image, sample_mask = images[0], masks[0]
            display([sample_image, sample_mask])

if __name__ == '__main__':
    unittest.main()
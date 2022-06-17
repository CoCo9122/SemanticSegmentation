import tensorflow_datasets as tfds
import tensorflow as tf

class LoadDataSet_Oxford_Pet:
    def __init__(self, buffer_size, batch_size):

        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size

        dataset, self.info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
        self.train_images = dataset['train'].map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_images = dataset['test'].map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)

    def normalize(self, input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask = tf.ones(input_mask.shape, tf.int32) - tf.cast(tf.math.round(tf.cast(input_mask, tf.float32) / 3), tf.int32)
        return input_image, input_mask

    def load_image(self, datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
        input_image, input_mask = self.normalize(input_image, input_mask)
        return input_image, input_mask

    def get_train_batch(self,):
        return (self.train_images.cache().shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE).repeat().prefetch(buffer_size=tf.data.AUTOTUNE))

    def get_test_batch(self, ):
        return self.test_images.batch(self.BATCH_SIZE)

    def get_info(self, ):
        return self.info
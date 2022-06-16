import tensorflow as tf

def gpu_setting(limit=3):
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    g = len(tf.config.experimental.list_physical_devices('GPU'))
    message = ''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            message = len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
        except RuntimeError as e:
    #   Virtual devices must be set before GPUs have been initialized
            message = e
    return g, message

if __name__ == '__main__':
    gpu_setting()
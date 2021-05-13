import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

IMAGE_SIZE = 224
# Data augmentation here may increase the accuracy in the training results.
# Augmentation parameters should be defined in a config file.
Train_Augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(factor=0.055),
        layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2),
        layers.experimental.preprocessing.RandomTranslation(height_factor=0.05, width_factor=0.05)
    ]
)
# Data augmentation here may increase the accuracy in the training results.
Test_Augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1),
    ]
)

def Get_database(name, shuffle_files=True, batch_size=16):
    # Load my dataset here
    # This functions were created in order to follow the full-pipeline format from TFDS-nightly.
    train_ds = tfds.load(name, split='train', shuffle_files=shuffle_files, batch_size=batch_size)
    test_ds = tfds.load(name, split='test', shuffle_files=shuffle_files, batch_size=batch_size)
    
    train_ds = train_ds.map(lambda data: (Train_Augmentation(data["image"]), data["label"]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = test_ds.map(lambda data: (Test_Augmentation(data["image"]), data["label"]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, test_ds
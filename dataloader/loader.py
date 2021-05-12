import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

# Data augmentation here may increase the accuracy in the training results.
# Augmentation parameters should be defined in a config file.
def Train_Augmentation(image: tf.Tensor):
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = image/255.0 - 0.5
    image = tf.image.resize(image,(224,224))
    # image = tf.image.resize_with_crop_or_pad(image, 224 + 16, 224 + 16) # PAD 16.
    # image = tf.image.random_crop(value=image, size=(224, 224, 3))

    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    image = tf.image.rot90(image, k=np.random.randint(0,4))
    # image = tf.keras.preprocessing.image.random_rotation(image, 20)
    return image

# Data augmentation here may increase the accuracy in the training results.
def Test_Augmentation(image: tf.Tensor):
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = image/255.0 - 0.5
    image = tf.image.resize(image,(224,224))
    return image

def Get_database(name, shuffle_files=True, batch_size=16):
    # Load my dataset here
    # This functions were created in order to follow the full-pipeline format from TFDS-nightly.
    train_ds = tfds.load(name, split='train', shuffle_files=shuffle_files)
    test_ds = tfds.load(name, split='test', shuffle_files=shuffle_files)
    train_ds = train_ds.map(lambda data: (Train_Augmentation(data["image"]), data["label"]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


    test_ds = test_ds.map(lambda data: (Test_Augmentation(data["image"]), data["label"]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


    return train_ds, test_ds
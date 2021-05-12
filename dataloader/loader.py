import tensorflow_datasets as tfds
import tensorflow as tf

# Data augmentation here may increase the accuracy in the training results.
def Demo_Augmentation(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image/255.0 - 0.5
    image = tf.image.resize(image,(224,224))
    image = tf.image.flip_left_right(image)
    # image = tf.image.flip_up_down(image)
    # image = tf.image.resize_with_crop_or_pad(image,20,20)
    image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    return image

def Get_database(name, shuffle_files=True, batch_size=16):
    # Load my dataset here
    # This functions were created in order to follow the full-pipeline format from TFDS-nightly.
    train_ds = tfds.load(name, split='train', shuffle_files=shuffle_files, batch_size=batch_size)
    test_ds = tfds.load(name, split='test', shuffle_files=shuffle_files, batch_size=batch_size)
    train_ds = train_ds.map(lambda data: (Demo_Augmentation(data["image"]), data["label"])).cache()
    test_ds = test_ds.map(lambda data: (Demo_Augmentation(data["image"]), data["label"])).cache()
    return train_ds, test_ds
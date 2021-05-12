import tensorflow as tf
from tensorflow.keras import layers # Dense, GlobalAveragePooling2D

IMG_SIZE=224 # template resolution for EfficientNetB0. Bx would change this value
class EfficientNetB0(tf.keras.Model):
    def __init__(self,classes=1000):
        super(EfficientNetB0, self).__init__()
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        # x = img_augmentation(inputs)
        model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(model.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.75, name="top_dropout")(x)
        outputs = layers.Dense(classes, activation="softmax", name="pred")(x)
        # Compile
        self.model = tf.keras.Model(inputs, outputs, name="EfficientNet-B0")
    def call(self, x):
        return self.model(x)
if __name__ == '__main__':
    # Create an instance of the model
    model = EfficientNetB0()
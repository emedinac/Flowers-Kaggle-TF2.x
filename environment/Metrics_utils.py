import tensorflow as tf

class Losses:
    def __init__(self):
        pass;
    def SetCrossEntropy():
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # loss_function

class Optimizers:
    def __init__(self):
        pass;
    def SetAdam(lr, epochs, decay=True):
        if decay:
            lr = tf.optimizers.schedules.PolynomialDecay(lr, epochs, 1e-5, 2)
        return tf.keras.optimizers.Adam(lr)
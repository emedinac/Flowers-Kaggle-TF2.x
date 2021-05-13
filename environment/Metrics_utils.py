import tensorflow as tf

# Now the system is independent of TF2.x
# Other for analysis and inference

        

class Losses:
    def __init__(self):
        pass;
    def SetCrossEntropy():
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # loss_function
    def SetMean(name):
        return tf.keras.metrics.Mean(name=name)
    def SparseCategoricalAccuracy(name):
        return tf.keras.metrics.SparseCategoricalAccuracy(name=name)

class Optimizers:
    def __init__(self):
        pass;
    def SetAdam(lr, epochs, decay=True):
        if decay:
            lr = tf.optimizers.schedules.PolynomialDecay(lr, epochs, 1e-6, 0.9)
        return tf.keras.optimizers.Adam(lr)
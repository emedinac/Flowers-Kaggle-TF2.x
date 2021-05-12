from .Metrics_utils import *

class Simple_Train:
    def __init__(self):
        self.train_loss = Losses.SetMean('train_loss')
        self.train_accuracy = Losses.SparseCategoricalAccuracy('train_accuracy')
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
    @tf.function
    def reset_states(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
    @tf.function
    def train_step(self, model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = self.loss_function(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

def choose_methodology(option):
    option = option.lower()
    if "simple" in option: return Simple_Train()
    elif "other" in option: return None
    else: None
    return None

if __name__ == '__main__':
    pass; # UnitTest modules
    choose_methodology("simple")

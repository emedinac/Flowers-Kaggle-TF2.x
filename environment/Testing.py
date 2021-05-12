from .Metrics_utils import *

class Simple_Test:
    def __init__(self):
        self.test_loss = Losses.SetMean('test_loss')
        self.test_accuracy = Losses.SparseCategoricalAccuracy('test_accuracy')
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        self.prev_best_loss = 10000.
        self.loss_updated = False
    @tf.function
    def reset_states(self):
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
    @tf.function
    def test_step(self, model, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = self.loss_function(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
    def Update_best_loss(self):
        # Comparison steps
        if self.test_loss.result()<self.prev_best_loss:
            self.loss_updated = True
            self.prev_best_loss = self.test_loss.result().numpy()
        else:
            self.loss_updated = False

def choose_methodology(option):
    option = option.lower()
    if "simple" in option: return Simple_Test()
    elif "other" in option: return None
    else: None
    return None

if __name__ == '__main__':
    pass; # UnitTest modules
    choose_methodology("simple")


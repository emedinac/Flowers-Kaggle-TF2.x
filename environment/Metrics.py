import tensorflow as tf

# Save and storage information using tensorboard
# This logger should be completed later
class Logger:
    def __init__(self, log_file='logdir', epoch=0):
        self.summary_writer = tf.summary.create_file_writer(log_file)
        self.epoch = epoch
    def SaveScalar(self, name, value):
        with self.summary_writer.as_default():
            tf.summary.scalar(name,value,self.epoch)
    def UpdateEpoch(self):
        self.epoch = self.epoch+1
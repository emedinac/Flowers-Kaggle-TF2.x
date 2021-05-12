import dataloader
import models as m
from environment import Metrics as metric
from environment import Training as train
from environment import Testing as test

# Config arguments or configuration file
Epochs = 20
classes=2
eval_each_num_epochs = 5
# ...

# Training environment configuration
train_stage = train.choose_methodology("simple")
# Testing environment configuration
test_stage = test.choose_methodology("simple")
# Load dataset created in TFDS
train_ds, test_ds = dataloader.Get_database(name="dataset")
# Load DEMO model
model = m.EfficientNetB0(classes=classes) # User-defined

storage = metric.Logger(log_file='logdir')
for epoch in range(Epochs):
	pass;

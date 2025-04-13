# Dependencies
# !pip install -q mediapipe-model-maker

# Imports
from mediapipe_model_maker import gesture_recognizer
import tensorflow as tf
import matplotlib.pyplot as plt

# Check TF version
assert tf.__version__.startswith('2')

# Load Dataset
data = gesture_recognizer.Dataset.from_folder(
    dirname='/data',
    split_ratio=0.8,
    shuffle=True,
    random_seed=42
)

train_data = data['train']
val_data = data['validation']

# Train the model
hparams = gesture_recognizer.HParams(export_dir='exported_model')
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=val_data,
    hparams=hparams
)

# Evaluate the model
loss, acc = model.evaluate(val_data)
print(f"Validation Accuracy: {acc * 100:.2f}%")

# Export the model
model.export_model()

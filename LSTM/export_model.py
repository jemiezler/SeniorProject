import tensorflow as tf
from keras.losses import MeanSquaredError

# Load the .h5 model
model = tf.keras.models.load_model(
    "model.h5",
    custom_objects={"mse": MeanSquaredError()}
)
# Export as SavedModel format
model.export("exported_model")  # This creates a directory with the model files

print("Model exported successfully as SavedModel format!")
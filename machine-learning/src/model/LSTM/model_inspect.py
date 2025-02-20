import tensorflow as tf
from keras.saving import register_keras_serializable

# Register the missing function
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load the model with custom objects
model = tf.keras.models.load_model("lstm.h5", custom_objects={"mse": mse})

# Print model summary
model.summary()

# Print input details
print("✅ Model Input Shape:", model.input_shape)
print("✅ Model Input Names:", [layer.name for layer in model.layers])

# Check the first layer to confirm expected features
print("✅ First Layer Config:", model.layers[0].get_config())


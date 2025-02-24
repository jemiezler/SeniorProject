import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from resize_img import resize_with_padding 

# Load your trained model (adjust the path as needed)
model = tf.keras.models.load_model('../../models/best_model.keras')
print("Model loaded successfully.")

# Path to the input image for inference
img_path = '../../dataset/images/8_5_1_2.jpg'

# Load and preprocess the image
image = cv2.imread(img_path)
if image is None:
    raise ValueError("Image not found. Check the path.")
# Convert from BGR (OpenCV default) to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Resize using your function (e.g., target size of 512)
resized_image = resize_with_padding(image, 512)
# Normalize the image to [0, 1]
input_image = resized_image.astype("float32") / 255.0
# Expand dimensions to create a batch of size 1
input_image = np.expand_dims(input_image, axis=0)

# Run inference
predictions = model.predict(input_image)

# Assuming the model outputs logits/probabilities per class for each pixel,
# use argmax to obtain the predicted segmentation mask.
predicted_mask = np.argmax(predictions, axis=-1).squeeze()  # Shape: (512, 512)

# Visualize the input and the predicted mask
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.savefig("output.png")
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask, cmap="jet")
plt.title("Predicted Segmentation Mask")
plt.axis("off")

plt.show()

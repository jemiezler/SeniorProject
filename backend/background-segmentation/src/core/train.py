import os
import keras
from keras import ops
import keras_hub
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from resize_img import process_images
import keras.layers as layers
from tensorflow.keras.callbacks import ModelCheckpoint  # Import ModelCheckpoint

IMAGE_DIR = "../../dataset/images"
MASK_DIR = "../../dataset/masks"

def load_images():
    images, masks = process_images(IMAGE_DIR, MASK_DIR, 512)
    
    if images is None or masks is None or images.size == 0 or masks.size == 0:
        print("Error loading images and masks.")
        raise ValueError("Error: No valid images or masks found!")
    
    images = images.astype("float32") / 255.0
    masks = masks.astype("float32") / 255.0
    indices = np.random.choice(len(images), 3, replace=False)
    plt.figure(figsize=(12, 6))
    for i, index in enumerate(indices):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[index])
        plt.title("Image")
        plt.axis("off")
        
        plt.subplot(2, 3, i + 4)
        plt.imshow(masks[index])
        plt.title("Mask")
        plt.axis("off")
    plt.show()
    return images, masks

def load_model():
    model = keras_hub.models.DeepLabV3ImageSegmenter.from_preset("deeplab_v3_plus_resnet50_pascalvoc")
    output = layers.Resizing(512, 512)(model.output)  # Resize to match masks
    model = keras.Model(inputs=model.input, outputs=output)
    return model

def prepare_dataset(images, masks, batch_size=2):
    """Create a TensorFlow dataset from NumPy arrays."""
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def train_test_split(images, masks, test_size=0.05):
    """Split images and masks into training and testing sets."""
    n = len(images)
    indices = np.random.permutation(n)
    n_test = int(n * test_size)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    train_images, train_masks = images[train_indices], masks[train_indices]
    test_images, test_masks = images[test_indices], masks[test_indices]
    print(f"Train samples: {len(train_images)}")
    print(f"Validation samples: {len(test_images)}")
    
    return train_images, train_masks, test_images, test_masks

def main():
    images, masks = load_images()
    
    # Split the data into train and test sets
    train_images, train_masks, test_images, test_masks = train_test_split(images, masks, test_size=0.05)
    
    # Prepare TensorFlow datasets for training and validation
    train_dataset = prepare_dataset(train_images, train_masks, batch_size=2)
    test_dataset = prepare_dataset(test_images, test_masks, batch_size=2)
    
    # Load and compile the model
    model = load_model()
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam')
    
    # Create ModelCheckpoint callback
    checkpoint = ModelCheckpoint(
        filepath='../../models/best_model.keras',  # Path where the model will be saved
        monitor='val_loss',                    # Monitor the validation loss
        verbose=1,                             # Verbose output on saving
        save_best_only=True,                   # Only save when the model improves
        mode='min'                             # Lower loss is better
    )
    
    # Train the model with the checkpoint callback
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=50,
        callbacks=[checkpoint]
    )
    
    # Optionally, save the final model
    keras.saving.save_model(model, "../../models/deeplabv3_trained.keras")
    print("Training complete!")

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled.")
    
    main()

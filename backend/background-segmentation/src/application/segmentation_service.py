import numpy as np
import cv2
import tensorflow as tf
from src.domain.image_processing import resize_with_padding

class SegmentationService:
    @staticmethod
    def preprocessing_img(image: np.ndarray, target_size=(512, 512)):
        image = self.resize_with_padding(image, target_size[0])
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    @staticmethod
    def segment_img(img: np.ndarray, model: tf.keras.Model):
        img = self.preprocessing_img(img)
        mask = model.predict(img)[0]
        mask = (mask > 0.5).astype(np.uint8) * 255
        return mask
        
    @staticmethod
    def remove_background(img: np.ndarray, mask: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img[:, :, 3] = mask
        return img
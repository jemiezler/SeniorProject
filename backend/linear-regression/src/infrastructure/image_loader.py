import cv2
import numpy as np
from PIL import Image
import io

class ImageLoader:
    """Handles image loading from request."""

    @staticmethod
    def load(image_bytes: bytes) -> np.ndarray:
        """Loads image from bytes and converts to BGR format."""
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)

        # Convert to BGR for OpenCV processing
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

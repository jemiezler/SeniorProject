import os
import cv2
import numpy as np

def resize_with_padding(image, target_size):
    """Resize image while maintaining aspect ratio with padding."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded

def process_images(IMAGE_DIR, MASK_DIR, TARGET_SIZE):
    """Resize all images and masks in a directory while maintaining aspect ratio."""
    resized_images = []
    resized_masks = []
    
    # if not os.path.exists(OUTPUT_IMAGE_DIR):
    #     os.makedirs(OUTPUT_IMAGE_DIR)
    # if not os.path.exists(OUTPUT_MASK_DIR):
    #     os.makedirs(OUTPUT_MASK_DIR)

    # Get all images and masks
    image_paths = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
    mask_paths = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.png')])

    for img_path, mask_path in zip(image_paths, mask_paths):
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale

        # Resize with padding
        resized_image = resize_with_padding(image, TARGET_SIZE)
        resized_mask = resize_with_padding(mask, TARGET_SIZE)
        
        resized_images.append(resized_image)
        resized_masks.append(resized_mask)
        
    return np.array(resized_images), np.array(resized_masks)

# IMAGE_DIR = "../../dataset/images"
# MASK_DIR = "../../dataset/masks"
# OUTPUT_IMAGE_DIR = "../../output/images"
# OUTPUT_MASK_DIR = "../../output/masks"

# process_directory(IMAGE_DIR, MASK_DIR, OUTPUT_IMAGE_DIR, OUTPUT_MASK_DIR)

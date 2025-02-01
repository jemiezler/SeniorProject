import cv2
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def extract_leaves_with_transparency(image_path, output_path):
    try:
        logging.info(f"Processing image: {image_path}")

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return

        logging.info("Image loaded successfully.")

        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        logging.info("Converted image to HSV color space.")

        # Define color ranges for green and yellow hues
        lower_bound_green = np.array([30, 40, 40])  # Adjust for green leaves
        upper_bound_green = np.array([85, 255, 255])  # Adjust for green leaves
        lower_bound_yellow = np.array([20, 40, 40])  # Adjust for yellow leaves
        upper_bound_yellow = np.array([30, 255, 255])  # Adjust for yellow leaves

        # Create masks for green and yellow hues
        mask_green = cv2.inRange(hsv, lower_bound_green, upper_bound_green)
        mask_yellow = cv2.inRange(hsv, lower_bound_yellow, upper_bound_yellow)

        # Combine the masks
        mask = cv2.bitwise_or(mask_green, mask_yellow)

        # Clean up the mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        logging.info("Applied morphological operations to clean the mask.")

        # Create an output with transparency (add alpha channel)
        b, g, r = cv2.split(image)
        alpha = mask  # Use the mask as the alpha channel
        result = cv2.merge([b, g, r, alpha])
        logging.info("Created output image with transparency.")

        # Save the result as PNG
        cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        logging.info(f"Processed and saved image with transparency to: {output_path}")

    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")

def process_folder_with_transparency(input_folder, output_folder):
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logging.info(f"Created output folder: {output_folder}")

        image_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        logging.info(f"Found {len(image_files)} images in the input folder.")

        for filename in image_files:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{filename}.png")
            extract_leaves_with_transparency(input_path, output_path)

        logging.info("All images processed successfully.")

    except Exception as e:
        logging.error(f"Error processing folder: {str(e)}")

# Example usage
if __name__ == "__main__":
    input_folder = "./raw_images"  # Replace with your folder containing input images
    output_folder = "./output/color_based"  # Replace with your desired output folder
    process_folder_with_transparency(input_folder, output_folder)

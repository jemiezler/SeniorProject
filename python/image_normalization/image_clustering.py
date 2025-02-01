import os
import re
import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# -------------------------------------------------------------------------
# 1) Configure Logging
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("process.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 2) Define Functions
# -------------------------------------------------------------------------

def cluster_colors(image, k, random_state):
    """
    Clusters the colors in a BGR image using K-Means, returning:
      - clustered_image (RGB)
      - palette (RGB)
      - dominant_colors (RGB array of shape (k, 3))
    """
    logger.debug("Converting image from BGR to RGB for clustering.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image into a 2D array of RGB values
    pixels = image_rgb.reshape(-1, 3)
    pixels = np.float32(pixels)

    logger.debug("Applying K-Means clustering.")
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(pixels)

    # Extract results
    dominant_colors = np.uint8(kmeans.cluster_centers_)  # shape: (k, 3) in RGB
    labels = kmeans.labels_
    clustered_image = dominant_colors[labels].reshape(image_rgb.shape)

    logger.debug("Building a color palette for the clustered image.")
    palette = np.zeros((50, 300, 3), dtype=np.uint8)
    steps = 300 // k
    for i, color in enumerate(dominant_colors):
        palette[:, i * steps : (i + 1) * steps, :] = color

    return clustered_image, palette, dominant_colors


def rgb_to_hex(r, g, b):
    """Convert RGB values to a #RRGGBB hex string."""
    return f"#{r:02X}{g:02X}{b:02X}"

def remove_gray_background(image, threshold=30):
    """
    Extract non-gray areas from an image by masking gray areas.
    
    Args:
        image: Input image (RGB or BGR).
        threshold: Intensity threshold for detecting gray colors.
        
    Returns:
        An image with gray areas set to black or removed.
    """
    # Convert the image to RGB if it's in BGR format
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    # Calculate the difference between the max and min color channels
    max_channel = np.max(image_rgb, axis=2)
    min_channel = np.min(image_rgb, axis=2)
    diff = max_channel - min_channel

    # Create a mask for non-gray areas
    non_gray_mask = diff >= threshold

    # Create an output image with non-gray areas kept and gray areas set to black
    output_image = np.zeros_like(image_rgb)
    output_image[non_gray_mask] = image_rgb[non_gray_mask]

    return output_image

def remove_background_with_gaussian(image, labels, dominant_colors, blur_radius=15):
    """
    Remove the background using K-Means clustering and refine with Gaussian blur.

    Args:
        image: Original image (RGB).
        labels: Cluster labels for each pixel.
        dominant_colors: RGB values of the clusters.
        blur_radius: Radius for Gaussian blur to smooth edges.

    Returns:
        Image with the largest cluster (background) removed.
    """
    # Identify the largest cluster (assumed to be the background)
    unique, counts = np.unique(labels, return_counts=True)
    background_label = unique[np.argmax(counts)]

    # Create a binary mask for the non-background pixels
    mask = (labels != background_label).astype(np.uint8) * 255  # Foreground: 255, Background: 0
    mask = mask.reshape(image.shape[:2])  # Reshape to 2D

    # Apply Gaussian blur to smooth edges
    blurred_mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

    # Use the blurred mask to blend the original image with a transparent background
    no_background = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):  # Apply the mask to each color channel
        no_background[:, :, c] = image[:, :, c] * (blurred_mask / 255.0)

    return no_background


def process_image(filename, input_folder, output_folder, k, random_state):
    """
    Process a single image:
      - Load it
      - Run K-Means clustering
      - Save the clustered image, palette, and refined background removal.

    Args:
        filename: Name of the image file.
        input_folder: Path to input images.
        output_folder: Path to save processed images.
        k: Number of clusters for K-Means.
        random_state: Seed for reproducibility.
    """
    logger.info(f"Processing file: {filename}")
    image_path = os.path.join(input_folder, filename)
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        logger.warning(f"Could not read image file: {filename}")
        return

    # Resize image for faster processing
    resized_image = cv2.resize(image_bgr, (1920, 1080), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Cluster colors with K-Means
    clustered_image, palette, dominant_colors, labels = cluster_colors(image_rgb, k=k, random_state=random_state)

    # Remove background using Gaussian blur refinement
    refined_background = remove_background_with_gaussian(image_rgb, labels, dominant_colors, blur_radius=15)

    # Save outputs
    base_name, ext = os.path.splitext(filename)
    clustered_path = os.path.join(output_folder, "clustered", f"{base_name}_clustered{ext}")
    palette_path = os.path.join(output_folder, "palette", f"{base_name}_palette{ext}")
    refined_path = os.path.join(output_folder, "refined", f"{base_name}_refined{ext}")

    # Create output directories
    os.makedirs(os.path.join(output_folder, "clustered"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "palette"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "refined"), exist_ok=True)

    cv2.imwrite(clustered_path, cv2.cvtColor(clustered_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(palette_path, cv2.cvtColor(palette, cv2.COLOR_RGB2BGR))
    cv2.imwrite(refined_path, cv2.cvtColor(refined_background, cv2.COLOR_RGB2BGR))

    logger.info(f"Saved clustered image to: {clustered_path}")
    logger.info(f"Saved palette to: {palette_path}")
    logger.info(f"Saved refined background image to: {refined_path}")

def main():
    input_folder = "./raw_images"
    output_folder = "./output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # K-means parameters
    k = 3
    random_state = 42

    # Collect images
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ]
    logger.info(f"Found {len(image_files)} image(s) in '{input_folder}' to process.")

    # We will gather all color data into one list
    all_color_data = []

    # Process images in parallel
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, fn, input_folder, output_folder, k, random_state)
            for fn in image_files
        ]
        for f in as_completed(futures):
            # Each future returns a list of rows for that image
            image_rows = f.result()  # e.g., [[filename, idx, R, G, B, Hex], ...]
            all_color_data.extend(image_rows)

    # --- After all images are processed, export all data to a single CSV ---
    if all_color_data:
        all_colors_csv = os.path.join(output_folder, "all_colors.csv")
        logger.info(f"Writing all color data to {all_colors_csv}")

        with open(all_colors_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write a header row
            writer.writerow(["Filename", "ClusterIndex", "R", "G", "B, Hex"])
            # Write all collected data
            writer.writerows(all_color_data)

        logger.info(f"Finished writing all color data to {all_colors_csv}")
    else:
        logger.info("No color data was collected, so no CSV was written.")

    logger.info("Done with all images!")
    

# -------------------------------------------------------------------------
# 3) Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    
    main()

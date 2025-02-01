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

def calculate_inertia(image, max_k=10, random_state=42):
    """
    Calculate inertia values for a range of k values using K-Means clustering.

    Args:
        image: Input image (BGR).
        max_k: Maximum number of clusters to evaluate.
        random_state: Seed for reproducibility in K-Means.

    Returns:
        A list of inertia values for k=1 to max_k.
    """
    logger.info("Calculating inertia values for Elbow Method...")
    resized_image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    inertia_values = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(pixels)
        inertia_values.append(kmeans.inertia_)

    return inertia_values


def process_image_for_elbow(filename, input_folder, max_k, random_state):
    """
    Process a single image to calculate inertia values for the Elbow Method.

    Args:
        filename: Image filename.
        input_folder: Folder containing the image.
        max_k: Maximum number of clusters to evaluate.
        random_state: Seed for reproducibility in K-Means.

    Returns:
        Filename and the inertia values for k=1 to max_k.
    """
    logger.info(f"Processing file for Elbow Method: {filename}")
    image_path = os.path.join(input_folder, filename)
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        logger.warning(f"Could not read image file: {filename}")
        return filename, []

    inertia_values = calculate_inertia(image_bgr, max_k=max_k, random_state=random_state)
    return filename, inertia_values


def main():
    input_folder = "./raw_images"
    output_folder = "./output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    max_k = 10
    random_state = 42

    # Collect images
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ]
    logger.info(f"Found {len(image_files)} image(s) in '{input_folder}' to process.")

    all_inertia_data = {}

    # Process images in parallel for Elbow Method
    with ProcessPoolExecutor(max_workers=14) as executor:
        futures = [
            executor.submit(process_image_for_elbow, fn, input_folder, max_k, random_state)
            for fn in image_files
        ]
        for f in as_completed(futures):
            filename, inertia_values = f.result()
            if inertia_values:
                all_inertia_data[filename] = inertia_values

    # Aggregate inertia data to determine the optimal k
    aggregated_inertia = np.zeros(max_k)
    count = 0

    for filename, inertia_values in all_inertia_data.items():
        aggregated_inertia[:len(inertia_values)] += np.array(inertia_values)
        count += 1

    if count > 0:
        # Average the inertia values across all images
        aggregated_inertia /= count

        # Plot the Elbow Method graph
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_k + 1), aggregated_inertia, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Average Inertia (Sum of Squared Distances)')
        plt.title('Elbow Method for Optimal k (Aggregated Across Images)')
        plt.grid()
        plt.show()

        # Save aggregated inertia values to CSV
        elbow_csv_path = os.path.join(output_folder, "elbow_method_results.csv")
        with open(elbow_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["k", "Average Inertia"])
            for k in range(1, max_k + 1):
                writer.writerow([k, aggregated_inertia[k - 1]])

        logger.info(f"Elbow Method results saved to {elbow_csv_path}")
    else:
        logger.warning("No inertia data was collected. Elbow Method could not be applied.")

    logger.info("Elbow Method processing completed.")

# -------------------------------------------------------------------------
# 3) Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()

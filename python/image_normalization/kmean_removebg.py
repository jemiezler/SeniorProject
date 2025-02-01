import cv2
import numpy as np
from sklearn.cluster import KMeans

def remove_background_kmeans(image_path, n_clusters=2, random_state=42):
    """
    Uses K-Means clustering to segment an image into 'n_clusters' clusters.
    Identifies the largest cluster as 'background' and removes it (sets to white).
    Returns the image with background removed and a binary mask.
    """
    # --- 1) Read the image in BGR format ---
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise IOError(f"Could not read image: {image_path}")

    # Convert BGR to RGB for clustering
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # --- 2) Reshape image to a 2D array of pixels [N, 3] ---
    h, w, c = image_rgb.shape
    pixels = image_rgb.reshape(-1, 3)  # shape: (N, 3)
    
    # --- 3) Apply K-Means clustering ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(pixels)
    labels = kmeans.labels_  # Array of shape (N,) with cluster indices [0..n_clusters-1]
    cluster_centers = kmeans.cluster_centers_  # (n_clusters, 3)

    # --- 4) Identify which cluster is 'background' ---
    # Option A: pick the cluster with the most pixels
    # Option B: pick the cluster with the highest average brightness, etc.
    # Here, we pick the cluster with the most pixels as background:
    label_counts = np.bincount(labels)
    background_label = np.argmax(label_counts)

    # --- 5) Create a mask where background pixels = True ---
    # shape: (N,) but we can reshape to (h, w) for an image
    background_mask = (labels == background_label).reshape(h, w)

    # --- 6) Remove or replace the background ---
    # Let's create a copy of the original image (in RGB)
    result_rgb = image_rgb.copy()
    # For background pixels, replace with white (255, 255, 255)
    result_rgb[background_mask] = [255, 255, 255]

    # Alternatively, to make background transparent, you’d have to create
    # a 4-channel RGBA image and set alpha=0 for background pixels.

    # --- Convert back to BGR for OpenCV's standard saving ---
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    # --- Create a binary mask (0 or 255) for background vs. foreground ---
    # For background pixels, set 255; for foreground, set 0 (or invert).
    mask_binary = np.zeros((h, w), dtype=np.uint8)
    mask_binary[background_mask] = 255

    return result_bgr, mask_binary


if __name__ == "__main__":
    # Example usage:
    input_image_path = "./raw_images/0_5_1_1.jpg"
    
    result_bgr, background_mask = remove_background_kmeans(input_image_path, n_clusters=2)
    
    # Save the resulting image (with background set to white)
    cv2.imwrite("image_no_background.jpg", result_bgr)

    # Save the mask
    cv2.imwrite("background_mask.png", background_mask)

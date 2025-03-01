import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

class FeatureExtractor:
    """Extracts only the required features (HSV, GLCM, LBP, Temp, Yellow, Cyan, Chroma)."""

    @staticmethod
    def extract_lbp_features(image_gray):
        """Extracts Local Binary Pattern (LBP) histogram features (Only LBP_0 to LBP_7)."""
        lbp = local_binary_pattern(image_gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        return {f"LBP_{i}": hist[i] for i in range(8)}  # ✅ Extracting only LBP_0 to LBP_7

    @staticmethod
    def extract_glcm_features(image_gray):
        """Extracts GLCM (Gray-Level Co-occurrence Matrix) texture features."""
        distances = [1]  # Distance of pixel pairs
        angles = [0]  # Horizontal direction
        glcm = graycomatrix(image_gray, distances=distances, angles=angles, symmetric=True, normed=True)

        glcm_features = {
            "GLCM_contrast": graycoprops(glcm, 'contrast')[0, 0],
            "GLCM_dissimilarity": graycoprops(glcm, 'dissimilarity')[0, 0],
            "GLCM_homogeneity": graycoprops(glcm, 'homogeneity')[0, 0],
            "GLCM_energy": graycoprops(glcm, 'energy')[0, 0],
            "GLCM_correlation": graycoprops(glcm, 'correlation')[0, 0],
        }
        return glcm_features

    @staticmethod
    def extract_color_features(image):
        """Extracts HSV mean/std, Yellow, Cyan, and Chroma computation."""
        # Convert image to HSV
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(image_hsv, axis=(0, 1))
        std_hsv = np.std(image_hsv, axis=(0, 1))

        # Convert image to RGB for Yellow and Cyan calculation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean_rgb = np.mean(image_rgb, axis=(0, 1))

        # Compute additional color features
        cyan = mean_rgb[1] + mean_rgb[2]  # G + B
        yellow = mean_rgb[0] + mean_rgb[1]  # R + G
        chroma = np.sqrt(mean_hsv[1]**2 + mean_hsv[2]**2)  # Saturation + Value combined

        color_features = {
            # HSV Features (Mean First)
            "Mean_HSV_H": mean_hsv[0],
            "Mean_HSV_S": mean_hsv[1],
            "Mean_HSV_V": mean_hsv[2],

            # HSV Features (Standard Deviation Next)
            "Std_HSV_H": std_hsv[0],
            "Std_HSV_S": std_hsv[1],
            "Std_HSV_V": std_hsv[2],

            # Additional Features
            "Cyan": cyan,
            "Yellow": yellow,
            "Chroma": chroma,
        }
        return color_features

    @staticmethod
    def extract_all_features(image: np.ndarray, temp: float):
        """Extracts all the required features from the image."""
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ✅ Extract features
        lbp_features = FeatureExtractor.extract_lbp_features(image_gray)
        glcm_features = FeatureExtractor.extract_glcm_features(image_gray)
        color_features = FeatureExtractor.extract_color_features(image)

        # ✅ Add temperature as a feature
        color_features["Temp"] = temp

        # ✅ Combine all extracted features
        all_features = {**color_features, **glcm_features, **lbp_features}

        return all_features

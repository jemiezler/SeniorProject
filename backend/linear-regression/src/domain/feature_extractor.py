# import cv2
# import numpy as np
# import pandas as pd
# from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

# # Color space mapping
# COLOR_SPACES = {
#     "RGB": (None, ["R", "G", "B"]),
#     "LAB": (cv2.COLOR_BGR2LAB, ["L", "A", "B"]),
#     "HSV": (cv2.COLOR_BGR2HSV, ["H", "S", "V"]),
#     "GRAY": (cv2.COLOR_BGR2GRAY, ["Gray"])
# }

# # GLCM Properties
# GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# class FeatureExtractor:
#     """Extracts color and texture features from images."""

#     @staticmethod
#     def extract_color_features(image: np.ndarray):
#         """Extracts mean and std from multiple color spaces and calculates derived parameters."""
#         color_features = {}
        
#         for space, (conversion, channels) in COLOR_SPACES.items():
#             img = cv2.cvtColor(image, conversion) if conversion else image
#             mean_vals = np.mean(img, axis=(0, 1))
#             std_vals = np.std(img, axis=(0, 1))
            
#             # Handle grayscale image case where mean/std_vals return a single scalar
#             if isinstance(mean_vals, np.ndarray):
#                 for i, channel in enumerate(channels):
#                     color_features[f"Mean_{space}_{channel}"] = mean_vals[i]
#                     color_features[f"Std_{space}_{channel}"] = std_vals[i]
#             else:
#                 color_features[f"Mean_{space}_{channels[0]}"] = mean_vals
#                 color_features[f"Std_{space}_{channels[0]}"] = std_vals

#         # Compute additional color-derived features
#         if "Mean_RGB_R" in color_features and "Mean_RGB_G" in color_features and "Mean_RGB_B" in color_features:
#             color_features["Yellow"] = color_features["Mean_RGB_R"] + color_features["Mean_RGB_G"]
#             color_features["Cyan"] = color_features["Mean_RGB_G"] + color_features["Mean_RGB_B"]
#             color_features["Magenta"] = color_features["Mean_RGB_R"] + color_features["Mean_RGB_B"]
#             color_features["Brightness"] = (
#                 color_features["Mean_RGB_R"] + color_features["Mean_RGB_G"] + color_features["Mean_RGB_B"]
#             ) / 3
#             color_features["Chroma"] = (
#                 max(color_features["Mean_RGB_R"], color_features["Mean_RGB_G"], color_features["Mean_RGB_B"])
#                 - min(color_features["Mean_RGB_R"], color_features["Mean_RGB_G"], color_features["Mean_RGB_B"])
#             )

#         return color_features

#     @staticmethod
#     def extract_glcm_features(image_gray):
#         """Extracts GLCM features from grayscale image."""
#         glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
#         return {f"GLCM_{prop}": graycoprops(glcm, prop).flatten()[0] for prop in GLCM_PROPS}

#     @staticmethod
#     def extract_lbp_features(image_gray):
#         """Extracts Local Binary Pattern (LBP) histogram features."""
#         lbp = local_binary_pattern(image_gray, P=8, R=1, method="uniform")
#         hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
#         return {f"LBP_{i}": hist[i] for i in range(10)}

#     @staticmethod
#     def extract_hog_features(image_gray):
#         """Extracts Histogram of Oriented Gradients (HOG) features."""
#         hog_features = hog(image_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)[:10]
#         return {f"HOG_{i}": hog_features[i] for i in range(10)}

#     @staticmethod
#     def extract_all_features(image: np.ndarray, temp: float):
#         """Extracts all features (color + texture) from an image and includes temperature."""
#         image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         color_features = FeatureExtractor.extract_color_features(image)
#         glcm_features = FeatureExtractor.extract_glcm_features(image_gray)
#         lbp_features = FeatureExtractor.extract_lbp_features(image_gray)
#         hog_features = FeatureExtractor.extract_hog_features(image_gray)

#         # Combine all features into a single dictionary
#         all_features = {**color_features, **glcm_features, **lbp_features, **hog_features}

#         # Add temperature as a feature
#         all_features["Temp"] = temp

#         return all_features
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

class FeatureExtractor:
    """Extracts only the required features from images to match the model input."""

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
        """Extracts mean and standard deviation of RGB channels + Cyan computation."""
        mean_rgb = np.mean(image, axis=(0, 1))  # Mean RGB values
        std_rgb = np.std(image, axis=(0, 1))  # Std RGB values
        cyan = mean_rgb[1] + mean_rgb[2]  # G + B

        color_features = {
            "Mean_RGB_R": mean_rgb[0], "Std_RGB_R": std_rgb[0],
            "Mean_RGB_G": mean_rgb[1], "Std_RGB_G": std_rgb[1],
            "Mean_RGB_B": mean_rgb[2], "Std_RGB_B": std_rgb[2],
            "Cyan": cyan,
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

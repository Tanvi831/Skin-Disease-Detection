import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Define input and output paths
image_folder = "clean_Dataset"  # Change this to match your dataset path
output_csv = "skin_disease_features.csv"

# Function to extract features from an image
def extract_features(image_path, class_name):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        return None  # Skip if the image is not loaded properly
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Texture Features
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # Color Features
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_r, mean_g, mean_b = cv2.mean(image)[:3]
    mean_h, mean_s, mean_v = cv2.mean(hsv_image)[:3]
    
    # Shape Features
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    hu_moments = cv2.HuMoments(cv2.moments(contours[0])).flatten() if num_contours > 0 else np.zeros(7)
    
    # Edge & Gradient Analysis
    canny_edges = np.mean(cv2.Canny(gray, 100, 200))
    sobel_x = np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5))
    sobel_y = np.mean(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
    laplacian = np.mean(cv2.Laplacian(gray, cv2.CV_64F))
    
    # Feature Detection
    orb = cv2.ORB_create()
    keypoints, _ = orb.detectAndCompute(gray, None)
    orb_keypoints = len(keypoints)
    
    # Extract image name
    image_name = os.path.basename(image_path)
    
    # Return all features
    return [class_name, image_name, image_path, contrast, homogeneity, mean_r, mean_g, mean_b, mean_h, mean_s, mean_v,
            num_contours, canny_edges, sobel_x, sobel_y, laplacian, orb_keypoints] + list(hu_moments)

# Process images
data = []
total_images = sum(len(files) for _, _, files in os.walk(image_folder))
processed_images = 0

for class_folder in os.listdir(image_folder):
    class_path = os.path.join(image_folder, class_folder)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            features = extract_features(img_path, class_folder)
            if features:
                data.append(features)
            processed_images += 1
            print(f"Processed {processed_images}/{total_images} images...")

# Define column names
columns = ["Class", "Image Name", "Image Path", "GLCM_Contrast", "GLCM_Homogeneity", "Mean_R", "Mean_G", "Mean_B", "Mean_H", "Mean_S", "Mean_V",
           "Num_Contours", "Canny_Edges", "Sobel_X", "Sobel_Y", "Laplacian", "ORB_Keypoints"] + [f"Hu_Moment{i}" for i in range(7)]

# Save features to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)

print(f"Feature extraction completed! Data saved to {output_csv}")

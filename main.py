import os
import cv2
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt

# Define dataset paths
dataset_path = 'Dataset'
clean_dataset_path = 'clean_Dataset'
image_statistics_csv = "image_statistics.csv"
filtered_images_csv = "filtered_good_images.csv"
image_features_csv = "image_features.csv"
#histogram_folder = "histograms"
TARGET_SIZE = (256, 256)  # Target image size for resizing

# # Create histogram folder if it doesn't exist
# if not os.path.exists(histogram_folder):
#     os.makedirs(histogram_folder)

# Function to compute image statistics and histogram
def compute_metrics(image, img_name):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute mean and variance
    mean_gray = np.mean(grayscale)
    var_gray = np.var(grayscale)
    mean_r, mean_g, mean_b = np.mean(image, axis=(0, 1))
    var_r, var_g, var_b = np.var(image, axis=(0, 1))
    
    # Compute histograms
    # hist_gray = cv2.calcHist([grayscale], [0], None, [256], [0, 256]).flatten()
    # hist_r = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    # hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
    # hist_b = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()

    # Normalize histograms
    # hist_gray = hist_gray / np.sum(hist_gray)
    # hist_r = hist_r / np.sum(hist_r)
    # hist_g = hist_g / np.sum(hist_g)
    # hist_b = hist_b / np.sum(hist_b)
    
    # Save histogram as an image (optional)
    # plt.figure(figsize=(8, 6))
    # plt.plot(hist_gray, color='black', label="Grayscale")
    # plt.plot(hist_r, color='red', label="Red")
    # plt.plot(hist_g, color='green', label="Green")
    # plt.plot(hist_b, color='blue', label="Blue")
    # plt.legend()
    # plt.title(f"Histogram: {img_name}")
    # plt.savefig(os.path.join(histogram_folder, f"{img_name}.png"))
    # plt.close()

    return mean_gray, var_gray, mean_r, mean_g, mean_b, var_r, var_g, var_b

# Step 1: Compute Image Statistics
data = []
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                metrics = compute_metrics(image, img_name)
                data.append([img_name, class_folder, *metrics])

# Save statistics to CSV
df = pd.DataFrame(data, columns=['Image Name', 'Class', 'Mean_Gray', 'Var_Gray', 
                                 'Mean_R', 'Mean_G', 'Mean_B', 'Var_R', 'Var_G', 'Var_B'])
df.to_csv(image_statistics_csv, index=False)

# Step 2: Filter Good Images

# Compute grayscale thresholds
mean_gray_lower = df["Mean_Gray"].quantile(0.1)
mean_gray_upper = df["Mean_Gray"].quantile(0.9)
var_gray_upper = df["Var_Gray"].quantile(0.9)

# Compute RGB thresholds
mean_r_lower = df["Mean_R"].quantile(0.1)
mean_r_upper = df["Mean_R"].quantile(0.9)
mean_g_lower = df["Mean_G"].quantile(0.1)
mean_g_upper = df["Mean_G"].quantile(0.9)
mean_b_lower = df["Mean_B"].quantile(0.1)
mean_b_upper = df["Mean_B"].quantile(0.9)

var_r_upper = df["Var_R"].quantile(0.9)
var_g_upper = df["Var_G"].quantile(0.9)
var_b_upper = df["Var_B"].quantile(0.9)

# Apply filtering
good_images = df[
    (df["Mean_Gray"] >= mean_gray_lower) & (df["Mean_Gray"] <= mean_gray_upper) &
    (df["Var_Gray"] <= var_gray_upper) &
    (df["Mean_R"] >= mean_r_lower) & (df["Mean_R"] <= mean_r_upper) &
    (df["Mean_G"] >= mean_g_lower) & (df["Mean_G"] <= mean_g_upper) &
    (df["Mean_B"] >= mean_b_lower) & (df["Mean_B"] <= mean_b_upper) &
    (df["Var_R"] <= var_r_upper) & (df["Var_G"] <= var_g_upper) & (df["Var_B"] <= var_b_upper)
]

# Save filtered images to CSV
good_images.to_csv(filtered_images_csv, index=False)

# Step 3: Resize and Move Good Images
if not os.path.exists(clean_dataset_path):
    os.makedirs(clean_dataset_path)

for _, row in good_images.iterrows():
    source_path = os.path.join(dataset_path, row["Class"], row["Image Name"])
    dest_folder = os.path.join(clean_dataset_path, row["Class"])
    dest_path = os.path.join(dest_folder, row["Image Name"])
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    image = cv2.imread(source_path)
    if image is not None:
        resized_image = cv2.resize(image, TARGET_SIZE)
        cv2.imwrite(dest_path, resized_image)



print("Process completed successfully! Image statistics, filtering, resizing are saved.")


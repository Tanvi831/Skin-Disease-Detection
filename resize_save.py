# resize_and_save.py

import os
from PIL import Image

def resize_and_save_image(uploaded_file, output_dir="resized_images", size=(256, 256)):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert to RGB, resize, and save
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize(size)

    resized_path = os.path.join(output_dir, uploaded_file.name)
    resized_image.save(resized_path)

    return resized_path

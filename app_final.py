
import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from disease_info import disease_info
from resize_save import resize_and_save_image



with open("cnn.pkcls", "rb") as f:
    cnn_model = pickle.load(f)

inception_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

class_names = {
    0: "acne",
    1: "athelete's foot",
    2: "basal cell carcinoma",
    3: "chickenpox",
    4: "impetigo",
    5: "insect bite",
    6: "melanoma",
    7: "normal",
    8: "ringworm",
    9: "shingles",
    10: "vitiligo",
}

def extract_embedding(img):
    img = img.convert('RGB') 
    img = img.resize((299, 299))  
    img_array = np.array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array)  
    embedding = inception_model.predict(img_array)  
    return embedding.flatten()

st.markdown("""
    <style>
        body {
            background-color: #fdfaf6;
        }
        .title {
            font-size: 38px;
            font-weight: bold;
            text-align: center;
            color: #6D4C41;
            padding-bottom: 10px;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #fff7f0;
            border-radius: 12px;
            padding: 8px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stTabs [data-baseweb="tab"] {
            color: #5D4037;
            font-size: 18px;
            font-weight: 600;
            padding: 10px;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e8dfd6;
            color: #6D4C41;
            border-radius: 10px;
            font-weight: bold;
        }
        .stFileUploader {
            border: 2px solid #6D4C41;
            background-color: #f3e5ab;
            border-radius: 10px;
            padding: 10px;
        }
        .stSuccess {
            background-color: #e6f4e6;
            color: #1B5E20;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Skin Disease Detection</h1>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Webcam Capture"])

with tab1:
    st.header("Upload an Image ", divider="rainbow")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        st.subheader("Prediction Results")
        for i in range(0, len(uploaded_files), 3):
            cols = st.columns(min(3, len(uploaded_files) - i))
            for j, uploaded_file in enumerate(uploaded_files[i:i+3]):
                with cols[j]:
                   
                    img = Image.open(uploaded_file)  # Ensure image is in RGB format
                    resized_path = resize_and_save_image(uploaded_file)
                    img = Image.open(resized_path)
                    st.image(img, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
                    st.success("Image uploaded successfully!", icon="✅")
                    
                    img_embedding = extract_embedding(img)
                    pred_class = cnn_model.predict([img_embedding])[0]
                    pred_class = int(pred_class)
                    pred_name = class_names.get(pred_class, "Unknown")
                    info = disease_info.get(pred_name.lower())
                    st.subheader("Prediction")
                    st.success(f"**Predicted Disease:** {pred_name}")

                    if info:
                        st.subheader("Disease Overview")
                        st.write(info["overview"])
                        print("\n")
                        st.subheader("Symptoms")
                        st.write(info["symptoms"])
                        print("\n")
                        st.subheader("Causes")
                        st.write(info["causes"])
                        print("\n")
                        st.subheader("Treatment")
                        st.write(info["treatment"])
                        print("\n")
                        st.subheader("Medical Advice")
                        st.write(info["medical advice"])
                    else:
                        st.warning("No additional information available for this condition.")

with tab2:
    st.header("Capture from Webcam", divider="rainbow")
    cam = st.camera_input("Take a picture")
    if cam:
        image = Image.open(cam)
        st.image(image, caption="Captured Image", use_container_width=True)
        st.success("Webcam image captured successfully!", icon="✅")
        image_embedding = extract_embedding(image)
        predicted_class = cnn_model.predict([image_embedding])[0]
        predicted_class = int(predicted_class)
        predicted_class_name = class_names.get(predicted_class, "Unknown")
        st.subheader(f"Prediction for Captured Image:")
        st.success(f"**Predicted Disease:** {predicted_class_name}")
        info = disease_info.get(predicted_class_name.lower())
        if info:
            st.subheader("Disease Overview")
            st.write(info["overview"])
            st.subheader("Symptoms")
            st.write(info["symptoms"])
            st.subheader("Causes")
            st.write(info["causes"])
            st.subheader("Treatment")
            st.write(info["treatment"])
            st.subheader("Medical Advice")
            st.write(info["medical advice"])
        else:
            st.warning("No additional information available for this condition.")

st.markdown("""
---
📝 **Disclaimer:**
This Skin Disease Detection System is an AI-assisted application developed for **academic and research purposes** as part of a final year project. The predictions provided by this system are based on image analysis using a machine learning model and should be treated as **preliminary assessments only**.
This tool is **not a certified diagnostic system** and is **not intended to replace professional medical advice, diagnosis, or treatment**. The system may not accurately detect all skin conditions, and results can vary depending on image quality, lighting, and other factors.
Users are **strongly advised to consult a licensed dermatologist or healthcare professional** for a proper diagnosis and treatment plan.
""", unsafe_allow_html=True)

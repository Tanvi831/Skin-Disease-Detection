# Skin-Disease-Detection System using CNN

This is a deep learning-based web application built using **Streamlit** that can detect common skin diseases from uploaded images. The model is trained using CNN and provides prediction results along with disease information like symptoms, causes, and treatment.

---

## Features

- Upload or capture skin images using webcam
- Predicts one of 11 skin conditions using a trained CNN model
- Provides disease overview, symptoms, causes, and treatments
- Beautiful UI with multi-image support and full-screen comparison
- Displays prediction confidence (probability)
- Modular and well-structured Python code

---

## Technologies Used

- **Python**
- **Streamlit** – for UI
- **TensorFlow / Keras** – for InceptionV3-based embedding
- **Scikit-learn** – for model training (Logistic Regression / KNN)
- **PIL & OpenCV** – for image handling
- **Orange3** – for model training and exporting
- **Pickle** – to load saved trained model

---

## Supported Skin Diseases

- Acne  
- Athlete's foot  
- Basal cell carcinoma  
- Chickenpox  
- Impetigo  
- Insect bite  
- Melanoma  
- Normal  
- Ringworm  
- Shingles  
- Vitiligo  

---

## Install dependencies

pip install -r requirements.txt

---

## Run the Streamlit app

streamlit run app.py

---

## Disclaimer

This system is for academic and research purposes only. It is not a certified medical tool and should not be used as a replacement for professional diagnosis. Always consult a qualified dermatologist for accurate medical advice.

---



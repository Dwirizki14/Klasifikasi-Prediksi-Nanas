import streamlit as st
import joblib
import numpy as np
from PIL import Image
import cv2

# Load model & scaler
model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler_knn.pkl")

st.title("Prediksi Kematangan Nanas 🍍")

uploaded_file = st.file_uploader("Upload gambar nanas", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Resize & konversi ke RGB
    img = image.resize((100, 100))
    img_array = np.array(img)
    avg_rgb = img_array.mean(axis=(0, 1))
    r, g, b = int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2])

    st.markdown(f"**Nilai RGB**: R={r}, G={g}, B={b}")

    sample_scaled = scaler.transform([[r, g, b]])
    prediksi = model.predict(sample_scaled)[0]

    st.markdown(f"### Prediksi Kematangan: **{prediksi.upper()}**")

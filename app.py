import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load model dan scaler
model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler_knn.pkl")

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Kematangan Nanas", layout="wide")
st.markdown("<h1 style='text-align: center; color: green;'>🍍 Prediksi Kematangan Buah Nanas 🍍</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Buat dua kolom
col1, col2 = st.columns([1.8, 1.2])

with col1:
    st.subheader("📤 Upload Gambar & Prediksi")

    uploaded_file = st.file_uploader("Pilih gambar buah nanas", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Baca dan olah gambar
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)

        # Hitung nilai RGB
        r = round(np.mean(img_array[:, :, 0]))
        g = round(np.mean(img_array[:, :, 1]))
        b = round(np.mean(img_array[:, :, 2]))
        rgb_input = np.array([[r, g, b]])
        scaled_input = scaler.transform(rgb_input)
        pred = model.predict(scaled_input)[0]

        # Tampilkan hasil prediksi
        st.success(f"🎯 Prediksi: **{pred.upper()}**")
        st.info(f"📊 RGB = 🔴 R: {r}, 🟢 G: {g}, 🔵 B: {b}")
    else:
        st.warning("Silakan unggah gambar terlebih dahulu.")

with col2:
    st.subheader("🖼️ Gambar")
    if uploaded_file is not None:
        st.image(image, width=250, caption="Gambar yang diunggah")
    

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:12px;'>© 2025 Aplikasi Prediksi Nanas - Dibuat dengan Streamlit</p>", unsafe_allow_html=True)

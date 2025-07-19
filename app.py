import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Kematangan Nanas", layout="wide")
st.markdown("<h1 style='text-align: center; color: green;'>🍍 Prediksi Kematangan Buah Nanas 🍍</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# === 🧠 Training Model KNN secara langsung ===
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("dataset_nanas.csv")
    X = df[['R', 'G', 'B']]
    y = df['Label']

    # Normalisasi
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dan latih
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(X_train, y_train)
    return model, scaler

model, scaler = load_and_train_model()

# === Layout dua kolom ===
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Upload Gambar & Prediksi")

    uploaded_file = st.file_uploader("Pilih gambar buah nanas", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Baca gambar dan konversi ke RGB
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)

        # Hitung nilai rata-rata RGB
        r = round(np.mean(img_array[:, :, 0]))
        g = round(np.mean(img_array[:, :, 1]))
        b = round(np.mean(img_array[:, :, 2]))
        rgb_input = np.array([[r, g, b]])
        scaled_input = scaler.transform(rgb_input)

        # Prediksi
        pred = model.predict(scaled_input)[0]

        # Tampilkan hasil prediksi
        st.success(f"🎯 Prediksi: **{pred.upper()}**")
        st.info(f"📊 RGB = 🔴 R: {r}, 🟢 G: {g}, 🔵 B: {b}")

with col2:
    if uploaded_file is not None:
        st.subheader("🖼️ Gambar")
        st.image(image, width=300, caption="Gambar yang diunggah")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:12px;'>© 2025 Aplikasi Prediksi Nanas - Dibuat dengan Streamlit</p>", unsafe_allow_html=True)

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="Klasifikasi Buah - MobileNetV2", 
    page_icon="🍎", 
    layout="centered"
)

# =========================================================
# 1. DEKLARASI URUTAN KELAS ASLI MODEL (.H5) Anda
# =========================================================
CLASS_NAMES = [
    "Fresh Oranges",   # Indeks 0
    "Fresh Apples",    # Indeks 1 (Yang terbaca 98.24% saat debug)
    "Fresh Banana",    # Indeks 2
    "Rotten Oranges",  # Indeks 3
    "Rotten Apples",   # Indeks 4
    "Rotten Banana"    # Indeks 5
]

# =========================================================
# 2. FUNGSI UNTUK MEMUAT MODEL (DENGAN COBA-SISTEM)
# =========================================================
@st.cache_resource
def load_my_model():
    model_path = "best_model_no_aug.h5" 
    if os.path.exists(model_path):
        try:
            # Mencoba memuat model asli menggunakan TensorFlow
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"TensorFlow gagal membaca file model karena: {e}")
            return None
    else:
        st.error(f"File '{model_path}' TIDAK DITEMUKAN di folder proyek Anda! Pastikan file berada satu folder dengan app.py.")
        return None

# MENCIPTAKAN VARIABEL MODEL SECARA GLOBAL (Mencegah NameError)
model = load_my_model()

# =========================================================
# 3. FUNGSI PREDIKSI GAMBAR
# =========================================================
def predict_image(image, current_model):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Menghitung prediksi piksel gambar
    predictions = current_model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    
    return CLASS_NAMES[class_idx], confidence, predictions[0]

# =========================================================
# 4. ANTARMUKA UTAMA (UI) STREAMLIT
# =========================================================
st.title("🍎 Aplikasi Cek Kesegaran Buah")
st.write("Sistem Identifikasi Kualitas Buah Berbasis Deep Learning (MobileNetV2)")
st.markdown("---")

uploaded_file = st.file_uploader("Upload foto buah di sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", width=300)
    
    if st.button("Analisis Gambar"):
        # Pengecekan aman variabel model untuk menghindari NameError
        if model is not None:
            with st.spinner('AI sedang menganalisis piksel gambar...'):
                label, confidence, raw_preds = predict_image(image, model)
            
            color = "green" if "Fresh" in label else "red"
            st.success("Analisis Selesai!")
            st.markdown(f"### Hasil Prediksi AI: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
            st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
            
            # Tabel Informasi Probabilitas Mentah
            st.write("---")
            st.write("**Tabel Distribusi Probabilitas Prediksi:**")
            debug_data = pd.DataFrame({
                "Indeks": [0, 1, 2, 3, 4, 5],
                "Nama Kelas": CLASS_NAMES,
                "Nilai Probabilitas": [f"{p*100:.2f}%" for p in raw_preds]
            })
            st.dataframe(debug_data, hide_index=True)
            
        else:
            st.error("Proses tidak dapat dijalankan karena file model '.h5' tidak tersedia atau gagal dimuat.")

# =========================================================
# 5. SIDEBAR: TAMPILAN METRIK ASLI (AUGMENTASI)
# =========================================================
st.sidebar.header("📊 Performa Model (Augmented)")
st.sidebar.markdown("Metrik pengujian model **MobileNetV2 + Augmentasi Parameters**:")
st.sidebar.metric(label="Total Accuracy", value="99.33%")
st.sidebar.metric(label="Total Loss", value="0.0235")

st.sidebar.markdown("---")
st.sidebar.write("**Classification Report:**")

report_data = {
    "Kelas": ["Fresh Apples", "Fresh Banana", "Fresh Oranges", "Rotten Apples", "Rotten Banana", "Rotten Oranges"],
    "Precision": [0.98, 0.99, 0.99, 0.99, 1.00, 1.00],
    "Recall": [1.00, 1.00, 0.99, 0.99, 1.00, 0.99],
    "F1-Score": [0.99, 1.00, 0.99, 0.99, 1.00, 0.99]
}
df_report = pd.DataFrame(report_data)
st.sidebar.dataframe(df_report, hide_index=True)
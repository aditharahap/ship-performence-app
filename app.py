import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Ship Performance Clustering", layout="centered")

# Judul Aplikasi
st.title("🚢 Aplikasi Clustering Performa Kapal")
st.write("""
Aplikasi ini menggunakan model **K-Means Clustering** untuk mengelompokkan kapal 
berdasarkan profil kecepatan, daya mesin, biaya, dan pendapatannya.
""")

# Load Model dan Scaler
@st.cache_resource
def load_models():
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_models()
except FileNotFoundError:
    st.error("Model tidak ditemukan! Pastikan file 'kmeans_model.pkl' dan 'scaler.pkl' ada di direktori yang sama.")
    st.stop()

# Form Input Data
st.sidebar.header("Masukkan Spesifikasi Kapal")
speed = st.sidebar.slider("Speed Over Ground (knots)", min_value=0.0, max_value=40.0, value=15.0)
power = st.sidebar.number_input("Engine Power (kW)", min_value=0.0, value=5000.0)
cost = st.sidebar.number_input("Operational Cost (USD)", min_value=0.0, value=250000.0)
revenue = st.sidebar.number_input("Revenue per Voyage (USD)", min_value=0.0, value=300000.0)

# Tombol Prediksi
if st.button("Tentukan Segmentasi Kapal"):
    # data input
    input_data = np.array([[speed, power, cost, revenue]])
    
    # Standarisasi data menggunakan scaler dari proses training
    input_scaled = scaler.transform(input_data)
    
    # Melakukan prediksi cluster
    cluster_result = model.predict(input_scaled)[0]
    
    # Tampilkan hasil
    st.success(f"Berdasarkan data yang diinput, kapal ini masuk ke dalam **Cluster {cluster_result}**")
    
    # Tampilkan interpretasi dummy (bisa Anda sesuaikan dengan hasil analisis evaluasi Anda)
    if cluster_result == 0:
        st.info("Karakteristik Umum: Cluster ini mungkin mewakili kapal dengan biaya seimbang dan performa standar.")
    elif cluster_result == 1:
        st.info("Karakteristik Umum: Cluster ini mungkin mewakili kapal dengan daya mesin tinggi atau biaya operasional besar.")
    else:
        st.info("Karakteristik Umum: Cluster ini mungkin mewakili kapal dengan efisiensi profitabilitas terbaik.")
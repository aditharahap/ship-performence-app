import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Ship Performance Clustering", layout="centered")

# Judul Aplikasi
st.title("🚢 Aplikasi Clustering Performa Kapal")
st.write("""
Aplikasi ini menggunakan model **K-Means Clustering** untuk mengelompokkan performa kapal 
berdasarkan kecepatan, daya mesin, biaya operasional, dan pendapatannya.
""")

st.divider() # Garis pembatas

# Load Model dan Scaler
@st.cache_resource
def load_models():
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_models()
except FileNotFoundError:
    st.error("Model tidak ditemukan! Pastikan file 'kmeans_model.pkl' dan 'scaler.pkl' sudah diunggah.")
    st.stop()

# --- FORM INPUT DATA (Bukan Sidebar) ---
st.subheader("🛠️ Masukkan Spesifikasi Kapal")
st.write("Silakan isi parameter di bawah ini untuk menentukan segmen performa kapal:")

# Menggunakan kolom agar tampilan form lebih rapi (2 kolom)
col1, col2 = st.columns(2)

with col1:
    speed = st.slider("Speed Over Ground (knots)", min_value=0.0, max_value=40.0, value=15.0)
    power = st.number_input("Engine Power (kW)", min_value=0.0, value=5000.0)

with col2:
    cost = st.number_input("Operational Cost (USD)", min_value=0.0, value=250000.0)
    revenue = st.number_input("Revenue per Voyage (USD)", min_value=0.0, value=300000.0)

st.write("") # Memberi sedikit jarak

# Tombol Prediksi
if st.button("Tentukan Segmentasi Kapal", type="primary"): # Menggunakan warna tombol utama
    # data input
    input_data = np.array([[speed, power, cost, revenue]])
    
    # Standarisasi data menggunakan scaler dari proses training
    input_scaled = scaler.transform(input_data)
    
    # Melakukan prediksi cluster
    cluster_result = model.predict(input_scaled)[0]
    
    # Tampilkan hasil
    st.divider()
    st.success(f"🎯 Berdasarkan spesifikasi yang diinput, kapal ini masuk ke dalam **Cluster {cluster_result}**")
    
    # Tampilkan interpretasi
    st.subheader("📊 Interpretasi Cluster")
    if cluster_result == 0:
        st.info("Karakteristik Umum: Cluster ini mungkin mewakili kapal dengan biaya seimbang dan performa operasional standar.")
    elif cluster_result == 1:
        st.info("Karakteristik Umum: Cluster ini mungkin mewakili kapal dengan daya mesin tinggi atau pengeluaran biaya operasional yang besar.")
    else:
        st.info("Karakteristik Umum: Cluster ini mungkin mewakili kapal dengan efisiensi bahan bakar dan profitabilitas yang terbaik.")

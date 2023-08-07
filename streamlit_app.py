import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

df = pd.read_excel ('https://github.com/arypr/Prediksi-Harga-Rumah-Tebet-app/blob/main/data_bersih%20(1).xlsx')

model = LinearRegression()

# Menyimpan model ke dalam file menggunakan pickle
with open('modelku.pkl', 'wb') as file:
    pickle.dump(model, file)
    
with open('modelku.pkl', 'rb') as file:
    model = pickle.load(file)

# Fungsi untuk melakukan prediksi harga rumah
def predict_price(lb, lt, kt, km, grs):
    # Memuat model yang telah dilatih sebelumnya
    model = LinearRegression()
    model.fit(df[["LB", "LT", "KT", "KM", "GRS"]], df["HARGA"])
    # Melakukan prediksi harga rumah berdasarkan input pengguna
    prediction = model.predict([[lb, lt, kt, km, grs]])
    return prediction[0]

# Menampilkan header dan judul
st.write("""
# Prediksi Harga Rumah Tebet
""")

# Membuat sidebar
st.sidebar.header("Input Data")

lb = st.sidebar.text_input("Luas Bangunan (m2)", value=str(df["LB"].min()))

# Validasi input lb sebagai angka positif
if lb:
    try:
        lb = float(lb)
        if lb < 0:
            st.sidebar.error("Luas bangunan tidak boleh negatif")
    except ValueError:
        st.sidebar.error("Masukkan angka yang valid untuk Luas Bangunan (LB)")

lt = st.sidebar.text_input("Luas Tanah (m2)", value=str(df["LT"].min()))

# Validasi input lt sebagai angka positif
if lt:
    try:
        lt = float(lt)
        if lt < 0:
            st.sidebar.error("Luas tanah tidak boleh negatif")
    except ValueError:
        st.sidebar.error("Masukkan angka yang valid untuk Luas Tanah (LT)")




# Menampilkan input jumlah kamar tidur (KT)
kt = st.sidebar.selectbox("Jumlah Kamar Tidur", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Menampilkan input jumlah kamar mandi (KM)
km = st.sidebar.selectbox("Jumlah Kamar Mandi", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Menampilkan input kapasitas mobil dalam garasi (GRS)
grs = st.sidebar.selectbox("Kapasitas Mobil dalam Garasi", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Menampilkan tombol untuk memproses input pengguna
button = st.sidebar.button("Prediksi")

# Menampilkan hasil prediksi harga rumah jika tombol "Prediksi" ditekan
if button:
    price = predict_price(lb, lt, kt, km, grs)
    formatted_price = float(price)  # Mengubah ke tipe data numerik

    # Validasi hasil prediksi
    min_harga_aktual = df["HARGA"].min()
    selisih_minimal = 0.05 * min_harga_aktual
    if formatted_price < (min_harga_aktual - selisih_minimal):
        st.write("Hasil tidak ditemukan")
    else:
        # Mengambil dataframe nama rumah dan harga aktual
        harga_aktual = df['HARGA']
        luas_bangunan = df['LB']
        luas_tanah = df['LT']
        kamar_tidur = df['KT']
        kamar_mandi = df['KM']
        garasi = df['GRS']

        # Menghitung selisih antara harga aktual dan harga prediksi
        selisih = abs(harga_aktual - formatted_price)  # Menggunakan formatted_price
    

        # Membuat dataframe hasil prediksi dengan harga yang tidak jauh berbeda
        hasil_prediksi = pd.DataFrame({'Harga Aktual': harga_aktual, 'Selisih': selisih, 'LB': luas_bangunan, 'LT': luas_tanah, 'KT': kamar_tidur, 'KM': kamar_mandi, 'Garasi': garasi})
        hasil_prediksi = hasil_prediksi[hasil_prediksi['Selisih'] <= selisih_minimal]  # Menggunakan selisih_minimal
        hasil_prediksi = hasil_prediksi[(hasil_prediksi['Selisih'] <= selisih_minimal) & 
                                (hasil_prediksi['LB'] <= selisih_minimal) & 
                                (hasil_prediksi['LT'] <= selisih_minimal) & 
                                (hasil_prediksi['KT'] <= selisih_minimal) & 
                                (hasil_prediksi['KM'] <= selisih_minimal) & 
                                (hasil_prediksi['Garasi'] <= selisih_minimal)]

        # Menampilkan dataframe hasil prediksi jika data ditemukan
        if not hasil_prediksi.empty:
            fp2 = "{:,.2f}".format(float(price))
            st.write("Harga rumah yang diprediksi adalah Rp.", fp2)
            st.write("Daftar Rumah Rekomendasi Sesuai Harga Prediksi:")
            st.dataframe(hasil_prediksi)
        else:
            st.write("Hasil tidak ditemukan")

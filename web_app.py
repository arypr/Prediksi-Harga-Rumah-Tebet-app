import streamlit as st
import pandas as pd
from projek_pi_prediksihargarumah import df, mlr
from sklearn.linear_model import LinearRegression

# Fungsi untuk melakukan prediksi harga rumah
def predict_price(lb, lt, kt, km, grs):
    # Memuat model yang telah dilatih sebelumnya
    model = LinearRegression()
    model.fit(df[["LB", "LT", "KT", "KM", "GRS"]], df["HARGA"])
    # Melakukan prediksi harga rumah berdasarkan input pengguna
    prediction = mlr.predict([[lb, lt, kt, km, grs]])
    return prediction[0]

# Menampilkan header dan judul
st.write("""
# Prediksi Harga Rumah Tebet
""")

# Membuat sidebar
st.sidebar.header("Input Data")

# Menampilkan input luas bangunan (LB)
lb = st.sidebar.text_input("Luas Bangunan (m2)", value=str(df["LB"].min()))

# Validasi input lb sebagai angka
try:
    lb = float(lb)
except ValueError:
    st.sidebar.error("Masukkan angka yang valid untuk Luas Bangunan (LB)")

# Menampilkan input luas tanah (LT)
lt = st.sidebar.text_input("Luas Tanah (m2)", value=str(df["LT"].min()))

# Validasi input lt sebagai angka
try:
    lt = float(lt)
except ValueError:
    st.sidebar.error("Masukkan angka yang valid untuk Luas Tanah (LT)")

# Menampilkan input jumlah kamar tidur (KT)
kt = st.sidebar.selectbox("Jumlah Kamar Tidur", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Menampilkan input jumlah kamar mandi (KM)
km = st.sidebar.selectbox("Jumlah Kamar Mandi", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Menampilkan input kapasitas mobil dalam garasi (GRS)
grs = st.sidebar.selectbox("Kapasitas Mobil dalam Garasi", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Menampilkan tombol untuk memproses input pengguna
button = st.sidebar.button("Prediksi")

# Menampilkan hasil prediksi harga rumah jika tombol "Prediksi" ditekan
if button:
    price = predict_price(lb, lt, kt, km, grs)
    formatted_price = "{:,.2f}".format(float(price))
    st.write("Harga rumah yang diprediksi adalah Rp.", formatted_price)
    
    # Mengambil dataframe nama rumah dan harga aktual
    nama_rumah = df['NAMA RUMAH']
    harga_aktual = df['HARGA']
    
    # Menghitung selisih antara harga aktual dan harga prediksi
    selisih = abs(harga_aktual - price)
    
    # Membuat dataframe hasil prediksi dengan harga yang tidak jauh berbeda
    hasil_prediksi = pd.DataFrame({'NAMA RUMAH': nama_rumah, 'Harga Aktual': harga_aktual, 'Selisih': selisih})
    hasil_prediksi = hasil_prediksi[hasil_prediksi['Selisih'] <= 0.1 * float(price)]  # Harga dengan selisih maksimum 10% dari harga prediksi
    
    # Menampilkan dataframe hasil prediksi
    st.write("Daftar Rumah Rekomendasi Sesuai Harga Prediksi:")
    st.dataframe(hasil_prediksi)

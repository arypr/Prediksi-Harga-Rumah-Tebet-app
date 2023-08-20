# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load Dataset
df = pd.read_excel('https://raw.githubusercontent.com/arypr/Prediksi-Harga-Rumah-Tebet-app/main/df_prediksi.xlsx', engine='openpyxl')

X= df[['LB', 'LT', 'KT', 'KM', 'GRS']]
y = df['HARGA']

df.drop(columns='Unnamed: 0', inplace=True)

import pickle

with open('model_regression.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

import streamlit as st

def predict_price(features):
    predicted_price = loaded_model.predict([features])
    return predicted_price[0]

# Fungsi untuk menghitung perbedaan antara harga prediksi dan harga asli dalam dataset
def calculate_feature_difference(predicted_features, original_features):
    return abs(predicted_features - original_features)

def calculate_price_difference(predicted_price, original_price):
    return abs(predicted_price - original_price) / original_price

# Tampilan aplikasi Streamlit
st.title('Prediksi Harga Rumah Tebet')

# Masukkan fitur-fitur di dalam sidebar
with st.sidebar:

    st.title('Input Data')
    lb = st.number_input('Luas Bangunan', min_value=0)
    lt = st.number_input('Luas Tanah', min_value=0)
    kt = st.number_input('Jumlah Kamar Tidur', min_value=0)
    km = st.number_input('Jumlah Kamar Mandi', min_value=0)
    grs = st.number_input('Jumlah Garasi', min_value=0)
    predict_button = st.button('Prediksi')

# Jika tombol prediksi ditekan
if predict_button:
    features = np.array([lb, lt, kt, km, grs])
    predicted_price = predict_price(features)

    if predicted_price < 0:
        st.write('Hasil Tidak Ditemukan')
    else:
        formatted_price = "Rp {:,.2f}".format(predicted_price)
        st.write(f'Prediksi Harga: {formatted_price}')

        # Hitung perbedaan nilai fitur-fitur dalam dataset dengan nilai fitur-fitur yang diinputkan
        feature_diff = calculate_feature_difference(X.values, features)

        # Tentukan batas maksimal perbedaan untuk setiap fitur
        max_feature_diff = np.array([40, 40, 1, 1, 1])

        # Cari indeks rumah-rumah yang memenuhi batas perbedaan fitur
        similar_houses = np.all(feature_diff <= max_feature_diff, axis=1)

        # Tampilkan rumah-rumah yang sesuai dengan kriteria
        similar_houses_df = df[similar_houses]
        st.write('Rumah-rumah yang Sesuai:')
        st.dataframe(similar_houses_df)
        st.write('Rumah-rumah yang Sesuai:')
        st.dataframe(similar_houses_df)

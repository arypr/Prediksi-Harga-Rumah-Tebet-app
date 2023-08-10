# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load Dataset
df = pd.read_excel(r'C:\Users\arypr\Downloads\df_prediksi.xlsx')


# Pembuatan Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Misalkan X dan y adalah data Anda
X = df[['LB', 'LT', 'KT', 'KM','GRS']]
y = df['HARGA']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model regresi linear berganda
model = LinearRegression()

# Latih model dengan data latih
model.fit(X_train, y_train)

# Lakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Multiple Linear Regression:")
print(f"R-squared: {r2}")

import pickle
with open('model_linreg.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('model_linreg.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


# Uji Model
import math

print("Rumah impian shani berkisar pada harga IDR {:,} juta".format(math.floor(model.predict([[100, 120, 2, 2, 1]])/1000000)))

# Pembuatan Web App
import streamlit as st

def predict_price(features):
    predicted_price = model.predict([features])
    return predicted_price[0]

# Fungsi untuk menghitung perbedaan antara harga prediksi dan harga asli dalam dataset
def calculate_feature_difference(predicted_features, original_features):
    return abs(predicted_features - original_features)

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
    formatted_price = "Rp {:,.2f}".format(predicted_price)
    st.write(f'Prediksi Harga: {formatted_price}')

    # Hitung perbedaan nilai fitur-fitur dalam dataset dengan nilai fitur-fitur yang diinputkan
    feature_diff = calculate_feature_difference(X.values, features)

    # Tentukan batas maksimal perbedaan untuk setiap fitur
    max_feature_diff = np.array([50, 50, 1, 1, 1])

    # Cari indeks rumah-rumah yang memenuhi batas perbedaan fitur
    similar_houses = np.all(feature_diff <= max_feature_diff, axis=1)

    # Tampilkan rumah-rumah yang sesuai dengan kriteria
    similar_houses_df = df[similar_houses]
    st.write('Rumah-rumah yang Sesuai:')
    st.dataframe(similar_houses_df)

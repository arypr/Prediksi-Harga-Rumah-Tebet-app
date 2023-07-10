#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[3]:


df = pd.read_excel('https://raw.githubusercontent.com/username/repository/main/DATA RUMAH (1).xlsx')


# Berikut merupakan atribut dalam dataset DATA RUMAH (1), yaitu:
# 
# 1. Nama Rumah : Nama-nama rumah yang berlokasi di daerah Tebet
# 2. Harga : Harga rumah dalam rupiah
# 3. LB : Lebar rumah dalam satuan meter
# 4. LT : Luas tanah dalam satuan mater persegi
# 5. KT : Jumlah kamar tidur
# 6. KM : Jumlah kamar mandi
# 7. GRS : Kapasitas garasi untuk menampung mobil

# ## Data Preprocessing

# ## Identifikasi Missing Value

# ## Identifikasi Outliers

# ## Box Plot

# ## Jangkauan Kuartil IQR

# In[4]:


q11,q31=np.percentile(df['LB'], [25,75])
s1 = q31-q11
ba1 = q31+(1.5*s1)
bw1 = q11-(1.5*s1)
print(q11) # kuartil 1
print(q31) # kuartil 3
print(s1)  # IQR
print(ba1) # Batas atas
print(bw1) # Batas bawah


# In[5]:


# Menampilkan outliers pada atribut LB
outliers_LB = df[(df['LB']<bw1) | (df['LB']>ba1)]
outliers_LB.info()


# In[6]:


q12,q32=np.percentile(df['LT'], [25,75])
s2 = q32-q12
ba2 = q32+(1.5*s2)
bw2 = q12-(1.5*s2)
print(q12) # Kuartil 1
print(q32) # Kuartil 3
print(s2)  # IQR
print(ba2) # Batas atas
print(bw2) # Batas bawah


# In[7]:


# Menampilkan outliers pada atribut LT
outliers_LT = df[(df['LT']<bw2) | (df['LT']>ba2)]
outliers_LT.info()


# In[8]:


q13,q33=np.percentile(df['HARGA'], [25,75])
s3 = q33-q13
ba3 = q33+(1.5*s3)
bw3 = q13-(1.5*s3)
print(q13)
print(q33)
print(s3)
print(ba3)
print(bw3)


# In[9]:


# Menampilkan outliers pada atribut HARGA
outliers_HARGA = df[(df['HARGA']<bw3) | (df['HARGA']>ba3)]
outliers_HARGA.info()


# In[10]:


merged_data = pd.concat([outliers_HARGA, outliers_LT, outliers_LB]).drop_duplicates()


# In[11]:


q14,q34=np.percentile(merged_data['LT'], [25,75])
s4 = q34-q14
ba4 = q34+(1.5*s4)
bw4 = q14-(1.5*s4)
print(q14)
print(q34)
print(s4)
print(ba4)
print(bw4)


# In[12]:


dt2 = merged_data[(merged_data['LT']<bw4) | (merged_data['LT']>ba4)]
dt2.info()


# In[13]:


q15,q35=np.percentile(merged_data['HARGA'], [25,75])
s5 = q35-q15
ba5 = q35+(1.5*s5)
bw5 = q15-(1.5*s5)
print(q15)
print(q35)
print(s5)
print(ba5)
print(bw5)


# In[14]:


dt3 = merged_data[(merged_data['HARGA']<bw5) | (merged_data['HARGA']>ba5)]
dt3.info()


# In[15]:


final_data = pd.concat([dt2,dt3]).drop_duplicates()
final_data.info()


# In[16]:


df = df[~df['NO'].isin(final_data['NO'])]


# ## Modelling

# In[17]:


y = df['HARGA'].values.reshape(-1,1)
x = df[['LB','LT','KT','KM','GRS']]


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=200)


# In[19]:


mlr = LinearRegression()
mlr.fit(x_train, y_train)


# In[20]:


print("Intercept: ", mlr.intercept_)
print("Coefficients:")


# In[21]:


print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(x, mlr.coef_[0]))


# In[22]:


y_pred_mlr= mlr.predict(x_test)


# In[23]:


y_test_res = y_test.tolist()
y_pred_res = y_pred_mlr.tolist()

mlr_diff = pd.DataFrame({'Actual value': y_test_res, 'Predicted value': y_pred_res})


# In[24]:


diff_percentage = np.abs((np.array(y_test_res) - np.array(y_pred_res)) / np.array(y_pred_res) - 1) * 100

within_threshold_indices = np.where(diff_percentage <= 5)[0]

within_threshold_data = [(y_test_res[i], y_pred_res[i]) for i in within_threshold_indices]

print("Data within threshold:")
for actual_val, pred_val in within_threshold_data:
    print("Actual value: {}, Predicted value: {}".format(actual_val, pred_val))

num_within_threshold = len(within_threshold_indices)
print("Total data within threshold (<= 5%):", num_within_threshold)


# In[25]:


meanAbErr = metrics.mean_absolute_error(y_test_res, y_pred_res)
meanSqErr = metrics.mean_squared_error(y_test_res, y_pred_res)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test_res, y_pred_res))
print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# In[26]:


import math

print("Rumah impian Ary kurang lebih berada pada harga IDR {:,} juta".format(math.floor(mlr.predict([[100, 300, 3, 2, 1]])/1000000)))


# In[29]:


import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression


# In[30]:


def predict_price(lb, lt, kt, km, grs):
    # Memuat model yang telah dilatih sebelumnya
    model = LinearRegression()
    model.fit(df[["LB", "LT", "KT", "KM", "GRS"]], df["HARGA"])
    # Melakukan prediksi harga rumah berdasarkan input pengguna
    prediction = mlr.predict([[lb, lt, kt, km, grs]])
    return prediction[0]


# In[31]:


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


# In[32]:


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


# In[ ]:





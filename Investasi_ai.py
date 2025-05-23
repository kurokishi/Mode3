import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

st.set_page_config(page_title="Prediksi Saham", layout="wide")
st.title("Prediksi Harga Saham (Regresi Linier)")

# Input saham dan harga historis (dummy untuk saat ini)
st.subheader("Input Data Harga Historis Saham")
data = {
    'Tanggal': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'Harga': np.cumsum(np.random.normal(0, 2, 100)) + 100
}
df = pd.DataFrame(data)
df['Tanggal'] = pd.to_datetime(df['Tanggal'])

# Model regresi linier
X = np.arange(len(df)).reshape(-1, 1)
y = df['Harga'].values
model = LinearRegression().fit(X, y)

# Prediksi 30 hari ke depan
n_pred = 30
X_future = np.arange(len(df), len(df) + n_pred).reshape(-1, 1)
predicted = model.predict(X_future)
future_dates = [df['Tanggal'].max() + timedelta(days=i+1) for i in range(n_pred)]
df_pred = pd.DataFrame({'Tanggal': future_dates, 'Harga': predicted})

# Plot hasil
st.subheader("Grafik Prediksi Harga")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Tanggal'], df['Harga'], label='Historis', linewidth=2)
ax.plot(df_pred['Tanggal'], df_pred['Harga'], 'r--', label='Prediksi', linewidth=2)
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga Saham')
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.info("Prediksi ini menggunakan regresi linier sederhana dan bersifat ilustratif. Gunakan data aktual dan model yang lebih kompleks untuk analisis lebih akurat.")

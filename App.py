import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ====================== Helper Functions =========================
def calculate_fair_value(ticker):
    stock = yf.Ticker(ticker)
    earnings = stock.info.get('trailingEps', 1)
    pe_industry_avg = 15
    return earnings * pe_industry_avg

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    ema12 = data.ewm(span=12, adjust=False).mean()
    ema26 = data.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def plot_technical_indicators(df):
    df['SMA20'] = calculate_sma(df['Close'], window=20)
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(df['Close'], label='Close')
    axs[0].plot(df['SMA20'], label='SMA20')
    axs[0].legend()
    axs[0].set_title("Harga & SMA")

    axs[1].plot(df['RSI'], label='RSI', color='purple')
    axs[1].axhline(70, linestyle='--', color='red')
    axs[1].axhline(30, linestyle='--', color='green')
    axs[1].set_title("RSI")

    axs[2].plot(df['MACD'], label='MACD', color='blue')
    axs[2].plot(df['MACD_signal'], label='Signal', color='orange')
    axs[2].set_title("MACD")
    axs[2].legend()

    plt.tight_layout()
    return fig

# Fungsi lain (price_forecast, average_down_strategy, dll) tetap seperti sebelumnya.

# ====================== Streamlit App =========================
st.title("Aplikasi Analisa Saham Profesional")

ticker = st.text_input("Masukkan kode saham (contoh: UNVR.JK)", value="UNVR.JK")
budget = st.number_input("Dana tersedia untuk average down", value=500000, step=100000)
avg_price = st.number_input("Harga rata-rata beli sebelumnya", value=4000)
owned_lots = st.number_input("Jumlah lot dimiliki", value=10)

if st.button("Analisa Saham"):
    df_data = yf.download(ticker, period="6mo")
    fair = calculate_fair_value(ticker)
    advice, pred_price = price_forecast(ticker)
    curr_price = df_data['Close'][-1]
    buy_lots = average_down_strategy(curr_price, avg_price, owned_lots, budget)
    growth_est = 0.12

    st.subheader("1. Harga Wajar Saham")
    st.write(f"Harga Wajar: Rp {fair:,.2f}")

    st.subheader("2. Rekomendasi 1 Bulan Kedepan")
    st.write(f"Rekomendasi: {advice}, Prediksi Harga: Rp {pred_price:,.2f}")

    st.subheader("3. Strategi Average Down")
    st.write(f"Disarankan beli {buy_lots} lot untuk menurunkan rata-rata beli")

    st.subheader("4. Proyeksi Bunga Majemuk")
    for year in [3, 5, 10]:
        st.write(f"{year} Tahun: Rp {compound_growth(budget, growth_est, year):,.2f}")

    st.subheader("5. Analisa Fundamental (BlackRock-style)")
    info = yf.Ticker(ticker).info
    pe = info.get('trailingPE', 15)
    roe = info.get('returnOnEquity', 0.12)
    div_yield = info.get('dividendYield', 0.04)
    rar = risk_adjusted_return(pe, roe, div_yield)
    st.write(f"Risk Adjusted Return: {rar:.4f}")

    st.subheader("6. Valuasi DCF")
    eps = info.get('trailingEps', 200)
    dcf_val = calculate_dcf(eps, growth_rate=0.08, discount_rate=0.1)
    st.write(f"DCF Valuation: Rp {dcf_val:,.2f}")

    st.subheader("7. Screening Saham Undervalued Potensial")
    undervalued = ai_undervalued_stocks(["UNVR.JK", "TLKM.JK", "SIDO.JK", "BBCA.JK", "ANTM.JK"])
    st.write("Saham-saham undervalued berdasarkan AI:", undervalued)

    st.subheader("8. PBV")
    pbv = info.get('priceToBook', 1.5)
    st.write(f"Price to Book Value: {pbv:.2f}")

    st.subheader("9. ESG Score (simulasi)")
    st.write("*Fitur ESG memerlukan integrasi data khusus. Untuk simulasi, skor ESG diasumsikan: 75/100*")

    st.subheader("10. Prediksi AI Harga Saham 3 Bulan Kedepan (LSTM)")
    pred_lstm = predict_lstm(ticker)
    st.write(f"Prediksi harga LSTM (3 bulan): Rp {pred_lstm:,.2f}")

    st.subheader("11. Ranking Saham Berdasarkan RAR")
    df_rank = rank_stocks(["UNVR.JK", "TLKM.JK", "SIDO.JK", "BBCA.JK", "ANTM.JK"])
    st.dataframe(df_rank)

    st.subheader("12. Simulasi Portofolio")
    st.write("*Simulasi sederhana: diversifikasi rata untuk 5 saham dengan CAGR 12%*")
    port_value = compound_growth(budget, 0.12, 5)
    st.write(f"Estimasi nilai portofolio 5 tahun: Rp {port_value:,.2f}")

    st.subheader("13. Grafik Teknis (SMA, RSI, MACD)")
    fig = plot_technical_indicators(df_data)
    st.pyplot(fig)
    

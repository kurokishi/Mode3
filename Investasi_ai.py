# saham_investasi_ai.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Fungsi bantu ---
def load_portfolio():
    return [
        {"Kode": "ADRO", "Lot": 17, "Harga_Beli": 2605},
        {"Kode": "ANTM", "Lot": 15, "Harga_Beli": 1423},
        {"Kode": "BFIN", "Lot": 30, "Harga_Beli": 1080},
        {"Kode": "BJBR", "Lot": 23, "Harga_Beli": 1145},
        {"Kode": "BSSR", "Lot": 10, "Harga_Beli": 4500},
        {"Kode": "PTBA", "Lot": 4, "Harga_Beli": 2400},
        {"Kode": "UNVR", "Lot": 60, "Harga_Beli": 1860},
        {"Kode": "WIIM", "Lot": 35, "Harga_Beli": 871},
        {"Kode": "PGAS", "Lot": 10, "Harga_Beli": 1600},
    ]

def get_current_price(ticker):
    try:
        data = yf.Ticker(ticker + ".JK").history(period="5d")
        return data["Close"].iloc[-1]
    except:
        return None

def analisis_saham(kode, harga_beli, lot):
    current_price = get_current_price(kode)
    if current_price is None:
        return f"{kode}: Data tidak tersedia"
    
    total_modal = harga_beli * lot * 100
    nilai_sekarang = current_price * lot * 100
    selisih = nilai_sekarang - total_modal
    persentase = (selisih / total_modal) * 100
    rekomendasi = "HOLD"
    if persentase >= 15:
        rekomendasi = "JUAL"
    elif persentase <= -15:
        rekomendasi = "BELI TAMBAH"

    return f"{kode}: Harga beli {harga_beli}, harga saat ini {int(current_price)}, P/L: {int(selisih)} ({persentase:.2f}%) → Rekomendasi: {rekomendasi}"

def simulasi_bunga_majemuk(nominal_awal, rate, tahun):
    return nominal_awal * ((1 + rate) ** tahun)

def prediksi_harga_masa_depan(ticker, hari_ke_depan=30):
    try:
        df = yf.Ticker(ticker + ".JK").history(period="6mo")["Close"].reset_index()
        df["Tanggal"] = df["Date"].map(datetime.toordinal)
        model = LinearRegression()
        X = df[["Tanggal"]]
        y = df["Close"]
        model.fit(X, y)
        pred_day = datetime.now().toordinal() + hari_ke_depan
        pred_price = model.predict([[pred_day]])[0]
        return round(pred_price, 2)
    except:
        return None

def alokasi_otomatis(modal, portfolio):
    rekomendasi = []
    harga_terkini = {}
    undervalued = []

    for saham in portfolio:
        kode = saham["Kode"]
        harga_beli = saham["Harga_Beli"]
        lot = saham["Lot"]
        harga_sekarang = get_current_price(kode)
        if harga_sekarang and harga_sekarang < harga_beli:
            persentase_diskon = (harga_beli - harga_sekarang) / harga_beli
            undervalued.append((kode, harga_sekarang, persentase_diskon))

    if not undervalued:
        return ["Tidak ada saham undervalued ditemukan."]

    total_diskon = sum(x[2] for x in undervalued)
    for kode, harga, diskon in undervalued:
        bobot = diskon / total_diskon
        alokasi = (modal * bobot)
        lot_beli = int(alokasi // (harga * 100))
        if lot_beli > 0:
            rekomendasi.append(f"Beli {lot_beli} lot {kode} @Rp{int(harga)} (alokasi Rp{int(lot_beli * harga * 100):,})")

    return rekomendasi if rekomendasi else ["Modal tidak cukup untuk pembelian minimal 1 lot."]

# --- Streamlit UI ---
st.set_page_config(page_title="AI Saham Pintar", layout="wide")
st.title("📈 Aplikasi Analisis & Strategi Saham")

st.header("1. Portofolio Anda")
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()

df = pd.DataFrame(st.session_state.portfolio)
st.dataframe(df)

st.header("2. Analisis Saham Saat Ini + Rekomendasi AI")
for saham in st.session_state.portfolio:
    analisis = analisis_saham(saham['Kode'], saham['Harga_Beli'], saham['Lot'])
    st.write(analisis)

st.header("3. Strategi Penambahan Saham Berdasarkan Modal Baru")
modal = st.number_input("Masukkan modal baru (Rp)", min_value=0, step=100000)
if modal:
    st.subheader("📊 Rekomendasi Alokasi Otomatis untuk Saham Undervalued")
    rekomendasi = alokasi_otomatis(modal, st.session_state.portfolio)
    for r in rekomendasi:
        st.write("- ", r)

st.header("4. Simulasi Bunga Majemuk Portofolio")
total_nilai = 0
for saham in st.session_state.portfolio:
    price_now = get_current_price(saham['Kode'])
    if price_now:
        total_nilai += price_now * saham['Lot'] * 100
rate_estimasi = st.slider("Estimasi return tahunan (%)", min_value=0.0, max_value=30.0, value=10.0) / 100

if total_nilai > 0:
    st.markdown(f"### Nilai saat ini: Rp {int(total_nilai):,}")
    for tahun in [3, 5, 7, 10]:
        nilai_proj = simulasi_bunga_majemuk(total_nilai, rate_estimasi, tahun)
        st.write(f"Proyeksi dalam {tahun} tahun: Rp {int(nilai_proj):,}")

st.header("5. Edit Portofolio")
st.subheader("Tambah Saham Baru")
with st.form("form_tambah"):
    kode_baru = st.text_input("Kode saham", value="")
    lot_baru = st.number_input("Jumlah lot", min_value=1, step=1)
    harga_beli_baru = st.number_input("Harga beli", min_value=1, step=1)
    submit = st.form_submit_button("Tambah Saham")
    if submit and kode_baru:
        st.session_state.portfolio.append({"Kode": kode_baru.upper(), "Lot": lot_baru, "Harga_Beli": harga_beli_baru})
        st.success(f"Saham {kode_baru.upper()} berhasil ditambahkan.")

st.subheader("Hapus Saham")
kode_hapus = st.selectbox("Pilih saham yang ingin dihapus", [s['Kode'] for s in st.session_state.portfolio])
if st.button("Hapus Saham"):
    st.session_state.portfolio = [s for s in st.session_state.portfolio if s['Kode'] != kode_hapus]
    st.success(f"Saham {kode_hapus} berhasil dihapus.")

st.header("6. Prediksi Harga Masa Depan")
for saham in st.session_state.portfolio:
    prediksi = prediksi_harga_masa_depan(saham['Kode'], hari_ke_depan=30)
    if prediksi:
        st.write(f"📊 Prediksi harga {saham['Kode']} dalam 30 hari ke depan: Rp {prediksi:,}")
    else:
        st.write(f"❌ Tidak bisa memprediksi harga {saham['Kode']}.")

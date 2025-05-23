import streamlit as st
from stock_analysis import AdvancedStockAnalysis

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Saham Canggih",
    page_icon="📈",
    layout="wide"
)

# UI Streamlit
st.title("📈 Analisis Saham Canggih")
st.markdown("""
Aplikasi ini memberikan analisis saham komprehensif dengan:
1. Valuasi harga wajar
2. Rekomendasi beli/tahan/jual
3. Strategi moving average
4. Proyeksi bunga majemuk
5. AI analisis saham undervalued
""")

# Input pengguna
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Masukkan kode saham (contoh: AAPL):", "AAPL").upper()
with col2:
    current_cash = st.number_input("Jumlah dana yang tersedia ($):", min_value=100, value=10000)

if st.button("Analisis Saham"):
    if ticker:
        with st.spinner('Melakukan analisis saham...'):
            analyzer = AdvancedStockAnalysis(ticker, current_cash)
            
            st.header(f"Analisis Saham {ticker}")
            
            # 1. Harga wajar
            st.subheader("1. Valuasi Harga Wajar")
            fair_value = analyzer.get_fair_value()
            current_price = analyzer.historical['Close'].iloc[-1]
            
            col1, col2 = st.columns(2)
            col1.metric("Harga Saat Ini", f"${current_price:.2f}")
            
            if fair_value:
                col2.metric("Harga Wajar Estimasi", f"${fair_value:.2f}", 
                           delta=f"{(current_price - fair_value)/fair_value * 100:.2f}%")
            else:
                col2.warning("Tidak dapat menghitung harga wajar (data tidak lengkap)")
            
            # 2. Rekomendasi
            st.subheader("2. Rekomendasi Beli/Tahan/Jual")
            recs = analyzer.get_recommendation()
            if recs:
                cols = st.columns(3)
                for i, (timeframe, rec) in enumerate(recs.items()):
                    with cols[i]:
                        st.metric(timeframe.replace('_', ' ').title(), rec)
            else:
                st.warning("Tidak dapat memberikan rekomendasi (data tidak lengkap)")
            
            # 3. Strategi Moving Average
            st.subheader("3. Strategi Moving Average")
            ma_strategy = analyzer.moving_average_strategy()
            if ma_strategy:
                st.metric("Aksi", ma_strategy['action'])
                st.info(ma_strategy['message'])
            else:
                st.warning("Tidak dapat menganalisis strategi MA")
            
            # 4. Bunga Majemuk
            st.subheader("4. Proyeksi Bunga Majemuk (asumsi return 12% per tahun)")
            compound = analyzer.compound_interest()
            if compound:
                cols = st.columns(3)
                for i, (period, amount) in enumerate(compound.items()):
                    with cols[i]:
                        st.metric(period.replace('_', ' ').title(), f"${amount:,.2f}")
            else:
                st.warning("Tidak dapat menghitung bunga majemuk")
            
            # 5. AI Undervalued Analysis
            st.subheader("5. AI Analisis Saham Undervalued")
            ai_rec = analyzer.ai_undervalued_analysis()
            if ai_rec:
                st.metric("Rekomendasi AI", ai_rec)
                if analyzer.undervalued_score:
                    st.metric("Undervalued Score", f"{analyzer.undervalued_score:.2f}%")
            else:
                st.warning("Analisis AI tidak tersedia")
            
            # Visualisasi
            st.subheader("Visualisasi Analisis")
            analyzer.visualize_analysis()
    else:
        st.error("Silakan masukkan kode saham")

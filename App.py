import streamlit as st
from stock_analysis import AdvancedStockAnalysisIDX

st.set_page_config(
    page_title="Analisis Saham IDX",
    page_icon="🇮🇩",
    layout="wide"
)

st.title("🇮🇩 Analisis Saham Bursa Efek Indonesia")
st.markdown("""
Aplikasi khusus untuk analisis saham di Bursa Efek Indonesia (IDX)
""")

with st.sidebar:
    st.header("Parameter Saham IDX")
    ticker = st.text_input("Kode Saham (contoh: BBCA):", "BBCA").upper()
    current_cash = st.number_input("Dana Tersedia (Rp):", min_value=1000000, value=100000000)
    annual_return = st.slider("Asumsi Return Tahunan (%)", 5, 20, 12)

if st.button("Analisis Saham IDX"):
    analyzer = AdvancedStockAnalysisIDX(ticker, current_cash)
    
    if analyzer.historical.empty:
        st.error(f"Tidak bisa mendapatkan data {ticker}.JK")
        st.stop()
    
    current_price = analyzer.historical['Close'].iloc[-1]
    current_price_idr = analyzer._convert_to_idr(current_price)
    
    st.header(f"Analisis {ticker}.JK")
    st.metric("Harga Terakhir", f"Rp{current_price_idr:,.0f}")
    
    # Tampilkan data khusus IDX
    idx_data = analyzer.get_idx_fundamental()
    if idx_data:
        cols = st.columns(4)
        cols[0].metric("Dividend Yield", f"{idx_data['dividend_yield']:.2f}%")
        cols[1].metric("Beta", f"{idx_data['beta']:.2f}")
        cols[2].metric("Volume Rata2", f"{idx_data['avg_volume']:,.0f}")
        cols[3].metric("Market Cap", f"Rp{analyzer._convert_to_idr(idx_data['market_cap']/1e12:.2f}T")
    
    # ... (tab-tab lainnya sama, tapi konversi ke IDR)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
import pickle
import os

# Konfigurasi awal
st.set_page_config(layout="wide", page_title="Stock Investment Analyzer")

# Fungsi untuk mendapatkan data saham
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker + ".JK")
    hist = stock.history(period=period)
    return stock, hist

# Fungsi analisis fundamental
def fundamental_analysis(stock):
    try:
        info = stock.info
        per = info.get('trailingPE', 'N/A')
        pbv = info.get('priceToBook', 'N/A')
        dy = info.get('dividendYield', 'N/A') * 100 if info.get('dividendYield') else 'N/A'
        
        # Analisis sederhana valuasi
        valuation = "Fairly Valued"
        if isinstance(per, float):
            if per < 10:
                valuation = "Undervalued"
            elif per > 20:
                valuation = "Overvalued"
                
        return {
            "PER": per,
            "PBV": pbv,
            "Dividend Yield (%)": dy,
            "Valuation": valuation
        }
    except:
        return {
            "PER": "Error",
            "PBV": "Error",
            "Dividend Yield (%)": "Error",
            "Valuation": "Error"
        }

# Fungsi analisis teknikal
def technical_analysis(hist):
    if hist.empty:
        return {
            "RSI": "Error",
            "MACD": "Error",
            "MA50 vs MA200": "Error"
        }
    
    # Hitung RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    last_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else "N/A"
    
    # Hitung MACD
    ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
    ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_status = "Bullish" if macd.iloc[-1] > signal.iloc[-1] else "Bearish"
    
    # Hitung Moving Average
    ma50 = hist['Close'].rolling(window=50).mean()
    ma200 = hist['Close'].rolling(window=200).mean()
    ma_status = "Golden Cross" if ma50.iloc[-1] > ma200.iloc[-1] else "Death Cross"
    
    return {
        "RSI": f"{last_rsi:.2f}" if isinstance(last_rsi, float) else last_rsi,
        "RSI Interpretation": "Overbought (>70)" if isinstance(last_rsi, float) and last_rsi > 70 else 
                              "Oversold (<30)" if isinstance(last_rsi, float) and last_rsi < 30 else "Neutral",
        "MACD": macd_status,
        "MA50 vs MA200": ma_status
    }

# Fungsi proyeksi harga
def price_projection(hist):
    if len(hist) < 30:
        return {"6 Month": "Insufficient Data", "12 Month": "Insufficient Data"}
    
    # Siapkan data untuk model
    X = np.array(range(len(hist))).reshape(-1, 1)
    y = hist['Close'].values
    
    # Latih model regresi linear
    model = LinearRegression()
    model.fit(X, y)
    
    # Prediksi 6 dan 12 bulan ke depan
    future_6m = len(hist) + 126  # ~6 bulan trading days
    future_12m = len(hist) + 252  # ~12 bulan trading days
    
    pred_6m = model.predict(np.array([[future_6m]]))[0]
    pred_12m = model.predict(np.array([[future_12m]]))[0]
    
    return {
        "6 Month": f"{pred_6m:,.2f}",
        "12 Month": f"{pred_12m:,.2f}"
    }

# Fungsi untuk plot candlestick
def plot_candlestick(hist):
    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'])])
    
    fig.update_layout(title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# Fungsi untuk analisis portofolio
def analyze_portfolio(portfolio):
    analysis = {}
    total_value = 0
    total_investment = 0
    
    for item in portfolio:
        ticker = item['ticker']
        lots = item['lots']
        buy_price = item['buy_price']
        
        stock, hist = get_stock_data(ticker)
        current_price = hist['Close'].iloc[-1] if not hist.empty else 0
        
        # Hitung nilai saat ini
        shares = lots * 100  # 1 lot = 100 shares
        current_value = shares * current_price
        investment = shares * buy_price
        profit_loss = current_value - investment
        profit_loss_pct = (profit_loss / investment) * 100 if investment != 0 else 0
        
        # Analisis
        funda = fundamental_analysis(stock)
        tech = technical_analysis(hist)
        projection = price_projection(hist)
        
        analysis[ticker] = {
            "Lots": lots,
            "Buy Price": f"{buy_price:,.2f}",
            "Current Price": f"{current_price:,.2f}",
            "Current Value": f"{current_value:,.2f}",
            "Investment": f"{investment:,.2f}",
            "P/L": f"{profit_loss:,.2f}",
            "P/L %": f"{profit_loss_pct:.2f}%",
            "Valuation": funda['Valuation'],
            "PER": funda['PER'],
            "PBV": funda['PBV'],
            "Dividend Yield": funda['Dividend Yield (%)'],
            "RSI": tech['RSI'],
            "RSI Interpretation": tech['RSI Interpretation'],
            "MACD": tech['MACD'],
            "MA Status": tech['MA50 vs MA200'],
            "6 Month Projection": projection['6 Month'],
            "12 Month Projection": projection['12 Month'],
            "Recommendation": generate_recommendation(funda, tech, profit_loss_pct)
        }
        
        total_value += current_value
        total_investment += investment
    
    portfolio_return = ((total_value - total_investment) / total_investment * 100) if total_investment != 0 else 0
    
    return analysis, total_value, total_investment, portfolio_return

# Fungsi untuk generate rekomendasi
def generate_recommendation(funda, tech, profit_loss_pct):
    if funda['Valuation'] == "Error" or tech['RSI'] == "Error":
        return "Data Error"
    
    recommendations = []
    
    # Berdasarkan valuasi
    if funda['Valuation'] == "Undervalued":
        recommendations.append("Consider adding more (Undervalued)")
    elif funda['Valuation'] == "Overvalued":
        recommendations.append("Consider reducing position (Overvalued)")
    
    # Berdasarkan RSI
    if isinstance(tech['RSI'], str) and tech['RSI'] != "N/A":
        rsi = float(tech['RSI'])
        if rsi > 70:
            recommendations.append("Overbought - Consider taking profits")
        elif rsi < 30:
            recommendations.append("Oversold - Potential buying opportunity")
    
    # Berdasarkan MACD
    if tech['MACD'] == "Bearish":
        recommendations.append("MACD Bearish - Be cautious")
    
    # Berdasarkan P/L
    if profit_loss_pct > 20:
        recommendations.append("Significant profit - Consider partial profit taking")
    elif profit_loss_pct < -15:
        recommendations.append("Significant loss - Review fundamentals")
    
    if not recommendations:
        return "Hold - No strong signals"
    
    return " | ".join(recommendations)

# Fungsi untuk rekomendasi alokasi modal baru
def recommend_allocation(new_capital, portfolio_analysis):
    if new_capital <= 0:
        return {}
    
    # Prioritaskan saham yang undervalued dengan RSI rendah
    scores = {}
    for ticker, data in portfolio_analysis.items():
        score = 0
        
        # Valuasi
        if data['Valuation'] == "Undervalued":
            score += 3
        elif data['Valuation'] == "Fairly Valued":
            score += 1
        
        # RSI
        if data['RSI Interpretation'] == "Oversold (<30)":
            score += 2
        elif data['RSI Interpretation'] == "Neutral":
            score += 1
        
        # Dividend Yield
        try:
            if isinstance(data['Dividend Yield'], str) and data['Dividend Yield'] != "N/A":
                dy = float(data['Dividend Yield'])
                if dy > 5:
                    score += 2
                elif dy > 3:
                    score += 1
        except:
            pass
        
        scores[ticker] = score
    
    # Normalisasi skor
    total_score = sum(scores.values())
    if total_score == 0:
        return {ticker: 0 for ticker in scores}
    
    allocation = {ticker: (score / total_score) * new_capital for ticker, score in scores.items()}
    
    return allocation

# Fungsi untuk simulasi bunga majemuk
def compound_interest_simulation(initial_value, annual_return, years, monthly_add=0):
    results = []
    current_value = initial_value
    
    for year in range(1, years + 1):
        yearly_add = monthly_add * 12
        current_value = current_value * (1 + annual_return / 100) + yearly_add
        results.append({
            "Year": year,
            "Value": current_value
        })
    
    return results

# Fungsi untuk load/save portofolio
def save_portfolio(portfolio):
    with open('portfolio.pkl', 'wb') as f:
        pickle.dump(portfolio, f)

def load_portfolio():
    if os.path.exists('portfolio.pkl'):
        with open('portfolio.pkl', 'rb') as f:
            return pickle.load(f)
    return []

# Main App
def main():
    st.title("📊 Stock Investment Analyzer")
    st.markdown("""
    Aplikasi untuk menganalisis portofolio saham, memberikan rekomendasi strategi, 
    dan memproyeksikan pertumbuhan investasi.
    """)
    
    # Load atau inisialisasi portofolio
    portfolio = load_portfolio()
    
    # Sidebar untuk input/edit portofolio
    with st.sidebar:
        st.header("Portfolio Management")
        
        with st.expander("Add/Edit Stock"):
            ticker = st.text_input("Stock Code (e.g., BBCA)", "").upper()
            lots = st.number_input("Number of Lots", min_value=0.0, value=1.0, step=0.5)
            buy_price = st.number_input("Buy Price per Share", min_value=0, value=1000)
            
            if st.button("Add to Portfolio"):
                # Cek apakah saham sudah ada di portofolio
                found = False
                for item in portfolio:
                    if item['ticker'] == ticker:
                        item['lots'] += lots
                        found = True
                        break
                
                if not found:
                    portfolio.append({
                        'ticker': ticker,
                        'lots': lots,
                        'buy_price': buy_price
                    })
                
                save_portfolio(portfolio)
                st.success(f"Added {lots} lots of {ticker} to portfolio")
        
        with st.expander("Remove Stock"):
            if portfolio:
                to_remove = st.selectbox("Select stock to remove", 
                                       [f"{item['ticker']} ({item['lots']} lots)" for item in portfolio])
                if st.button("Remove Selected"):
                    index = [f"{item['ticker']} ({item['lots']} lots)" for item in portfolio].index(to_remove)
                    removed = portfolio.pop(index)
                    save_portfolio(portfolio)
                    st.success(f"Removed {removed['lots']} lots of {removed['ticker']}")
            else:
                st.warning("Portfolio is empty")
        
        st.markdown("---")
        new_capital = st.number_input("Additional Capital to Invest (IDR)", min_value=0, value=5000000)
        st.markdown("---")
        st.markdown("Created with ❤️ for smart investors")

    # Tab utama
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Analysis", "Allocation Recommendation", "Compound Simulation", "Stock Details"])

    with tab1:
        st.header("Current Portfolio Analysis")
        
        if not portfolio:
            st.warning("Your portfolio is empty. Add stocks using the sidebar.")
        else:
            # Analisis portofolio
            analysis, total_value, total_investment, portfolio_return = analyze_portfolio(portfolio)
            
            # Tampilkan summary
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Investment", f"Rp{total_investment:,.2f}")
            col2.metric("Current Value", f"Rp{total_value:,.2f}")
            col3.metric("Portfolio Return", f"{portfolio_return:.2f}%", 
                       delta_color="inverse" if portfolio_return < 0 else "normal")
            
            # Tampilkan detail saham
            st.subheader("Stock Details")
            df = pd.DataFrame.from_dict(analysis, orient='index')
            st.dataframe(df, use_container_width=True)
            
            # Grafik komposisi portofolio
            st.subheader("Portfolio Composition")
            fig, ax = plt.subplots()
            df['Current Value Num'] = df['Current Value'].str.replace(',', '').astype(float)
            ax.pie(df['Current Value Num'], labels=df.index, autopct='%1.1f%%')
            st.pyplot(fig)

    with tab2:
        st.header("Allocation Recommendation for New Capital")
        
        if not portfolio:
            st.warning("Your portfolio is empty. Add stocks using the sidebar.")
        elif new_capital <= 0:
            st.warning("Please enter a positive amount for additional capital.")
        else:
            analysis, _, _, _ = analyze_portfolio(portfolio)
            allocation = recommend_allocation(new_capital, analysis)
            
            if not allocation:
                st.error("Could not generate allocation recommendations.")
            else:
                st.write(f"Recommended allocation for Rp{new_capital:,.2f}:")
                
                # Konversi ke dataframe untuk tampilan lebih baik
                alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Amount'])
                alloc_df['Percentage'] = (alloc_df['Amount'] / new_capital) * 100
                alloc_df['Amount'] = alloc_df['Amount'].apply(lambda x: f"Rp{x:,.2f}")
                alloc_df['Percentage'] = alloc_df['Percentage'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(alloc_df.sort_values('Percentage', ascending=False), use_container_width=True)
                
                # Grafik alokasi
                st.subheader("Allocation Visualization")
                fig, ax = plt.subplots()
                alloc_values = [float(x.replace('Rp', '').replace(',', '')) for x in alloc_df['Amount']]
                ax.pie(alloc_values, labels=alloc_df.index, autopct='%1.1f%%')
                st.pyplot(fig)

    with tab3:
        st.header("Compound Growth Simulation")
        
        if not portfolio:
            st.warning("Your portfolio is empty. Add stocks using the sidebar.")
        else:
            analysis, total_value, _, _ = analyze_portfolio(portfolio)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                initial_value = st.number_input("Initial Value (IDR)", min_value=0, value=int(total_value))
            with col2:
                annual_return = st.number_input("Expected Annual Return (%)", min_value=-100.0, max_value=1000.0, value=12.0)
            with col3:
                monthly_add = st.number_input("Monthly Additional Investment (IDR)", min_value=0, value=1000000)
            
            years = st.slider("Years to Project", 1, 20, 10)
            
            if st.button("Run Simulation"):
                results = compound_interest_simulation(initial_value, annual_return, years, monthly_add)
                
                # Tampilkan tabel
                st.subheader("Projection Results")
                df = pd.DataFrame(results)
                df['Value'] = df['Value'].apply(lambda x: f"Rp{x:,.2f}")
                st.dataframe(df.set_index('Year'), use_container_width=True)
                
                # Tampilkan grafik
                st.subheader("Growth Visualization")
                fig, ax = plt.subplots()
                ax.plot([r['Year'] for r in results], [r['Value'] for r in results], marker='o')
                ax.set_xlabel("Year")
                ax.set_ylabel("Portfolio Value (IDR)")
                ax.grid(True)
                st.pyplot(fig)

    with tab4:
        st.header("Detailed Stock Analysis")
        
        if not portfolio:
            st.warning("Your portfolio is empty. Add stocks using the sidebar.")
        else:
            selected_ticker = st.selectbox("Select Stock", [item['ticker'] for item in portfolio])
            
            if selected_ticker:
                stock, hist = get_stock_data(selected_ticker, period="2y")
                
                if hist.empty:
                    st.error(f"Could not retrieve data for {selected_ticker}")
                else:
                    # Tampilkan info dasar
                    st.subheader(f"{selected_ticker} - Current Price: Rp{hist['Close'].iloc[-1]:,.2f}")
                    
                    # Tab untuk analisis
                    detail_tab1, detail_tab2, detail_tab3 = st.tabs(["Fundamental", "Technical", "Chart"])
                    
                    with detail_tab1:
                        st.subheader("Fundamental Analysis")
                        funda = fundamental_analysis(stock)
                        st.write(pd.DataFrame.from_dict(funda, orient='index', columns=['Value']))
                        
                        # Analisis tambahan
                        st.markdown("### AI Commentary")
                        if funda['Valuation'] == "Undervalued":
                            st.success("This stock appears to be undervalued based on its P/E ratio. It might be a good opportunity to accumulate more shares.")
                        elif funda['Valuation'] == "Overvalued":
                            st.warning("This stock appears to be overvalued based on its P/E ratio. Consider taking profits or waiting for a better entry point.")
                        else:
                            st.info("This stock appears to be fairly valued based on its P/E ratio.")
                        
                        if isinstance(funda['Dividend Yield (%)'], float) and funda['Dividend Yield (%)'] > 5:
                            st.success(f"High dividend yield ({funda['Dividend Yield (%)']}%) - attractive for income investors.")
                    
                    with detail_tab2:
                        st.subheader("Technical Analysis")
                        tech = technical_analysis(hist)
                        st.write(pd.DataFrame.from_dict(tech, orient='index', columns=['Value']))
                        
                        # Analisis proyeksi
                        st.markdown("### Price Projection")
                        projection = price_projection(hist)
                        st.write(pd.DataFrame.from_dict(projection, orient='index', columns=['Projected Price']))
                    
                    with detail_tab3:
                        st.subheader("Price Chart")
                        plot_candlestick(hist)
                        
                        # Grafik Moving Average
                        fig, ax = plt.subplots()
                        ax.plot(hist.index, hist['Close'], label='Close Price')
                        ax.plot(hist.index, hist['Close'].rolling(50).mean(), label='50-day MA')
                        ax.plot(hist.index, hist['Close'].rolling(200).mean(), label='200-day MA')
                        ax.set_title(f"{selected_ticker} Price with Moving Averages")
                        ax.legend()
                        st.pyplot(fig)

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import ta

plt.style.use('fivethirtyeight')

class AdvancedStockAnalysis:
    def __init__(self, ticker, current_cash):
        self.ticker = ticker
        self.current_cash = current_cash
        self.stock = yf.Ticker(ticker)
        self.historical = self.stock.history(period="10y")
        self.fair_value = None
        self.undervalued_score = None
        
    def get_fair_value(self):
        """Menghitung harga wajar saham menggunakan pendekatan multipel"""
        try:
            # Analisis Fundamental - Pendekatan BlackRock-like
            info = self.stock.info
            
            # Metode 1: Discounted Cash Flow (DCF)
            if 'trailingEps' in info and 'pegRatio' in info:
                eps = info['trailingEps']
                growth = 1/ info['pegRatio'] if info['pegRatio'] > 0 else 0.05  # default growth 5%
                dcf_value = eps * (8.5 + 2 * growth)  # Formula Graham sederhana
            else:
                dcf_value = None
                
            # Metode 2: Price to Book Value
            if 'priceToBook' in info and 'bookValue' in info:
                sector_pb = info.get('priceToBook', 3)  # default sector PB 3
                pb_value = info['bookValue'] * sector_pb
            else:
                pb_value = None
                
            # Metode 3: EV/EBITDA
            if 'enterpriseValue' in info and 'ebitda' in info and info['ebitda'] > 0:
                sector_ev_ebitda = 12  # rata-rata sektor
                ebitda_value = (info['enterpriseValue'] / info['ebitda']) * sector_ev_ebitda
            else:
                ebitda_value = None
                
            # Gabungkan semua metode valuasi
            values = [v for v in [dcf_value, pb_value, ebitda_value] if v is not None]
            self.fair_value = np.mean(values) if values else None
            
            return self.fair_value
            
        except Exception as e:
            st.error(f"Error in fair value calculation: {e}")
            return None
    
    def get_recommendation(self):
        """Memberikan rekomendasi beli/tahan/jual untuk berbagai timeframe"""
        try:
            # Analisis Teknikal - Moving Average Crossover
            df = self.historical.copy()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()
            
            current_price = df['Close'].iloc[-1]
            ma50 = df['MA50'].iloc[-1]
            ma200 = df['MA200'].iloc[-1]
            
            # Analisis Momentum
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            rsi = df['RSI'].iloc[-1]
            
            # Prediksi ARIMA untuk 1 minggu, 1 bulan, 3 bulan
            model = ARIMA(df['Close'], order=(5,1,0))
            model_fit = model.fit()
            
            # Prediksi
            forecast = model_fit.forecast(steps=63)  # 3 bulan (21 hari/bulan)
            week_forecast = forecast[4]
            month_forecast = forecast[20]
            quarter_forecast = forecast[62]
            
            # Buat rekomendasi
            recommendations = {}
            
            # 1 Minggu
            if week_forecast > current_price * 1.02:
                recommendations['1_week'] = 'BUY'
            elif week_forecast < current_price * 0.98:
                recommendations['1_week'] = 'SELL'
            else:
                recommendations['1_week'] = 'HOLD'
                
            # 1 Bulan
            if month_forecast > current_price * 1.05:
                recommendations['1_month'] = 'BUY'
            elif month_forecast < current_price * 0.95:
                recommendations['1_month'] = 'SELL'
            else:
                recommendations['1_month'] = 'HOLD'
                
            # 3 Bulan
            if quarter_forecast > current_price * 1.10:
                recommendations['3_month'] = 'BUY'
            elif quarter_forecast < current_price * 0.90:
                recommendations['3_month'] = 'SELL'
            else:
                recommendations['3_month'] = 'HOLD'
                
            return recommendations
            
        except Exception as e:
            st.error(f"Error in recommendation: {e}")
            return None
    
    def moving_average_strategy(self):
        """Strategi moving average dengan dana saat ini"""
        try:
            df = self.historical.copy()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()
            
            current_price = df['Close'].iloc[-1]
            ma50 = df['MA50'].iloc[-1]
            ma200 = df['MA200'].iloc[-1]
            
            # Signal beli ketika MA50 > MA200 dan harga > MA50
            if current_price > ma50 and ma50 > ma200:
                action = "BUY"
                shares = self.current_cash // current_price
                cash_left = self.current_cash - (shares * current_price)
                message = f"Beli {shares} saham. Sisa dana: {cash_left:.2f}"
            else:
                action = "HOLD"
                message = "Tahan dana tunai, tunggu sinyal beli"
                
            return {'action': action, 'message': message}
            
        except Exception as e:
            st.error(f"Error in MA strategy: {e}")
            return None
    
    def compound_interest(self, annual_return=0.12):
        """Menghitung bunga majemuk untuk 3,5,10 tahun"""
        try:
            periods = {
                '3_year': 3,
                '5_year': 5,
                '10_year': 10
            }
            
            results = {}
            for name, years in periods.items():
                amount = self.current_cash * (1 + annual_return) ** years
                results[name] = amount
                
            return results
            
        except Exception as e:
            st.error(f"Error in compound calculation: {e}")
            return None
    
    def ai_undervalued_analysis(self):
        """AI untuk mengidentifikasi saham undervalued dengan potensi tinggi"""
        try:
            # Dapatkan data komparatif sektor
            sector = self.stock.info.get('sector', 'Technology')
            competitors = self.get_sector_peers(sector)
            
            # Kumpulkan data untuk model
            X, y = self.prepare_ai_data(competitors)
            
            # Latih model Random Forest
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train_scaled, y_train)
            
            # Prediksi untuk saham kita
            our_features = self.get_stock_features()
            our_features_scaled = scaler.transform([our_features])
            predicted_value = model.predict(our_features_scaled)[0]
            
            # Hitung undervalued score
            current_price = self.historical['Close'].iloc[-1]
            self.undervalued_score = (predicted_value - current_price) / current_price * 100
            
            if self.undervalued_score > 20:
                return "STRONG BUY (Undervalued dengan potensi tinggi)"
            elif self.undervalued_score > 10:
                return "BUY (Undervalued)"
            elif self.undervalued_score > 0:
                return "HOLD (Sedikit undervalued)"
            else:
                return "SELL (Overvalued)"
                
        except Exception as e:
            st.error(f"Error in AI analysis: {e}")
            return None
    
    def get_sector_peers(self, sector):
        """Dapatkan daftar saham sejenis dalam sektor yang sama (simplifikasi)"""
        if sector == 'Technology':
            return ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA']
        elif sector == 'Financial':
            return ['JPM', 'BAC', 'WFC', 'C', 'GS']
        else:
            return ['AAPL', 'MSFT', 'JPM', 'WMT', 'PG']
    
    def prepare_ai_data(self, competitors):
        """Persiapkan data untuk model AI"""
        features = []
        targets = []
        
        for ticker in competitors:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5y")
                info = stock.info
                
                # Fitur fundamental
                pe = info.get('trailingPE', 0)
                pb = info.get('priceToBook', 0)
                ps = info.get('priceToSalesTrailing12Months', 0)
                debt_equity = info.get('debtToEquity', 0)
                roe = info.get('returnOnEquity', 0)
                
                # Fitur teknikal
                returns_1y = (hist['Close'].iloc[-1] / hist['Close'].iloc[-252]) - 1 if len(hist) > 252 else 0
                volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                
                # Fitur momentum
                rsi = ta.momentum.RSIIndicator(hist['Close']).rsi().iloc[-1]
                
                # Target - Harga saat ini (model akan memprediksi harga 'wajar')
                price = hist['Close'].iloc[-1]
                
                features.append([pe, pb, ps, debt_equity, roe, returns_1y, volatility, rsi])
                targets.append(price)
                
            except:
                continue
                
        return np.array(features), np.array(targets)
    
    def get_stock_features(self):
        """Dapatkan fitur untuk saham kita"""
        info = self.stock.info
        hist = self.historical
        
        pe = info.get('trailingPE', 0)
        pb = info.get('priceToBook', 0)
        ps = info.get('priceToSalesTrailing12Months', 0)
        debt_equity = info.get('debtToEquity', 0)
        roe = info.get('returnOnEquity', 0)
        
        returns_1y = (hist['Close'].iloc[-1] / hist['Close'].iloc[-252]) - 1 if len(hist) > 252 else 0
        volatility = hist['Close'].pct_change().std() * np.sqrt(252)
        rsi = ta.momentum.RSIIndicator(hist['Close']).rsi().iloc[-1]
        
        return [pe, pb, ps, debt_equity, roe, returns_1y, volatility, rsi]
    
    def visualize_analysis(self):
        """Visualisasi analisis"""
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot harga historis dengan MA
        axs[0, 0].plot(self.historical['Close'], label='Harga', alpha=0.5)
        axs[0, 0].plot(self.historical['MA50'], label='MA50')
        axs[0, 0].plot(self.historical['MA200'], label='MA200')
        axs[0, 0].set_title(f'{self.ticker} Price dengan Moving Average')
        axs[0, 0].legend()
        
        # Plot RSI
        rsi = ta.momentum.RSIIndicator(self.historical['Close']).rsi()
        axs[0, 1].plot(rsi, label='RSI')
        axs[0, 1].axhline(70, color='r', linestyle='--')
        axs[0, 1].axhline(30, color='g', linestyle='--')
        axs[0, 1].set_title('Relative Strength Index (RSI)')
        axs[0, 1].legend()
        
        # Plot distribusi harga
        returns = self.historical['Close'].pct_change().dropna()
        axs[1, 0].hist(returns, bins=50, density=True, alpha=0.6)
        axs[1, 0].set_title('Distribusi Return Harian')
        
        # Plot valuasi
        current_price = self.historical['Close'].iloc[-1]
        if self.fair_value:
            axs[1, 1].bar(['Current', 'Fair'], [current_price, self.fair_value], color=['blue', 'green'])
            axs[1, 1].set_title('Perbandingan Harga Saat ini vs Wajar')
        else:
            axs[1, 1].text(0.5, 0.5, 'Data valuasi tidak tersedia', ha='center')
        
        plt.tight_layout()
        st.pyplot(fig)

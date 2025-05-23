import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import ta
import streamlit as st

plt.style.use('fivethirtyeight')

class AdvancedStockAnalysisIDX:
    def __init__(self, ticker, current_cash):
        # Auto-append .JK untuk saham IDX
        if not ticker.endswith('.JK') and not ticker.startswith('USDIDR'):
            ticker += '.JK'
            
        self.ticker = ticker
        self.current_cash = current_cash
        self.stock = yf.Ticker(ticker)
        
        try:
            self.historical = self.stock.history(period="10y")
            if self.historical.empty:
                st.error(f"Data {ticker} tidak ditemukan. Pastikan kode saham benar (contoh: BBCA.JK)")
                return
            
            self._calculate_technical_indicators()
            self.fair_value = None
            self.undervalued_score = None
            
        except Exception as e:
            st.error(f"Error inisialisasi: {str(e)}")
            self.historical = pd.DataFrame()
    
    def _convert_to_idr(self, usd_amount):
        """Konversi USD ke IDR"""
        try:
            usd_idr = yf.Ticker("USDIDR=X").history(period='1d')['Close'].iloc[-1]
            return usd_amount * usd_idr
        except:
            return usd_amount * 16000  # Fallback rate
    
    def _calculate_technical_indicators(self):
        """Menghitung indikator teknikal"""
        if not self.historical.empty:
            self.historical['MA20'] = self.historical['Close'].rolling(window=20, min_periods=1).mean()
            self.historical['MA50'] = self.historical['Close'].rolling(window=50, min_periods=1).mean()
            self.historical['MA200'] = self.historical['Close'].rolling(window=200, min_periods=1).mean()
            self.historical['RSI'] = ta.momentum.RSIIndicator(self.historical['Close']).rsi()
    
    def get_fair_value(self):
        """Valuasi dengan multiple khas Indonesia"""
        try:
            info = self.stock.info
            
            # Adjust untuk pasar Indonesia
            dcf_value = None
            if 'trailingEps' in info:
                eps = info['trailingEps']
                # Pertumbuhan lebih konservatif untuk IDX
                growth = info.get('pegRatio', 0.08)  # Default 8%
                dcf_value = eps * (7 + 1.5 * growth)  # Formula disesuaikan
            
            pb_value = None
            if 'priceToBook' in info:
                pb_value = info['bookValue'] * 2.5  # PB lebih rendah
            
            ebitda_value = None
            if 'enterpriseValue' in info and 'ebitda' in info:
                ebitda_value = (info['enterpriseValue'] / info['ebitda']) * 8  # EV/EBITDA lebih rendah
            
            values = [v for v in [dcf_value, pb_value, ebitda_value] if v is not None]
            self.fair_value = np.mean(values) if values else None
            
            return self.fair_value
            
        except Exception as e:
            st.error(f"Error valuasi: {str(e)}")
            return None
    
    def get_idx_fundamental(self):
        """Data khusus emiten IDX"""
        try:
            return {
                'dividend_yield': self.stock.info.get('dividendYield', 0) * 100,
                'beta': self.stock.info.get('beta', 1),
                'avg_volume': self.historical['Volume'].mean(),
                'market_cap': self.stock.info.get('marketCap', 0)
            }
        except:
            return None

    # ... (method lainnya sama seperti sebelumnya, tapi gunakan get_sector_peers khusus IDX)

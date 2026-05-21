import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time
import threading

# ============================================================
# CONFIGURATION & ASSETS
# ============================================================
STOCKS_FOREX = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", 
    "USDCHF=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "GC=F", "SI=F", "BTC-USD", "ETH-USD"
]

SYMBOLS_MAP = {
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF",
    "NZDUSD=X": "NZD/USD", "EURGBP=X": "EUR/GBP", "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY", "GC=F": "الذهب (Gold Futures)", "SI=F": "الفضة (Silver Futures)",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"
}

# ============================================================
# DATA ENGINE
# ============================================================
class DataEngine:
    @staticmethod
    @st.cache_data(ttl=60)
    def fetch_data(symbol, period="1d", interval="1m"):
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            if data.empty:
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None

    @staticmethod
    def add_indicators(df):
        # Trend Indicators
        close_series = df['Close'].squeeze()
        high_series = df['High'].squeeze()
        low_series = df['Low'].squeeze()

        df['EMA_20'] = ta.trend.ema_indicator(close_series, window=20)
        df['EMA_50'] = ta.trend.ema_indicator(close_series, window=50)
        df['EMA_200'] = ta.trend.ema_indicator(close_series, window=200)
        
        # Momentum Indicators
        df['RSI'] = ta.momentum.rsi(close_series, window=14)
        
        # Volatility Indicators
        bb = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['ATR'] = ta.volatility.average_true_range(high_series, low_series, close_series, window=14)
        
        # MACD
        macd = ta.trend.MACD(close_series)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        return df

# ============================================================
# SMART MARKET STRUCTURE (SMC) ENGINE
# ============================================================
class SMCEngine:
    @staticmethod
    def detect_structure(df):
        last_close = float(df['Close'].iloc[-1].iloc[0] if isinstance(df['Close'].iloc[-1], pd.Series) else df['Close'].iloc[-1])
        prev_high = float(df['High'].iloc[-20:-1].max().iloc[0] if isinstance(df['High'].iloc[-20:-1].max(), pd.Series) else df['High'].iloc[-20:-1].max())
        prev_low = float(df['Low'].iloc[-20:-1].min().iloc[0] if isinstance(df['Low'].iloc[-20:-1].min(), pd.Series) else df['Low'].iloc[-20:-1].min())
        
        bos_bullish = last_close > prev_high
        bos_bearish = last_close < prev_low
        
        structure = "NEUTRAL"
        if bos_bullish: structure = "BOS BULLISH (صعودي)"
        elif bos_bearish: structure = "BOS BEARISH (هبوطي)"
        
        # Simple Order Block Detection
        ob = "None"
        if bos_bullish:
            ob = "Bullish Order Block Detected"
        elif bos_bearish:
            ob = "Bearish Order Block Detected"
            
        return {
            "structure": structure,
            "order_block": ob,
            "trend": "UP" if last_close > df['EMA_50'].iloc[-1] else "DOWN"
        }

# ============================================================
# AI PREDICTION ENGINE
# ============================================================
class AIEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        df = df.copy().dropna()
        features = pd.DataFrame()
        features['RSI'] = df['RSI'].squeeze()
        features['MACD_Diff'] = df['MACD_Diff'].squeeze()
        features['Price_EMA20'] = (df['Close'].squeeze() - df['EMA_20'].squeeze())
        features['Price_EMA50'] = (df['Close'].squeeze() - df['EMA_50'].squeeze())
        features['Volatility'] = (df['High'].squeeze() - df['Low'].squeeze())
        
        # Target: 1 if next close > current close
        target = (df['Close'].squeeze().shift(-1) > df['Close'].squeeze()).astype(int)[:-1]
        features = features[:-1]
        
        return features, target

    def predict(self, df):
        if len(df) < 50:
            return "NEUTRAL", 50.0
            
        features, target = self.prepare_features(df)
        if len(features) < 10:
            return "NEUTRAL", 50.0
            
        X = self.scaler.fit_transform(features)
        self.model.fit(X, target)
        
        # Predict last row
        def get_val(series, idx=-1):
            val = series.iloc[idx]
            return float(val.iloc[0] if isinstance(val, pd.Series) else val)

        last_features = pd.DataFrame([{
            'RSI': get_val(df['RSI']),
            'MACD_Diff': get_val(df['MACD_Diff']),
            'Price_EMA20': get_val(df['Close']) - get_val(df['EMA_20']),
            'Price_EMA50': get_val(df['Close']) - get_val(df['EMA_50']),
            'Volatility': get_val(df['High']) - get_val(df['Low'])
        }])
        
        X_last = self.scaler.transform(last_features)
        prob = self.model.predict_proba(X_last)[0]
        prediction = "BUY" if prob[1] > prob[0] else "SELL"
        confidence = max(prob) * 100
        
        return prediction, confidence

# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    st.set_page_config(page_title="UltraForex AI Pro", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for a professional look
    st.markdown("""
        <style>
        .main { background-color: #0e1117; color: #ffffff; }
        .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4150; }
        .signal-buy { color: #00ff00; font-weight: bold; font-size: 24px; }
        .signal-sell { color: #ff4b4b; font-weight: bold; font-size: 24px; }
        .header-style { font-size: 40px; font-weight: bold; color: #00d4ff; text-align: center; margin-bottom: 20px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header-style">💎 UltraForex AI Pro v2.0 💎</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Control Panel")
    selected_pair = st.sidebar.selectbox("اختر زوج العملات", STOCKS_FOREX, format_func=lambda x: SYMBOLS_MAP[x])
    timeframe = st.sidebar.selectbox("الفريم الزمني", ["1m", "5m", "15m", "1h", "1d"])
    period_map = {"1m": "1d", "5m": "5d", "15m": "5d", "1h": "1mo", "1d": "max"}
    
    if st.sidebar.button("🔄 تحديث البيانات"):
        st.rerun()

    # Data Loading
    with st.spinner('جاري تحليل السوق...'):
        data = DataEngine.fetch_data(selected_pair, period=period_map[timeframe], interval=timeframe)
        
    if data is not None:
        data = DataEngine.add_indicators(data)
        smc = SMCEngine.detect_structure(data)
        ai_engine = AIEngine()
        prediction, confidence = ai_engine.predict(data)
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        def get_val(series, idx=-1):
            val = series.iloc[idx]
            return float(val.iloc[0] if isinstance(val, pd.Series) else val)
            
        last_price = get_val(data['Close'])
        change = last_price - get_val(data['Close'], -2)
        
        col1.metric("السعر الحالي", f"{last_price:.5f}", f"{change:.5f}")
        col2.metric("RSI (14)", f"{get_val(data['RSI']):.2f}")
        col3.metric("هيكل السوق", smc['structure'])
        col4.metric("الاتجاه العام", smc['trend'])

        def get_val_local(series, idx=-1):
            val = series.iloc[idx]
            return float(val.iloc[0] if isinstance(val, pd.Series) else val)

        # Signal Box
        st.markdown("---")
        sig_col1, sig_col2 = st.columns([1, 2])
        with sig_col1:
            st.subheader("🤖 توصية الذكاء الاصطناعي")
            if prediction == "BUY":
                st.markdown(f'<p class="signal-buy">شراء (BUY) 🟢</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="signal-sell">بيع (SELL) 🔴</p>', unsafe_allow_html=True)
            st.write(f"**نسبة الثقة:** {confidence:.2f}%")
            
            # Risk Management
            atr = data['ATR'].iloc[-1]
            tp = last_price + (atr * 2) if prediction == "BUY" else last_price - (atr * 2)
            sl = last_price - (atr * 1.5) if prediction == "BUY" else last_price + (atr * 1.5)
            
            st.info(f"""
            **إدارة المخاطر:**
            - Entry: {last_price:.5f}
            - Take Profit: {tp:.5f}
            - Stop Loss: {sl:.5f}
            """)

        with sig_col2:
            # Plotly Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.03, subplot_titles=('Price Action', 'RSI'), 
                               row_heights=[0.7, 0.3])

            # Candlestick
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                        low=data['Low'], close=data['Close'], name='Market'), row=1, col=1)
            
            # Indicators
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'].squeeze(), line=dict(color='yellow', width=1), name='EMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'].squeeze(), line=dict(color='cyan', width=1), name='EMA 50'), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'].squeeze(), line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            fig.update_layout(height=600, template="plotly_dark", showlegend=False, 
                              xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        # Market Overview Table
        st.subheader("📊 نظرة عامة على أزواج العملات")
        overview_data = []
        for pair in STOCKS_FOREX:
            p_data = DataEngine.fetch_data(pair, period="1d", interval="15m")
            if p_data is not None:
                curr_p = float(p_data['Close'].iloc[-1].iloc[0] if isinstance(p_data['Close'].iloc[-1], pd.Series) else p_data['Close'].iloc[-1])
                prev_p = float(p_data['Close'].iloc[-2].iloc[0] if isinstance(p_data['Close'].iloc[-2], pd.Series) else p_data['Close'].iloc[-2])
                overview_data.append({
                    "الزوج": SYMBOLS_MAP[pair],
                    "السعر": f"{curr_p:.5f}",
                    "التغير": f"{(curr_p - prev_p):.5f}"
                })
        st.table(pd.DataFrame(overview_data))

    else:
        st.error("تعذر جلب البيانات. يرجى التحقق من اتصال الإنترنت أو الرمز المستخدم.")

    st.sidebar.markdown("---")
    st.sidebar.write("Developed with ❤️ by Manus for Professional Traders")

if __name__ == "__main__":
    main()

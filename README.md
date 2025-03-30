import ccxt
import numpy as np
import pandas as pd
import time
import logging
import os
import streamlit as st
import ta
import plotly.graph_objects as go
from datetime import datetime

# Настройка логов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("binance_bot")

class BinanceTradingBot:
    def __init__(self, api_key=None, api_secret=None, symbol='BTC/USDT', timeframe='1h'):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY") or st.secrets.get("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET") or st.secrets.get("BINANCE_API_SECRET")
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = None
        self.connected = False
        self.in_position = False
        self.buy_price = None
        
        # Параметры стратегии
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.sma_short_period = 20
        self.sma_long_period = 50
        self.stop_loss = 0.05  # 5%
        self.take_profit = 0.10  # 10%

    def connect(self):
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            self.exchange.load_markets()
            self.connected = True
            logger.info("Connected to Binance")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def fetch_ohlcv(self, limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV: {e}")
            return None

    def calculate_indicators(self, df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
        df['sma_short'] = ta.trend.SMAIndicator(df['close'], window=self.sma_short_period).sma_indicator()
        df['sma_long'] = ta.trend.SMAIndicator(df['close'], window=self.sma_long_period).sma_indicator()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        return df

    def get_signals(self, df):
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        buy_signal = (
            (prev['rsi'] < self.rsi_oversold < last['rsi']) |
            (prev['sma_short'] < prev['sma_long'] and last['sma_short'] > last['sma_long']) |
            (prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal'])
        )
        
        sell_signal = (
            (prev['rsi'] > self.rsi_overbought > last['rsi']) |
            (prev['sma_short'] > prev['sma_long'] and last['sma_short'] < last['sma_long']) |
            (prev['macd'] > prev['macd_signal'] and last['macd'] < last['macd_signal'])
        )
        
        return buy_signal, sell_signal

    def execute_order(self, side, amount_percent=0.1):
        try:
            balance = self.exchange.fetch_balance()
            symbol_info = self.exchange.market(self.symbol)
            
            if side == 'buy':
                quote_currency = symbol_info['quote']
                available = balance[quote_currency]['free']
                amount = available * amount_percent
                order = self.exchange.create_market_buy_order(self.symbol, amount)
                self.in_position = True
                self.buy_price = order['average']
                logger.info(f"Buy order executed: {order}")
            else:
                base_currency = symbol_info['base']
                available = balance[base_currency]['free']
                order = self.exchange.create_market_sell_order(self.symbol, available)
                self.in_position = False
                self.buy_price = None
                logger.info(f"Sell order executed: {order}")
                
            return order
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None

# Streamlit UI
def main():
import ccxt
import numpy as np
import pandas as pd
import time
import logging
import os
import streamlit as st
import ta
import plotly.graph_objects as go
from datetime import datetime

# Настройка логов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("binance_bot")

class BinanceTradingBot:
    def __init__(self, api_key=None, api_secret=None, symbol='BTC/USDT', timeframe='1h'):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY") or st.secrets.get("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET") or st.secrets.get("BINANCE_API_SECRET")
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = None
        self.connected = False
        self.in_position = False
        self.buy_price = None
        
        # Параметры стратегии
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.sma_short_period = 20
        self.sma_long_period = 50
        self.stop_loss = 0.05  # 5%
        self.take_profit = 0.10  # 10%

    def connect(self):
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            self.exchange.load_markets()
            self.connected = True
            logger.info("Connected to Binance")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def fetch_ohlcv(self, limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV: {e}")
            return None

    def calculate_indicators(self, df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
        df['sma_short'] = ta.trend.SMAIndicator(df['close'], window=self.sma_short_period).sma_indicator()
        df['sma_long'] = ta.trend.SMAIndicator(df['close'], window=self.sma_long_period).sma_indicator()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        return df

    def get_signals(self, df):
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        buy_signal = (
            (prev['rsi'] < self.rsi_oversold < last['rsi']) |
            (prev['sma_short'] < prev['sma_long'] and last['sma_short'] > last['sma_long']) |
            (prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal'])
        )
        
        sell_signal = (
            (prev['rsi'] > self.rsi_overbought > last['rsi']) |
            (prev['sma_short'] > prev['sma_long'] and last['sma_short'] < last['sma_long']) |
            (prev['macd'] > prev['macd_signal'] and last['macd'] < last['macd_signal'])
        )
        
        return buy_signal, sell_signal

    def execute_order(self, side, amount_percent=0.1):
        try:
            balance = self.exchange.fetch_balance()
            symbol_info = self.exchange.market(self.symbol)
            
            if side == 'buy':
                quote_currency = symbol_info['quote']
                available = balance[quote_currency]['free']
                amount = available * amount_percent
                order = self.exchange.create_market_buy_order(self.symbol, amount)
                self.in_position = True
                self.buy_price = order['average']
                logger.info(f"Buy order executed: {order}")
            else:
                base_currency = symbol_info['base']
                available = balance[base_currency]['free']
                order = self.exchange.create_market_sell_order(self.symbol, available)
                self.in_position = False
                self.buy_price = None
                logger.info(f"Sell order executed: {order}")
                
            return order
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None

# Streamlit UI
def main():
    st.set_page_config(page_title="Binance Bot", layout="wide")
    st.title("📈 Binance Trading Bot")
    
    # Инициализация сессии
    if 'bot' not in st.session_state:
        st.session_state.bot = BinanceTradingBot()
    
    # Сайдбар
    with st.sidebar:
        st.header("🔧 Настройки")
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox("Пара", ["BTC/USDT", "ETH/USDT", "BNB/USDT"])
        with col2:
            timeframe = st.selectbox("Таймфрейм", ["1h", "4h", "1d"])
        
        st.session_state.bot.symbol = symbol
        st.session_state.bot.timeframe = timeframe
        
        if st.button("Подключиться"):
            st.session_state.bot.api_key = api_key
            st.session_state.bot.api_secret = api_secret
            if st.session_state.bot.connect():
                st.success("Успешное подключение!")
            else:
                st.error("Ошибка подключения")
    
    # Основной интерфейс
    tab1, tab2 = st.tabs(["📊 Анализ", "⚙️ Управление"])
    
    with tab1:
        if st.session_state.bot.connected:
            df = st.session_state.bot.fetch_ohlcv()
            if df is not None:
                df = st.session_state.bot.calculate_indicators(df)
                
                # График цены
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'],
                    name="Price"
                ))
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['sma_short'],
                    line=dict(color='blue', width=1),
                    name=f"SMA {st.session_state.bot.sma_short_period}"
                ))
                fig.update_layout(height=500, title=f"{symbol} Price")
                st.plotly_chart(fig, use_container_width=True)
                
                # Сигналы
                buy_signal, sell_signal = st.session_state.bot.get_signals(df)
                st.write(f"🔴 SELL Signal: {sell_signal} | 🟢 BUY Signal: {buy_signal}")
        else:
            st.warning("Сначала подключитесь к Binance")
    
    with tab2:
        if st.session_state.bot.connected:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🟢 Запустить бота"):
                    st.session_state.bot.running = True
                    st.success("Бот запущен")
            with col2:
                if st.button("🔴 Остановить бота"):
                    st.session_state.bot.running = False
                    st.warning("Бот остановлен")
            
            st.write(f"Статус: {'🟢 Работает' if getattr(st.session_state.bot, 'running', False) else '🔴 Остановлен'}")
        else:
            st.warning("Требуется подключение к Binance")

if __name__ == "__main__":
    main()￼Enter    st.set_page_config(page_title="Binance Bot", layout="wide")
    st.title("📈 Binance Trading Bot")
    
    # Инициализация сессии
    if 'bot' not in st.session_state:
        st.session_state.bot = BinanceTradingBot()
    
    # Сайдбар
    with st.sidebar:
        st.header("🔧 Настройки")
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox("Пара", ["BTC/USDT", "ETH/USDT", "BNB/USDT"])
        with col2:
            timeframe = st.selectbox("Таймфрейм", ["1h", "4h", "1d"])
        
        st.session_state.bot.symbol = symbol
        st.session_state.bot.timeframe = timeframe
        
        if st.button("Подключиться"):
            st.session_state.bot.api_key = api_key
            st.session_state.bot.api_secret = api_secret
            if st.session_state.bot.connect():
                st.success("Успешное подключение!")
      else:
                st.error("Ошибка подключения")
    
    # Основной интерфейс
    tab1, tab2 = st.tabs(["📊 Анализ", "⚙️ Управление"])
    
    with tab1:
        if st.session_state.bot.connected:
            df = st.session_state.bot.fetch_ohlcv()
            if df is not None:
                df = st.session_state.bot.calculate_indicators(df)
                
                # График цены
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'],
                    name="Price"
                ))
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['sma_short'],
                    line=dict(color='blue', width=1),
                    name=f"SMA {st.session_state.bot.sma_short_period}"
                ))
                fig.update_layout(height=500, title=f"{symbol} Price")

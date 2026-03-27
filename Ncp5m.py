import logging
import time
import asyncio
from binance.client import Client
import pandas as pd
import talib as ta
from telegram import Bot
from telegram.error import TelegramError
import numpy as np
from typing import Tuple, Optional, List
import config

# Configuration
class Config:
    VWAP_PERIOD = 10  # Kept at 10, ideal for 5m
    ATR_MULTIPLIER_TP = 2.5
    ATR_MULTIPLIER_SL = 1.5
    VOLUME_SPIKE_MULTIPLIER = 1.5  # Lowered from 2 for more volume triggers
    STOCHASTIC_PARAMS = {'fastk_period': 7, 'slowk_period': 3, 'slowd_period': 3}  # Kept fast for 5m
    INTERVAL = '5m'
    LIMIT = 500
    SLEEP_INTERVAL = 180   # 3 minutes

    @staticmethod
    def get_symbols():
        return config.load_active_pairs()

# Credentials
CREDENTIALS = {
    'binance': {
        'api_key': 'qvwSsEuJmI6tZRWGMCHeYtcpqFXgmFaf0sRnh2kuO0XgClHdCh8E306LcmbQ1vLe',
        'api_secret': 'MC4CAQAwBQYDK2VwBCIEIDflCfCUgrDrbxuAaT1Z+oZA6G2vgzG7zJXfggddn3Di'
    },
    'telegram': {
        'token': '7789897244:AAGK79VJxXQVQkxAkowP2HnUNFqqm5EOVE8',
        'chat_id': '5932647180'
    }
}

# Initialize clients
client = Client(CREDENTIALS['binance']['api_key'], CREDENTIALS['binance']['api_secret'])
bot = Bot(token=CREDENTIALS['telegram']['token'])

# Custom Telegram handler for logging
class TelegramHandler(logging.Handler):
    def __init__(self, bot, chat_id):
        super().__init__()
        self.bot = bot
        self.chat_id = chat_id
        self.loop = asyncio.get_event_loop()

    async def emit_async(self, record):
        msg = self.format(record)
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=msg)
                break
            except TelegramError as e:
                if attempt < max_attempts - 1:
                    await asyncio.sleep(5)

    def emit(self, record):
        try:
            asyncio.ensure_future(self.emit_async(record), loop=self.loop)
        except RuntimeError:
            asyncio.run(self.emit_async(record))

# Logging setup
telegram_handler = TelegramHandler(bot, CREDENTIALS['telegram']['chat_id'])
telegram_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('trading_bot_5m.log'),
        logging.StreamHandler(),
        telegram_handler
    ]
)

def fetch_data(symbol: str, interval: str = Config.INTERVAL, limit: int = Config.LIMIT) -> Optional[pd.DataFrame]:
    logging.info(f"Fetching data for {symbol}")
    try:
        klines = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=limit)
        data = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                           'Close time', 'Quote asset volume', 'Number of trades',
                                           'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = pd.to_numeric(data[col])
            
        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data.set_index('Open time', inplace=True)
        logging.info(f"Successfully fetched data for {symbol}")
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_indicators(data: pd.DataFrame) -> dict:
    logging.info("Calculating indicators")
    indicators = {}
    indicators['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    indicators['atr'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    indicators['slowk'], indicators['slowd'] = ta.STOCH(
        data['High'], data['Low'], data['Close'],
        **Config.STOCHASTIC_PARAMS,
        slowk_matype=0, slowd_matype=0
    )
    indicators['volume_ma'] = data['Volume'].rolling(window=Config.VWAP_PERIOD).mean()
    return indicators

def detect_signals(data: pd.DataFrame, indicators: dict) -> Tuple[List[str], Optional[float], Optional[float], Optional[float]]:
    signals = []
    entry, stop_loss, take_profit = None, None, None
    current_price = data['Close'].iloc[-1]
    atr = indicators['atr'].iloc[-1]

    logging.info(f"Current price for analysis: {current_price:.4f}")

    # Candlestick Patterns (relaxed conditions)
    hammer = ta.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
    shooting_star = ta.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
    doji_threshold = (data['High'] - data['Low']).mean() * 0.25  # Increased from 0.1 for 5m
    
    if hammer.iloc[-1] != 0:
        signals.append("Hammer")
        entry = current_price
        stop_loss = data['Low'].iloc[-1] - (atr * 0.5)
        take_profit = current_price + (atr * Config.ATR_MULTIPLIER_TP)

    if shooting_star.iloc[-1] != 0:
        signals.append("Shooting Star")
        entry = current_price
        stop_loss = data['High'].iloc[-1] + (atr * 0.5)
        take_profit = current_price - (atr * Config.ATR_MULTIPLIER_TP)

    if abs(data['Close'].iloc[-1] - data['Open'].iloc[-1]) <= doji_threshold:
        signals.append("Doji")

    # Simplified Engulfing
    if (data['Close'].iloc[-1] > data['Open'].iloc[-1] and 
        data['Close'].iloc[-2] < data['Open'].iloc[-2]):
        signals.append("Bullish Engulfing")
        entry = current_price
        stop_loss = data['Low'].iloc[-2] - (atr * 0.5)
        take_profit = current_price + (atr * Config.ATR_MULTIPLIER_TP)

    elif (data['Close'].iloc[-1] < data['Open'].iloc[-1] and 
          data['Close'].iloc[-2] > data['Open'].iloc[-2]):
        signals.append("Bearish Engulfing")
        entry = current_price
        stop_loss = data['High'].iloc[-2] + (atr * 0.5)
        take_profit = current_price - (atr * Config.ATR_MULTIPLIER_TP)

    morning_star = ta.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
    if morning_star.iloc[-1] != 0:
        signals.append("Morning Star")
        entry = current_price
        stop_loss = data['Low'].iloc[-3] - (atr * 0.5)
        take_profit = current_price + (atr * Config.ATR_MULTIPLIER_TP)

    evening_star = ta.CDLEVENINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
    if evening_star.iloc[-1] != 0:
        signals.append("Evening Star")
        entry = current_price
        stop_loss = data['High'].iloc[-3] + (atr * 0.5)
        take_profit = current_price - (atr * Config.ATR_MULTIPLIER_TP)

    bullish_harami = ta.CDLHARAMI(data['Open'], data['High'], data['Low'], data['Close'])
    if bullish_harami.iloc[-1] > 0 and data['Close'].iloc[-1] > data['Open'].iloc[-1]:
        signals.append("Bullish Harami")
        entry = current_price
        stop_loss = data['Low'].iloc[-2] - (atr * 0.5)
        take_profit = current_price + (atr * Config.ATR_MULTIPLIER_TP)

    if bullish_harami.iloc[-1] < 0 and data['Close'].iloc[-1] < data['Open'].iloc[-1]:
        signals.append("Bearish Harami")
        entry = current_price
        stop_loss = data['High'].iloc[-2] + (atr * 0.5)
        take_profit = current_price - (atr * Config.ATR_MULTIPLIER_TP)

    three_white_soldiers = ta.CDL3WHITESOLDIERS(data['Open'], data['High'], data['Low'], data['Close'])
    if three_white_soldiers.iloc[-1] != 0:
        signals.append("Three White Soldiers")
        entry = current_price
        stop_loss = data['Low'].iloc[-3] - (atr * 0.5)
        take_profit = current_price + (atr * Config.ATR_MULTIPLIER_TP)

    three_black_crows = ta.CDL3BLACKCROWS(data['Open'], data['High'], data['Low'], data['Close'])
    if three_black_crows.iloc[-1] != 0:
        signals.append("Three Black Crows")
        entry = current_price
        stop_loss = data['High'].iloc[-3] + (atr * 0.5)
        take_profit = current_price - (atr * Config.ATR_MULTIPLIER_TP)

    piercing = ta.CDLPIERCING(data['Open'], data['High'], data['Low'], data['Close'])
    if piercing.iloc[-1] != 0:
        signals.append("Piercing Pattern")
        entry = current_price
        stop_loss = data['Low'].iloc[-2] - (atr * 0.5)
        take_profit = current_price + (atr * Config.ATR_MULTIPLIER_TP)

    dark_pool = ta.CDLDARKCLOUDCOVER(data['Open'], data['High'], data['Low'], data['Close'])
    if dark_pool.iloc[-1] != 0:
        signals.append("Dark Pool Cover")
        entry = current_price
        stop_loss = data['High'].iloc[-2] + (atr * 0.5)
        take_profit = current_price - (atr * Config.ATR_MULTIPLIER_TP)

    # Stochastic Signals (widened for 5m)
    if indicators['slowk'].iloc[-1] < 35 and indicators['slowd'].iloc[-1] < 35:
        signals.append("Buy (Stochastic Oversold)")
        if not entry:
            entry = current_price
            stop_loss = entry - (atr * Config.ATR_MULTIPLIER_SL)
            take_profit = entry + (atr * Config.ATR_MULTIPLIER_TP)
    elif indicators['slowk'].iloc[-1] > 65 and indicators['slowd'].iloc[-1] > 65:
        signals.append("Sell (Stochastic Overbought)")
        if not entry:
            entry = current_price
            stop_loss = entry + (atr * Config.ATR_MULTIPLIER_SL)
            take_profit = entry - (atr * Config.ATR_MULTIPLIER_TP)

    # Volume Spike
    if data['Volume'].iloc[-1] > (indicators['volume_ma'].iloc[-1] * Config.VOLUME_SPIKE_MULTIPLIER):
        signals.append("Volume Spike")

    # Fallback trade levels if signals exist but no entry set
    if signals and not entry:
        entry = current_price
        stop_loss = current_price - (atr * Config.ATR_MULTIPLIER_SL) if "Buy" in " ".join(signals) else current_price + (atr * Config.ATR_MULTIPLIER_SL)
        take_profit = current_price + (atr * Config.ATR_MULTIPLIER_TP) if "Buy" in " ".join(signals) else current_price - (atr * Config.ATR_MULTIPLIER_TP)

    logging.info(f"Detected signals: {signals}")
    return signals, entry, stop_loss, take_profit

async def send_to_telegram(message: str) -> None:
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            await bot.send_message(chat_id=CREDENTIALS['telegram']['chat_id'], text=message)
            logging.info(f"Sent message to Telegram: {message}")
            break
        except TelegramError as e:
            logging.error(f"Telegram error (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(5)

async def analyze_symbol(symbol: str) -> None:
    logging.info(f"Starting analysis for {symbol}")
    try:
        data = fetch_data(symbol)
        if data is None:
            logging.warning(f"No data retrieved for {symbol}")
            return

        indicators = calculate_indicators(data)
        signals, entry, stop_loss, take_profit = detect_signals(data, indicators)
        
        position = "above" if data['Close'].iloc[-1] > indicators['vwap'].iloc[-1] else "below"
        signal_strength = abs(indicators['slowk'].iloc[-1] - indicators['slowd'].iloc[-1])

        logging.info(f"{symbol} - Price: {data['Close'].iloc[-1]:.4f}, VWAP: {indicators['vwap'].iloc[-1]:.4f}, Stochastic K: {indicators['slowk'].iloc[-1]:.2f}, D: {indicators['slowd'].iloc[-1]:.2f}")

        if signals:  # Send signal even if only patterns detected
            message = (
                f"{symbol} Analysis (5M):\n" +
                "\n".join(signals) +
                (f"\nEntry: {entry:.4f}\nStop-Loss: {stop_loss:.4f}\nTake-Profit: {take_profit:.4f}" if entry else "") +
                f"\nSignal Strength: {signal_strength:.2f}\nPrice {position} VWAP ({indicators['vwap'].iloc[-1]:.4f})"
            )
            logging.info(f"Actionable signal detected: {message}")
            await send_to_telegram(message)
        else:
            logging.info(f"No actionable signals for {symbol}")
            
    except Exception as e:
        error_msg = f"Error analyzing {symbol}: {e}"
        logging.error(error_msg)
        await send_to_telegram(error_msg)

async def main():
    logging.info("Starting trading bot (5M timeframe)...")
    while True:
        try:
            tasks = [analyze_symbol(symbol) for symbol in Config.get_symbols()]
            await asyncio.gather(*tasks)
            logging.info(f"Completed analysis cycle, sleeping for {Config.SLEEP_INTERVAL} seconds")
            await asyncio.sleep(Config.SLEEP_INTERVAL)
        except Exception as e:
            error_msg = f"Main loop error: {e}"
            logging.error(error_msg)
            await send_to_telegram(error_msg)
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())

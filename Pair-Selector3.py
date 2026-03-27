import logging
import asyncio
import os
import time
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Literal

import pandas as pd
import numpy as np
import talib as ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from telegram import Bot
from telegram.error import TelegramError
from dotenv import load_dotenv

# ────────────────────────────────────────────────
#   CONFIG & CONSTANTS
# ────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

API_KEY    = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN   = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not all([API_KEY, API_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID]):
    logging.critical("Missing required environment variables")
    exit(1)

client = Client(API_KEY, API_SECRET)
bot    = Bot(token=TELEGRAM_TOKEN)

# ─── Trading parameters ───────────────────────────────
MIN_QUOTE_VOLUME     = 1_000_000
TOP_PAIRS_COUNT      = 8
PAIR_UPDATE_INTERVAL = 3600      # 1 hour
CANDLE_INTERVAL      = 300       # 5 min

SLIPPAGE              = 0.0005
TRANSACTION_FEE       = 0.001
MAX_HOLDING_CANDLES   = 360

BACKTEST_CANDLES      = 750
MIN_TRADES_REQUIRED   = 1

ATR_PERIOD            = 14
ATR_SL_MULT           = 1.6
ATR_TP_MULT           = 2.8

STOCH_FASTK           = 14
STOCH_SLOWK           = 3
STOCH_SLOWD           = 3

MACD_FAST             = 12
MACD_SLOW             = 26
MACD_SIGNAL           = 9
MACD_HIST_BIAS_THRESH = 0.0

ADX_PERIOD            = 14
ADX_TREND_THRESHOLD   = 20

VWAP_SOURCE           = "hlc3"

# ─── Timeframe settings ───────────────────────────────
TF_PRIMARY   = "5m"
TF_TREND_1   = "15m"
TF_TREND_2   = "1h"

INTERVAL_MAP = {
    "1m":  Client.KLINE_INTERVAL_1MINUTE,
    "5m":  Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "1h":  Client.KLINE_INTERVAL_1HOUR,
    "4h":  Client.KLINE_INTERVAL_4HOUR,
    "1d":  Client.KLINE_INTERVAL_1DAY,
}

# ────────────────────────────────────────────────
#   HELPERS
# ────────────────────────────────────────────────

def hlc3(high, low, close) -> pd.Series:
    return (high + low + close) / 3

def get_vwap(df: pd.DataFrame, source: str = "close") -> pd.Series:
    if source == "hlc3":
        typical_price = hlc3(df["High"], df["Low"], df["Close"])
    else:
        typical_price = df["Close"]
    cum_volume = df["Volume"].cumsum()
    cum_price_volume = (typical_price * df["Volume"]).cumsum()
    vwap = cum_price_volume / cum_volume.replace(0, np.nan)
    return vwap.ffill()

def fetch_klines(symbol: str, interval: str, limit: int = 500) -> Optional[pd.DataFrame]:
    if interval not in INTERVAL_MAP:
        logging.error(f"Unsupported interval: {interval} for {symbol}")
        return None

    for attempt in range(4):
        try:
            klines = client.get_klines(symbol=symbol, interval=INTERVAL_MAP[interval], limit=limit)
            if not klines:
                return None

            df = pd.DataFrame(klines, columns=[
                "Open time", "Open", "High", "Low", "Close", "Volume",
                "Close time", "QuoteVolume", "Trades",
                "TakerBuyBase", "TakerBuyQuote", "Ignore"
            ])

            numeric_cols = ["Open", "High", "Low", "Close", "Volume", "QuoteVolume"]
            df[numeric_cols] = df[numeric_cols].astype(float)

            df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
            df.set_index("Open time", inplace=True)
            df = df[["Open", "High", "Low", "Close", "Volume", "QuoteVolume"]]

            return df

        except BinanceAPIException as e:
            logging.warning(f"{symbol} {interval} fetch error (attempt {attempt+1}): {e.status_code} - {e.message}")
            if attempt < 3:
                time.sleep(1.2 ** attempt + 0.3)
        except Exception as e:
            logging.error(f"Unexpected error fetching {symbol} {interval}: {e}")
            time.sleep(2)

    logging.error(f"Failed to fetch {symbol} {interval} after retries")
    return None

def detect_candlestick_patterns(df: pd.DataFrame, idx: int = -1) -> List[str]:
    if idx < 0:
        idx = len(df) + idx
    if idx < 1 or idx >= len(df):
        return []

    o, c, h, l = df["Open"].iloc[idx], df["Close"].iloc[idx], df["High"].iloc[idx], df["Low"].iloc[idx]
    prev_o, prev_c = df["Open"].iloc[idx-1], df["Close"].iloc[idx-1]

    body = abs(c - o)
    candle_range = h - l

    patterns = []
    if body <= candle_range * 0.13:
        patterns.append("Doji-like")
    if c > o and prev_c < prev_o and c >= prev_o and o <= prev_c:
        patterns.append("Bullish Engulfing")
    if c < o and prev_c > prev_o and c <= prev_o and o >= prev_c:
        patterns.append("Bearish Engulfing")

    return patterns

def get_multi_timeframe_trend(symbol: str) -> Optional[Literal["Bull", "Bear", "Neutral"]]:
    df15 = fetch_klines(symbol, "15m", 60)
    df1h = fetch_klines(symbol, "1h", 60)
    if df15 is None or df1h is None:
        return None

    vwap15 = get_vwap(df15, VWAP_SOURCE)
    vwap1h = get_vwap(df1h, VWAP_SOURCE)

    price15 = df15["Close"].iloc[-1]
    price1h = df1h["Close"].iloc[-1]

    tf15 = "Bull" if price15 > vwap15.iloc[-1] else "Bear"
    tf1h = "Bull" if price1h > vwap1h.iloc[-1] else "Bear"

    return tf15 if tf15 == tf1h else "Neutral"

# ─── Regime Classification ──────────────────────────────────────────────────

def prepare_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Add indicators needed for regime classification"""
    df = df.copy()

    # EMA for macro context
    if timeframe in ['1d', '4h']:
        df['EMA_50']  = ta.EMA(df['Close'], timeperiod=50)
        df['EMA_200'] = ta.EMA(df['Close'], timeperiod=200)

    # ADX + DI
    if timeframe in ['1d', '4h', '1h', '15m']:
        df['ADX']   = ta.ADX(df['High'], df['Low'], df['Close'], 14)
        df['+DI']   = ta.PLUS_DI(df['High'], df['Low'], df['Close'], 14)
        df['-DI']   = ta.MINUS_DI(df['High'], df['Low'], df['Close'], 14)

    # RSI
    if timeframe == '4h':
        df['RSI_14'] = ta.RSI(df['Close'], 14)

    # MACD histogram change
    if timeframe == '4h':
        macd, _, hist = ta.MACD(df['Close'], 12, 26, 9)
        df['MACD_hist'] = hist
        df['MACD_hist_diff_3'] = hist.diff().rolling(3).sum().shift(1)

    # Bollinger Band Width
    if timeframe in ['1h', '15m']:
        upper, mid, lower = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_upper'] = upper
        df['BB_mid']   = mid
        df['BB_lower'] = lower
        df['BB_width'] = (upper - lower) / mid
        if timeframe == '1h':
            df['BB_width_sma20'] = df['BB_width'].rolling(20).mean()
        if timeframe == '15m':
            df['BB_width_sma40'] = df['BB_width'].rolling(40).mean()

    # ATR %
    if timeframe == '1h':
        df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], 14)

    # Stochastic for scalping
    if timeframe == '5m':
        slowk, _ = ta.STOCH(df['High'], df['Low'], df['Close'], 14, 3, 3)
        df['slowk'] = slowk

    return df

# ────────────────────────────────────────────────
#   BACKTEST
# ────────────────────────────────────────────────

def prepare_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Add indicators needed for regime classification"""
    df = df.copy()

    # EMA for macro context
    if timeframe in ['1d', '4h']:
        df['EMA_50']  = ta.EMA(df['Close'], timeperiod=50)
        df['EMA_200'] = ta.EMA(df['Close'], timeperiod=200)

    # ADX + DI
    if timeframe in ['1d', '4h', '1h', '15m']:
        df['ADX']   = ta.ADX(df['High'], df['Low'], df['Close'], 14)
        df['+DI']   = ta.PLUS_DI(df['High'], df['Low'], df['Close'], 14)
        df['-DI']   = ta.MINUS_DI(df['High'], df['Low'], df['Close'], 14)

    # RSI
    if timeframe == '4h':
        df['RSI_14'] = ta.RSI(df['Close'], 14)

    # MACD histogram change
    if timeframe == '4h':
        macd, _, hist = ta.MACD(df['Close'], 12, 26, 9)
        df['MACD_hist'] = hist
        df['MACD_hist_diff_3'] = hist.diff().rolling(3).sum().shift(1)

    # Bollinger Band Width
    if timeframe in ['1h', '15m']:
        upper, mid, lower = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_upper'] = upper
        df['BB_mid']   = mid
        df['BB_lower'] = lower
        df['BB_width'] = (upper - lower) / mid
        if timeframe == '1h':
            df['BB_width_sma20'] = df['BB_width'].rolling(20).mean()
        if timeframe == '15m':
            df['BB_width_sma40'] = df['BB_width'].rolling(40).mean()

    # ATR %
    if timeframe == '1h':
        df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], 14)

    # Stochastic for scalping
    if timeframe == '5m':
        slowk, _ = ta.STOCH(df['High'], df['Low'], df['Close'], 14, 3, 3)
        df['slowk'] = slowk

    return df


def classify_regime(tfs_data: Dict[str, pd.DataFrame], symbol: str = "") -> Tuple[str, Dict[str, float]]:
    scores = {'Trend': 0.0, 'Volatility': 0.0, 'Range': 0.0, 'Reversal': 0.0, 'Scalping': 0.0}

    def safe_get(df, col, idx=-1):
        if df is None or col not in df.columns or len(df) <= abs(idx):
            return np.nan
        v = df[col].iloc[idx]
        return v if not pd.isna(v) else np.nan

    # ─── Extract indicators ─────────────────────────────────────────────────
    # 1d
    df1d = tfs_data.get('1d')
    ema50_1d  = safe_get(df1d, 'EMA_50')
    ema200_1d = safe_get(df1d, 'EMA_200')
    adx_1d    = safe_get(df1d, 'ADX')
    close_1d  = safe_get(df1d, 'Close')

    # 4h
    df4h = tfs_data.get('4h')
    adx_4h           = safe_get(df4h, 'ADX')
    plus_di_4h       = safe_get(df4h, '+DI')
    minus_di_4h      = safe_get(df4h, '-DI')
    rsi_4h           = safe_get(df4h, 'RSI_14')
    macd_hist_diff_3 = safe_get(df4h, 'MACD_hist_diff_3')

    # 1h
    df1h = tfs_data.get('1h')
    atr_1h        = safe_get(df1h, 'ATR')
    close_1h      = safe_get(df1h, 'Close')
    bbw_1h        = safe_get(df1h, 'BB_width')
    bbw_1h_avg    = safe_get(df1h, 'BB_width_sma20')
    adx_1h        = safe_get(df1h, 'ADX')          # ← this was missing

    # 15m
    df15m = tfs_data.get('15m')
    adx_15m       = safe_get(df15m, 'ADX')
    bbw_15m       = safe_get(df15m, 'BB_width')
    bbw_15m_avg   = safe_get(df15m, 'BB_width_sma40')

    # 5m
    df5m = tfs_data.get('5m')
    stoch_k_5m = safe_get(df5m, 'slowk')

    # ─── Trend ──────────────────────────────────────────────────────────────
    if not np.isnan(adx_4h) and not np.isnan(adx_1h):
        if adx_4h > 24 and adx_1h > 21:
            scores['Trend'] += 2.8

    if not np.isnan(plus_di_4h) and not np.isnan(minus_di_4h):
        if plus_di_4h > minus_di_4h + 5:
            scores['Trend'] += 1.2

    if not np.isnan(close_1d) and not np.isnan(ema50_1d) and not np.isnan(ema200_1d):
        if close_1d > ema50_1d > ema200_1d:
            scores['Trend'] += 1.5

    # ─── Volatility ─────────────────────────────────────────────────────────
    if not np.isnan(atr_1h) and not np.isnan(close_1h):
        if atr_1h / close_1h > 0.013:
            scores['Volatility'] += 2.0

    if not np.isnan(bbw_1h) and not np.isnan(bbw_1h_avg):
        if bbw_1h > bbw_1h_avg * 1.45:
            scores['Volatility'] += 1.8

    # ─── Range ──────────────────────────────────────────────────────────────
    if not np.isnan(adx_15m) and not np.isnan(bbw_15m) and not np.isnan(bbw_15m_avg):
        if adx_15m < 19 and bbw_15m < bbw_15m_avg * 0.78:
            scores['Range'] += 2.6

    if not np.isnan(adx_1h) and not np.isnan(adx_4h):
        if adx_1h < 20 and adx_4h < 22:
            scores['Range'] += 1.4

    # ─── Reversal ───────────────────────────────────────────────────────────
    if not np.isnan(rsi_4h):
        if rsi_4h > 76 or rsi_4h < 24:
            scores['Reversal'] += 1.8

    if not np.isnan(macd_hist_diff_3) and not np.isnan(rsi_4h):
        if macd_hist_diff_3 < -0.0012 and rsi_4h > 74:
            scores['Reversal'] += 2.1
        if macd_hist_diff_3 > 0.0012 and rsi_4h < 26:
            scores['Reversal'] += 2.1

    # ─── Scalping fallback ──────────────────────────────────────────────────
    if max(scores.values()) < 1.8:
        scores['Scalping'] += 2.0

    if not np.isnan(stoch_k_5m):
        if 20 < stoch_k_5m < 80:
            scores['Scalping'] += 0.7

    # ─── Final dominant regime with veto ────────────────────────────────────
    dominant = max(scores, key=scores.get)

    if dominant == 'Trend' and not np.isnan(adx_1d) and adx_1d < 16:
        scores['Trend'] *= 0.4
        dominant = max(scores, key=scores.get)

    return dominant, {k: round(v, 2) for k, v in scores.items()}


def backtest_pair(symbol: str) -> Optional[dict]:
    """
    Backtest the strategy on 5m data with multi-TF trend + regime + bias filters.
    Returns performance statistics or None if invalid / insufficient trades.
    """
    try:
        logging.info(f"Backtesting {symbol} ...")

        df_5m = fetch_klines(symbol, "5m", BACKTEST_CANDLES)
        if df_5m is None or len(df_5m) < 100:
            logging.warning(f"Insufficient 5m data for {symbol}")
            return None

        df_15m = fetch_klines(symbol, "15m", 200)
        df_1h  = fetch_klines(symbol, "1h",   120)
        df_4h  = fetch_klines(symbol, "4h",   100)
        df_1d  = fetch_klines(symbol, "1d",    60)

        if df_15m is None or df_1h is None:
            logging.warning(f"Missing higher TF data for {symbol}")
            return None

        # Prepare regime indicators
        tfs = {
            '5m':  prepare_indicators(df_5m,  '5m'),
            '15m': prepare_indicators(df_15m, '15m'),
            '1h':  prepare_indicators(df_1h,  '1h'),
            '4h':  prepare_indicators(df_4h,  '4h'),
            '1d':  prepare_indicators(df_1d,  '1d'),
        }

        regime, regime_scores = classify_regime(tfs, symbol)
        logging.info(f"{symbol} regime detected: {regime} | {regime_scores}")

        # Optional: skip backtest in certain regimes
        # if regime in ['Range', 'Scalping']:
        #     logging.info(f"{symbol} skipped - regime {regime}")
        #     return None

        # ─── Pre-calculate trading indicators ─────────────────────────────────
        df_5m["VWAP"] = get_vwap(df_5m, VWAP_SOURCE)
        df_5m["ATR"]  = ta.ATR(df_5m.High, df_5m.Low, df_5m.Close, timeperiod=ATR_PERIOD)

        slowk, slowd = ta.STOCH(
            df_5m.High, df_5m.Low, df_5m.Close,
            fastk_period=STOCH_FASTK,
            slowk_period=STOCH_SLOWK,
            slowd_period=STOCH_SLOWD
        )
        df_5m["slowk"] = slowk
        df_5m["slowd"] = slowd

        df_15m["VWAP"] = get_vwap(df_15m, VWAP_SOURCE)
        df_1h["VWAP"]  = get_vwap(df_1h,  VWAP_SOURCE)

        macd_line, macd_signal, macd_hist_series = ta.MACD(
            df_5m["Close"],
            fastperiod=MACD_FAST,
            slowperiod=MACD_SLOW,
            signalperiod=MACD_SIGNAL
        )
        df_5m["MACD"]        = macd_line
        df_5m["MACD_signal"] = macd_signal
        df_5m["MACD_hist"]   = macd_hist_series

        df_1h["ADX"] = ta.ADX(
            df_1h["High"], df_1h["Low"], df_1h["Close"],
            timeperiod=ADX_PERIOD
        )

        if df_5m["MACD_hist"].isna().all() or df_1h["ADX"].isna().all():
            logging.warning(f"{symbol} - MACD or ADX computation failed (all NaN)")
            return None

        trades = []
        equity_curve = [1.0]

        start_idx = max(ATR_PERIOD, STOCH_FASTK + STOCH_SLOWK, 50) + 10

        for i in range(start_idx, len(df_5m) - 1):
            price     = df_5m["Close"].iloc[i]
            vwap_5m   = df_5m["VWAP"].iloc[i]
            atr       = df_5m["ATR"].iloc[i]
            k         = df_5m["slowk"].iloc[i]
            d         = df_5m["slowd"].iloc[i]
            macd_hist = df_5m["MACD_hist"].iloc[i]

            if pd.isna([atr, k, d, macd_hist]).any():
                continue

            t_5m = df_5m.index[i]

            idx_15m_arr = df_15m.index.get_indexer([t_5m], method="ffill")
            idx_15m = idx_15m_arr[0]
            if idx_15m == -1:
                continue

            idx_1h_arr = df_1h.index.get_indexer([t_5m], method="ffill")
            idx_1h = idx_1h_arr[0]
            if idx_1h == -1:
                continue

            trend_15m = df_15m["Close"].iloc[idx_15m] > df_15m["VWAP"].iloc[idx_15m]
            trend_1h  = df_1h["Close"].iloc[idx_1h]  > df_1h["VWAP"].iloc[idx_1h]

            is_bull_trend = trend_15m and trend_1h
            is_bear_trend = not trend_15m and not trend_1h

            adx_value = df_1h["ADX"].iloc[idx_1h]
            if pd.isna(adx_value) or adx_value < ADX_TREND_THRESHOLD:
                continue

            patterns = detect_candlestick_patterns(df_5m, i)

            signal = None

            if ("Bullish Engulfing" in patterns or "Doji-like" in patterns) and price > vwap_5m:
                if "Bullish Engulfing" in patterns and is_bull_trend:
                    signal = "BUY"
                elif "Doji-like" in patterns and k < 25 and d < 25 and is_bull_trend:
                    signal = "BUY"

            if ("Bearish Engulfing" in patterns or "Doji-like" in patterns) and price < vwap_5m:
                if "Bearish Engulfing" in patterns and is_bear_trend:
                    signal = "SELL"
                elif "Doji-like" in patterns and k > 75 and d > 75 and is_bear_trend:
                    signal = "SELL"

            if not signal:
                continue

            if signal == "BUY" and macd_hist <= MACD_HIST_BIAS_THRESH:
                continue
            if signal == "SELL" and macd_hist >= MACD_HIST_BIAS_THRESH:
                continue

            if signal == "BUY":
                entry       = price * (1 + SLIPPAGE)
                stop_loss   = entry - atr * ATR_SL_MULT
                take_profit = entry + atr * ATR_TP_MULT
                direction_mult = 1.0
            else:
                entry       = price * (1 - SLIPPAGE)
                stop_loss   = entry + atr * ATR_SL_MULT
                take_profit = entry - atr * ATR_TP_MULT
                direction_mult = -1.0

            exit_price = None
            exit_type  = "timeout"

            for j in range(i + 1, min(i + MAX_HOLDING_CANDLES + 1, len(df_5m))):
                high_j = df_5m["High"].iloc[j]
                low_j  = df_5m["Low"].iloc[j]

                if signal == "BUY":
                    if low_j <= stop_loss:
                        exit_price = stop_loss
                        exit_type = "SL"
                        break
                    if high_j >= take_profit:
                        exit_price = take_profit
                        exit_type = "TP"
                        break
                else:
                    if high_j >= stop_loss:
                        exit_price = stop_loss
                        exit_type = "SL"
                        break
                    if low_j <= take_profit:
                        exit_price = take_profit
                        exit_type = "TP"
                        break

            if exit_price is None:
                exit_price = df_5m["Close"].iloc[j]
                exit_type = "timeout"

            gross_profit_pct = direction_mult * (exit_price - entry) / entry
            net_profit_pct   = gross_profit_pct - 2 * TRANSACTION_FEE

            trades.append({
                "entry_time": df_5m.index[i],
                "exit_time":  df_5m.index[j],
                "direction":  signal,
                "entry":      entry,
                "exit":       exit_price,
                "exit_type":  exit_type,
                "net_return": net_profit_pct
            })

            equity_curve.append(equity_curve[-1] * (1 + net_profit_pct))

        if len(trades) < MIN_TRADES_REQUIRED:
            logging.info(f"{symbol} → only {len(trades)} trades → skipped")
            return None

        returns = np.array([t["net_return"] for t in trades])
        wins = returns > 0
        win_rate = np.mean(wins) if len(returns) > 0 else 0.0

        profit = returns[returns > 0].sum()
        loss   = abs(returns[returns < 0].sum())
        profit_factor = profit / loss if loss > 0 else 10.0
        profit_factor = min(profit_factor, 10.0)

        avg_return = returns.mean()
        expectancy = avg_return

        if len(returns) > 5 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(105120)
        else:
            sharpe = 0.0

        equity = np.array(equity_curve)
        drawdowns = 1 - equity / np.maximum.accumulate(equity)
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0

        score_components = {
            "win_rate":      min(win_rate, 0.90) * 1.0,
            "profit_factor": min(profit_factor / 3.0, 1.0) * 1.0,
            "expectancy":    min(avg_return / 0.005, 1.0) * 0.8,
            "sharpe":        min(sharpe / 2.0, 1.0) * 0.8,
            "drawdown":      max(1 - max_dd / 0.20, 0.0) * 1.2
        }

        total_score = sum(score_components.values()) / len(score_components) * 100

        result = {
            "symbol": symbol,
            "trades_count": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "avg_return_per_trade": avg_return,
            "score": round(total_score, 2),
            "score_components": score_components,
            "regime": regime,
            "regime_scores": regime_scores
        }

        logging.info(
            f"{symbol} | Regime: {regime} | Trades: {len(trades)} | WR: {win_rate:.1%} | "
            f"PF: {profit_factor:.2f} | Exp: {expectancy:.4f} | Sharpe: {sharpe:.2f} | "
            f"DD: {max_dd:.1%} | Score: {result['score']:.1f}"
        )

        return result

    except Exception as e:
        logging.error(f"Backtest failed for {symbol}: {e}", exc_info=True)
        return None


# ────────────────────────────────────────────────
#   REAL-TIME SIGNAL LOGIC
# ────────────────────────────────────────────────

async def analyze_pair(symbol: str) -> Tuple[Optional[str], float]:
    df5m = fetch_klines(symbol, TF_PRIMARY, 120)
    if df5m is None or len(df5m) < 60:
        return None, 0.0

    last_candle_end = df5m.index[-1].timestamp() + 300
    if time.time() < last_candle_end:
        return None, 0.0

    # Fetch all timeframes for regime classification
    tfs = {}
    for tf, lim in [('5m', 120), ('15m', 80), ('1h', 80), ('4h', 60), ('1d', 40)]:
        df = fetch_klines(symbol, tf, lim)
        if df is not None and len(df) >= 30:
            tfs[tf] = prepare_indicators(df, tf)

    regime, regime_scores = classify_regime(tfs, symbol)

    df5m["VWAP"] = get_vwap(df5m, VWAP_SOURCE)
    df5m["ATR"]  = ta.ATR(df5m.High, df5m.Low, df5m.Close, ATR_PERIOD)

    stoch_k, stoch_d = ta.STOCH(
        df5m.High, df5m.Low, df5m.Close,
        fastk_period=STOCH_FASTK,
        slowk_period=STOCH_SLOWK,
        slowd_period=STOCH_SLOWD
    )
    latest_k = stoch_k[-1]
    latest_d = stoch_d[-1]

    macd, macd_signal, macd_hist = ta.MACD(
        df5m["Close"],
        fastperiod=MACD_FAST,
        slowperiod=MACD_SLOW,
        signalperiod=MACD_SIGNAL
    )
    latest_macd_hist = macd_hist[-1] if len(macd_hist) > 0 else 0.0

    patterns = detect_candlestick_patterns(df5m)

    price = df5m["Close"].iloc[-1]
    vwap  = df5m["VWAP"].iloc[-1]
    atr   = df5m["ATR"].iloc[-1]

    trend = get_multi_timeframe_trend(symbol)
    if trend is None:
        return None, 0.0

    df1h = tfs.get('1h')
    latest_adx = safe_get(df1h, 'ADX') if df1h is not None else 0.0

    if latest_adx < ADX_TREND_THRESHOLD:
        return None, 0.0

    signal = None
    direction = None

    if "Bullish Engulfing" in patterns and price > vwap and trend == "Bull":
        signal = "Buy - Bullish Engulfing"
        direction = "long"
    elif "Bearish Engulfing" in patterns and price < vwap and trend == "Bear":
        signal = "Sell - Bearish Engulfing"
        direction = "short"
    elif "Doji-like" in patterns:
        if latest_k < 22 and latest_d < 22 and price > vwap and trend == "Bull":
            signal = "Buy - Doji in oversold"
            direction = "long"
        elif latest_k > 78 and latest_d > 78 and price < vwap and trend == "Bear":
            signal = "Sell - Doji in overbought"
            direction = "short"

    if not signal:
        return None, 0.0

    if direction == "long" and latest_macd_hist <= MACD_HIST_BIAS_THRESH:
        return None, 0.0

    if direction == "short" and latest_macd_hist >= MACD_HIST_BIAS_THRESH:
        return None, 0.0

    if direction == "long":
        entry = price * (1 + SLIPPAGE)
        sl    = entry - atr * ATR_SL_MULT
        tp    = entry + atr * ATR_TP_MULT
    else:
        entry = price * (1 - SLIPPAGE)
        sl    = entry + atr * ATR_SL_MULT
        tp    = entry - atr * ATR_TP_MULT

    k_diff     = abs(latest_k - latest_d) / 100
    atr_norm   = min(atr / price * 300, 1.0)
    vol_rel    = min(df5m["Volume"].iloc[-1] / df5m["Volume"].mean(), 6) / 6
    trend_mult = 1.0 if (direction == "long" and trend == "Bull") or \
                        (direction == "short" and trend == "Bear") else 0.4

    adx_factor      = min(latest_adx / 40.0, 1.5)
    macd_hist_norm  = min(abs(latest_macd_hist) / (price * 0.0008), 1.2)

    base_strength = (
        0.25 * k_diff +
        0.20 * atr_norm +
        0.15 * vol_rel +
        0.20 * trend_mult +
        0.10 * adx_factor +
        0.10 * macd_hist_norm
    ) * 100

    strength = min(round(base_strength, 1), 100)

    msg = (
        f"**{symbol}** ─ {signal}\n"
        f"Entry ≈ {entry:,.4f}\n"
        f"SL     {sl:,.4f}\n"
        f"TP     {tp:,.4f}\n"
        f"Strength: **{strength:.1f}/100**\n"
        f"Trend: {trend} (15m+1h)\n"
        f"Regime: **{regime}**\n"
        f"ADX: {latest_adx:.1f} | MACD Hist: {latest_macd_hist:.6f}\n"
        f"Patterns: {', '.join(patterns)}"
    )

    await bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")

    return direction, strength


# ────────────────────────────────────────────────
#   PAIR SELECTION (very simplified version here)
# ────────────────────────────────────────────────

async def fetch_top_pairs() -> List[str]:
    try:
        logging.info("Starting top pairs selection...")

        # ─── Step 1: Get recent high-volume USDT pairs ───────────────────────
        tickers = client.get_ticker()
        usdt_pairs = [
            t for t in tickers
            if t["symbol"].endswith("USDT")
            and float(t["quoteVolume"]) > MIN_QUOTE_VOLUME
            and t["symbol"] not in ["USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT"]
        ]

        usdt_pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        candidates = [t["symbol"] for t in usdt_pairs[:35]]

        logging.info(f"Initial candidates ({len(candidates)}): {', '.join(candidates[:10])} ...")

        # ─── Step 2: Quick pre-filter ─────────────────────────────────────────
        prefiltered = []
        for symbol in candidates:
            df5 = fetch_klines(symbol, "5m", 80)
            if df5 is None or len(df5) < 50:
                continue

            df5["ATR"] = ta.ATR(df5["High"], df5["Low"], df5["Close"], timeperiod=14)
            if pd.isna(df5["ATR"].iloc[-1]):
                continue

            atr_pct = df5["ATR"].iloc[-1] / df5["Close"].iloc[-1]
            vol_mean = df5["Volume"].mean()

            if vol_mean < 8_000 or atr_pct < 0.0004:
                continue

            prefiltered.append(symbol)

        logging.info(f"Prefiltered down to {len(prefiltered)} pairs")

        if not prefiltered:
            logging.warning("No pairs passed pre-filter → using defaults")
            await send_telegram_message("No pairs passed pre-filter. Using defaults.")
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

        # ─── Step 3: Backtest all candidates ──────────────────────────────────
        backtest_results = []
        for symbol in prefiltered:
            result = backtest_pair(symbol)
            if result is not None:
                backtest_results.append(result)

        if not backtest_results:
            logging.warning("No valid backtest results → using defaults")
            await send_telegram_message("Backtesting failed for all. Using defaults.")
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

        # ─── Step 4: Group by regime & select top 4 per regime ────────────────
        from collections import defaultdict

        regime_groups = defaultdict(list)
        for res in backtest_results:
            reg = res.get("regime", "Unknown")
            regime_groups[reg].append(res)

        # Sort each group by score descending
        for reg in regime_groups:
            regime_groups[reg].sort(key=lambda x: x["score"], reverse=True)

        # ─── Log grouped top 4 per regime ─────────────────────────────────────
        logging.info("═" * 60)
        logging.info("BACKTEST RESULTS GROUPED BY REGIME (top 4 per group)")
        logging.info("═" * 60)

        telegram_lines = ["**Backtest Results by Regime (Top 4 per group)**\n"]

        for regime, group in sorted(regime_groups.items()):
            if not group:
                continue

            logging.info(f"Regime: {regime} ({len(group)} pairs)")
            telegram_lines.append(f"**{regime}** ({len(group)} pairs)")

            for rank, res in enumerate(group[:4], 1):
                log_line = (
                    f"  #{rank} {res['symbol']:10} | Score: {res['score']:6.1f} | "
                    f"Trades: {res['trades_count']:3} | WR: {res['win_rate']:5.1%} | "
                    f"PF: {res['profit_factor']:5.2f} | DD: {res['max_drawdown']:5.1%}"
                )
                logging.info(log_line)
                telegram_lines.append(f"{rank}. **{res['symbol']}** (score: {res['score']:.1f})")

            logging.info("")  # empty line between regimes

        # ─── Overall top N for watchlist ──────────────────────────────────────
        backtest_results.sort(key=lambda x: x["score"], reverse=True)
        top_pairs = [r["symbol"] for r in backtest_results[:TOP_PAIRS_COUNT]]

        # ─── Final summary to Telegram ────────────────────────────────────────
        telegram_lines.append("\n**Final Watchlist (Top Overall)**")
        for i, sym in enumerate(top_pairs, 1):
            telegram_lines.append(f"{i}. **{sym}**")

        await send_telegram_message("\n".join(telegram_lines))

        logging.info(f"Selected watchlist ({len(top_pairs)}): {', '.join(top_pairs)}")

        return top_pairs

    except Exception as e:
        logging.exception(f"Error in fetch_top_pairs: {e}")
        await send_telegram_message(f"Error in pair selection: {str(e)}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

async def send_telegram_message(text: str):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        logging.info(f"Telegram sent: {text[:100]}...")
    except Exception as e:
        logging.error(f"Telegram failed: {e}")

# ────────────────────────────────────────────────
#   MAIN LOOP
# ────────────────────────────────────────────────

async def main_loop():
    symbols: List[str] = []
    last_pair_update = 0.0

    while True:
        now = time.time()

        # Update pair list periodically
        if now - last_pair_update >= PAIR_UPDATE_INTERVAL or not symbols:
            logging.info("Refreshing trading pairs...")
            symbols = await fetch_top_pairs()
            if symbols:
                logging.info(f"Active symbols: {', '.join(symbols)}")
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=f"Watchlist updated ({len(symbols)} pairs):\n{', '.join(symbols)}"
                )
            last_pair_update = now

        # Wait for next clean 5m candle (with small buffer)
        seconds_into_candle = now % 300
        wait_time = 300 - seconds_into_candle + 4  # +4s buffer
        if wait_time < 2:
            wait_time += 300

        logging.debug(f"Sleeping {wait_time:.0f}s until next 5m candle analysis")
        await asyncio.sleep(wait_time)

        # Analyze all symbols
        for symbol in symbols:
            try:
                direction, strength = await analyze_pair(symbol)
                if direction and strength > 30:  # example threshold
                    logging.info(f"Signal → {symbol} | {direction} | strength: {strength:.1f}")
            except Exception as e:
                logging.error(f"Analysis failed for {symbol}: {e}")

        # Optional: every X cycles, log summary or clean up old signals

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logging.info("Stopped by user")
    except Exception as e:
        logging.critical(f"Fatal error: {e}", exc_info=True)
#INDEXBASED + INSTITUTIONAL EDITION - PERFECT ENTRIES ONLY

import os
import time
import requests
import pandas as pd
import yfinance as yf
import ta
import warnings
import pyotp
import math
from datetime import datetime, time as dtime, timedelta
from SmartApi.smartConnect import SmartConnect
import threading
import numpy as np

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

# STRONGER CONFIRMATION THRESHOLDS
VCP_CONTRACTION_RATIO = 0.6
FAULTY_BASE_BREAK_THRESHOLD = 0.25
WYCKOFF_VOLUME_SPRING = 2.2
LIQUIDITY_SWEEP_DISTANCE = 0.005
PEAK_REJECTION_WICK_RATIO = 0.8
FVG_GAP_THRESHOLD = 0.0025
VOLUME_GAP_IMBALANCE = 2.5

# --------- EXPIRIES FOR KEPT INDICES ---------
EXPIRIES = {
    "NIFTY": "25 NOV 2025",
    "BANKNIFTY": "25 NOV 2025", 
    "SENSEX": "20 NOV 2025"
}

# --------- STRATEGY TRACKING ---------
STRATEGY_NAMES = {
    "institutional_liquidity_grab": "INSTITUTIONAL LIQUIDITY GRAB",
    "institutional_absorption": "INSTITUTIONAL ABSORPTION", 
    "institutional_momentum_ignition": "INSTITUTIONAL MOMENTUM",
    "liquidity_sweeps": "LIQUIDITY SWEEP",
    "liquidity_zone": "LIQUIDITY ZONE",
    "opening_play": "OPENING PLAY"
}

# --------- ENHANCED TRACKING FOR REPORTS ---------
all_generated_signals = []
strategy_performance = {}
signal_counter = 0
daily_signals = []

# --------- SIGNAL DEDUPLICATION AND COOLDOWN TRACKING ---------
active_strikes = {}
last_signal_time = {}
signal_cooldown = 1200

def initialize_strategy_tracking():
    global strategy_performance
    strategy_performance = {
        "INSTITUTIONAL LIQUIDITY GRAB": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "INSTITUTIONAL ABSORPTION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "INSTITUTIONAL MOMENTUM": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "LIQUIDITY SWEEP": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "LIQUIDITY ZONE": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "OPENING PLAY": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0}
    }

initialize_strategy_tracking()

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
TOTP = pyotp.TOTP(TOTP_SECRET).now()

client = SmartConnect(api_key=API_KEY)
session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
feedToken = client.getfeedToken()

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

def send_telegram(msg, reply_to=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=5).json()
        return r.get("result", {}).get("message_id")
    except:
        return None

# --------- MARKET HOURS ---------
def is_market_open():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return dtime(9,15) <= current_time_ist <= dtime(15,30)

def should_stop_trading():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return current_time_ist >= dtime(15,30)

# --------- ONE STEP AWAY STRIKE ROUNDING ---------
def round_strike_one_step_away(index, price):
    try:
        if price is None:
            return None
        if isinstance(price, float) and math.isnan(price):
            return None
        price = float(price)
        
        if index == "NIFTY": 
            nearest = int(round(price / 50.0) * 50)
            # One step away (50 points away from nearest)
            if price > nearest:
                return nearest + 50
            else:
                return nearest - 50
        elif index == "BANKNIFTY": 
            nearest = int(round(price / 100.0) * 100)
            # One step away (100 points away from nearest)
            if price > nearest:
                return nearest + 100
            else:
                return nearest - 100
        elif index == "SENSEX": 
            nearest = int(round(price / 100.0) * 100)
            # One step away (100 points away from nearest)
            if price > nearest:
                return nearest + 100
            else:
                return nearest - 100
        else: 
            nearest = int(round(price / 50.0) * 50)
            if price > nearest:
                return nearest + 50
            else:
                return nearest - 50
    except Exception:
        return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- FIXED: FETCH INDEX DATA WITH VOLUME ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN"
    }
    try:
        # Use yf.download with actions=False to get Volume
        df = yf.download(
            symbol_map[index], 
            period=period, 
            interval=interval, 
            auto_adjust=True, 
            progress=False,
            actions=False  # This ensures Volume data
        )
        
        if df.empty:
            print(f"❌ Empty data for {index}")
            return None
            
        # Ensure Volume column exists
        if 'Volume' not in df.columns:
            print(f"❌ No Volume column for {index}")
            return None
            
        # Check if volume data is valid
        if df['Volume'].isna().all() or (df['Volume'] == 0).all():
            print(f"❌ Invalid volume data for {index}")
            return None
            
        return df
        
    except Exception as e:
        print(f"❌ Error fetching {index}: {e}")
        return None

# --------- LOAD TOKEN MAP ---------
def load_token_map():
    try:
        url="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df.columns=[c.lower() for c in df.columns]
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        return df.set_index('symbol')['token'].to_dict()
    except:
        return {}

token_map=load_token_map()

# --------- SAFE LTP FETCH ---------
def fetch_option_price(symbol, retries=3, delay=3):
    token=token_map.get(symbol.upper())
    if not token:
        return None
    for _ in range(retries):
        try:
            exchange = "BFO" if "SENSEX" in symbol.upper() else "NFO"
            data=client.ltpData(exchange, symbol, token)
            return float(data['data']['ltp'])
        except:
            time.sleep(delay)
    return None

# --------- STRICT EXPIRY VALIDATION ---------
def validate_option_symbol(index, symbol, strike, opttype):
    try:
        expected_expiry = EXPIRIES.get(index)
        if not expected_expiry:
            return False
            
        expected_dt = datetime.strptime(expected_expiry, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = expected_dt.strftime("%y")
            month_code = expected_dt.strftime("%b").upper()
            day = expected_dt.strftime("%d")
            expected_pattern = f"SENSEX{day}{month_code}{year_short}"
            symbol_upper = symbol.upper()
            
            if expected_pattern in symbol_upper:
                return True
            else:
                print(f"❌ SENSEX expiry mismatch: Expected {expected_pattern}, Got {symbol_upper}")
                return False
        else:
            expected_pattern = expected_dt.strftime("%d%b%y").upper()
            symbol_upper = symbol.upper()
            
            if expected_pattern in symbol_upper:
                return True
            else:
                print(f"❌ {index} expiry mismatch: Expected {expected_pattern}, Got {symbol_upper}")
                return False
                
    except Exception as e:
        print(f"Symbol validation error: {e}")
        return False

# --------- GET OPTION SYMBOL WITH STRICT EXPIRY VALIDATION ---------
def get_option_symbol(index, expiry_str, strike, opttype):
    try:
        dt = datetime.strptime(expiry_str, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = dt.strftime("%y")
            month_code = dt.strftime("%b").upper()
            day = dt.strftime("%d")
            symbol = f"SENSEX{day}{month_code}{year_short}{strike}{opttype}"
        else:
            symbol = f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        
        if validate_option_symbol(index, symbol, strike, opttype):
            print(f"✅ Valid symbol generated: {symbol}")
            return symbol
        else:
            print(f"❌ Generated symbol validation FAILED: {symbol}")
            return None
            
    except Exception as e:
        print(f"Error generating symbol: {e}")
        return None

# 🚨 CRITICAL INSTITUTIONAL FILTERS WE DISCUSSED

# --------- INSTITUTIONAL VOLUME CHECK ---------
def is_institutional_volume(df):
    """Check if volume shows institutional activity"""
    try:
        volume = ensure_series(df['Volume'])
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1]
        
        # Institutional moves have 2.5x+ average volume
        return current_volume > avg_volume * 2.5
    except:
        return False

# --------- INSTITUTIONAL VS RETAIL VOLUME ANALYSIS ---------
def is_institutional_buying(index, df, signal):
    """
    DISTINGUISH: Institutional buying vs Retail buying
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1]
        
        # 🚨 FILTER 1: PRICE LOCATION MATTERS
        resistance = high.rolling(20).max().iloc[-2]
        support = low.rolling(20).min().iloc[-2]
        
        if signal == "CE":
            # 🚨 INSTITUTIONAL BUYING: At support zones, not at highs
            if current_close > resistance - 10:  # Buying near resistance
                return False  # Likely retail FOMO
            if current_close < support + 20:     # Buying at support
                return True   # Likely institutional
            
        else:  # PE
            # 🚨 INSTITUTIONAL SELLING: At resistance zones, not at lows  
            if current_close < support + 10:     # Selling near support
                return False  # Likely retail panic
            if current_close > resistance - 20:  # Selling at resistance
                return True   # Likely institutional
        
        # 🚨 FILTER 2: CANDLE STRUCTURE MATTERS
        current_body = abs(current_close - prev_close)
        current_range = high.iloc[-1] - low.iloc[-1]
        body_ratio = current_body / current_range
        
        # Institutional moves have STRONG bodies (minimal wicks)
        if body_ratio < 0.6:  # Weak candle, too much wick
            return False      # Likely retail indecision
            
        # 🚨 FILTER 3: FOLLOW-THROUGH PATTERNS
        if signal == "CE":
            # Institutional buying continues, doesn't immediately reverse
            if close.iloc[-1] < close.iloc[-2]:  # No follow-through
                return False
        else:
            if close.iloc[-1] > close.iloc[-2]:  # No follow-through  
                return False
                
        return True
        
    except Exception:
        return False

# --------- INSTITUTIONAL PRICE CONVICTION ---------
def has_price_conviction(df):
    """Check if price move has conviction"""
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        current_range = high.iloc[-1] - low.iloc[-1]
        avg_range = (high.rolling(10).max() - low.rolling(10).min()).iloc[-1]
        return current_range > avg_range * 0.8
    except:
        return False

# --------- INSTITUTIONAL FOLLOW-THROUGH ---------
def has_follow_through(df, signal):
    """Check if move has follow-through"""
    try:
        close = ensure_series(df['Close'])
        if signal == "CE":
            return close.iloc[-1] > close.iloc[-2]
        else:  # PE
            return close.iloc[-1] < close.iloc[-2]
    except:
        return False

# --------- VOLUME QUALITY ANALYSIS ---------
def analyze_volume_quality(df, signal):
    """
    BETTER: Analyze HOW volume is occurring
    """
    volume = ensure_series(df['Volume'])
    close = ensure_series(df['Close'])
    
    current_volume = volume.iloc[-1]
    prev_volume = volume.iloc[-2]
    avg_volume = volume.rolling(20).mean().iloc[-1]
    
    # 🚨 INSTITUTIONAL: Volume expands on continuation
    # RETAIL: Volume spikes then dies
    
    if signal == "CE":
        # Institutional buying: Volume expands as price rises
        volume_quality = (current_volume > prev_volume and 
                         close.iloc[-1] > close.iloc[-2])
    else:
        # Institutional selling: Volume expands as price falls  
        volume_quality = (current_volume > prev_volume and
                         close.iloc[-1] < close.iloc[-2])
    
    return volume_quality and (current_volume > avg_volume * 2.0)

# 🚨 INSTITUTIONAL STRATEGIES WITH CORRECTIONS

# --------- INSTITUTIONAL LIQUIDITY GRAB ---------
def institutional_liquidity_grab(index, df):
    """INSTITUTIONAL: Run stops then reverse"""
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        # Find where stops are clustered
        recent_high = high.iloc[-10:-2].max()
        recent_low = low.iloc[-10:-2].min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1]
        
        # 🚨 INSTITUTIONAL BEAR TRAP: Run bull stops then reverse
        if (current_high > recent_high + 25 and
            current_close < recent_high - 20 and
            current_volume > avg_volume * 2.5 and
            is_institutional_buying(index, df, "PE") and
            analyze_volume_quality(df, "PE")):
            return "PE"
        
        # 🚨 INSTITUTIONAL BULL TRAP: Run bear stops then reverse  
        if (current_low < recent_low - 25 and
            current_close > recent_low + 20 and
            current_volume > avg_volume * 2.5 and
            is_institutional_buying(index, df, "CE") and
            analyze_volume_quality(df, "CE")):
            return "CE"
            
    except Exception:
        return None
    return None

# --------- INSTITUTIONAL ABSORPTION ---------
def institutional_absorption(index, df):
    """INSTITUTIONAL: Big players absorbing at key levels"""
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        # Find key institutional levels
        resistance = high.rolling(20).max().iloc[-2]
        support = low.rolling(20).min().iloc[-2]
        
        current_close = close.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1]
        
        # 🚨 INSTITUTIONAL SELLING AT RESISTANCE
        if (abs(current_close - resistance) < 15 and
            current_volume > avg_volume * 2.5 and
            close.iloc[-1] < close.iloc[-2] and  # Rejection
            is_institutional_buying(index, df, "PE") and
            analyze_volume_quality(df, "PE")):
            return "PE"
        
        # 🚨 INSTITUTIONAL BUYING AT SUPPORT  
        if (abs(current_close - support) < 15 and
            current_volume > avg_volume * 2.5 and
            close.iloc[-1] > close.iloc[-2] and  # Bounce
            is_institutional_buying(index, df, "CE") and
            analyze_volume_quality(df, "CE")):
            return "CE"
            
    except Exception:
        return None
    return None

# --------- INSTITUTIONAL MOMENTUM IGNITION ---------
def institutional_momentum_ignition(index, df):
    """INSTITUTIONAL: When they decide to MOVE the market"""
    try:
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1]
        
        # 🚨 BULLISH MOMENTUM IGNITION
        bullish_ignition = (
            close.iloc[-1] > close.iloc[-2] + 15 and
            close.iloc[-2] > close.iloc[-3] + 10 and
            current_volume > avg_volume * 2.5 and
            volume.iloc[-2] > avg_volume * 2.0 and
            (high.iloc[-1] - low.iloc[-1]) > 50 and
            is_institutional_buying(index, df, "CE") and
            analyze_volume_quality(df, "CE")
        )
        
        # 🚨 BEARISH MOMENTUM IGNITION
        bearish_ignition = (
            close.iloc[-1] < close.iloc[-2] - 15 and
            close.iloc[-2] < close.iloc[-3] - 10 and
            current_volume > avg_volume * 2.5 and
            volume.iloc[-2] > avg_volume * 2.0 and
            (high.iloc[-1] - low.iloc[-1]) > 50 and
            is_institutional_buying(index, df, "PE") and
            analyze_volume_quality(df, "PE")
        )
        
        if bullish_ignition:
            return "CE"
        elif bearish_ignition:
            return "PE"
            
    except Exception:
        return None
    return None

# --------- LIQUIDITY SWEEPS ---------
def detect_liquidity_sweeps(df):
    """Detect institutional liquidity sweeps"""
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        recent_highs = high.iloc[-10:-2]
        recent_lows = low.iloc[-10:-2]
        
        liquidity_high = recent_highs.max()
        liquidity_low = recent_lows.min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1]
        
        # 🚨 BEARISH LIQUIDITY SWEEP
        if (current_high > liquidity_high * (1 + LIQUIDITY_SWEEP_DISTANCE) and
            current_close < liquidity_high * 0.998 and
            current_volume > avg_volume * 2.5 and
            has_follow_through(df, "PE") and
            has_price_conviction(df)):
            return "PE"
            
        # 🚨 BULLISH LIQUIDITY SWEEP
        if (current_low < liquidity_low * (1 - LIQUIDITY_SWEEP_DISTANCE) and
            current_close > liquidity_low * 1.002 and
            current_volume > avg_volume * 2.5 and
            has_follow_through(df, "CE") and
            has_price_conviction(df)):
            return "CE"
    except Exception:
        return None
    return None

# --------- LIQUIDITY ZONE ---------
def detect_liquidity_zone(df, lookback=20):
    """Detect liquidity zones for institutional entries"""
    try:
        high_series = ensure_series(df['High']).dropna()
        low_series = ensure_series(df['Low']).dropna()
        
        if len(high_series) <= lookback:
            high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
        else:
            high_pool = float(high_series.rolling(lookback).max().iloc[-2])
            
        if len(low_series) <= lookback:
            low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')
        else:
            low_pool = float(low_series.rolling(lookback).min().iloc[-2])

        if math.isnan(high_pool) and len(high_series)>0:
            high_pool = float(high_series.max())
        if math.isnan(low_pool) and len(low_series)>0:
            low_pool = float(low_series.min())

        return round(high_pool,0), round(low_pool,0)
    except Exception:
        return None, None

def institutional_liquidity_hunt(index, df):
    """Institutional liquidity hunting at key zones"""
    try:
        prev_high = None
        prev_low = None
        try:
            prev_high_val = ensure_series(df['High']).iloc[-2]
            prev_low_val = ensure_series(df['Low']).iloc[-2]
            prev_high = float(prev_high_val) if not (isinstance(prev_high_val,float) and math.isnan(prev_high_val)) else None
            prev_low = float(prev_low_val) if not (isinstance(prev_low_val,float) and math.isnan(prev_low_val)) else None
        except Exception:
            prev_high = None
            prev_low = None

        high_zone, low_zone = detect_liquidity_zone(df, lookback=15)

        last_close_val = None
        try:
            lc = ensure_series(df['Close']).iloc[-1]
            if isinstance(lc, float) and math.isnan(lc):
                last_close_val = None
            else:
                last_close_val = float(lc)
        except Exception:
            last_close_val = None

        if last_close_val is None:
            highest_ce_oi_strike = None
            highest_pe_oi_strike = None
        else:
            highest_ce_oi_strike = round_strike_one_step_away(index, last_close_val + 50)
            highest_pe_oi_strike = round_strike_one_step_away(index, last_close_val - 50)

        bull_liquidity = []
        if prev_low is not None: bull_liquidity.append(prev_low)
        if low_zone is not None: bull_liquidity.append(low_zone)
        if highest_pe_oi_strike is not None: bull_liquidity.append(highest_pe_oi_strike)

        bear_liquidity = []
        if prev_high is not None: bear_liquidity.append(prev_high)
        if high_zone is not None: bear_liquidity.append(high_zone)
        if highest_ce_oi_strike is not None: bear_liquidity.append(highest_ce_oi_strike)

        return bull_liquidity, bear_liquidity
    except Exception:
        return [], []

def liquidity_zone_entry(index, df):
    """Liquidity zone institutional entries"""
    try:
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        last_close = float(close.iloc[-1])
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1]
        
        bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
        
        # Check if price is at liquidity zone with institutional volume
        if current_volume > avg_volume * 2.5 and has_price_conviction(df):
            # Check bull liquidity zones
            for zone in bull_liq:
                if zone and abs(last_close - zone) <= 15:
                    if (close.iloc[-1] > close.iloc[-2] and 
                        has_follow_through(df, "CE") and
                        is_institutional_buying(index, df, "CE")):
                        return "CE"
            
            # Check bear liquidity zones
            for zone in bear_liq:
                if zone and abs(last_close - zone) <= 15:
                    if (close.iloc[-1] < close.iloc[-2] and 
                        has_follow_through(df, "PE") and
                        is_institutional_buying(index, df, "PE")):
                        return "PE"
    except Exception:
        return None
    return None

# 🚨 ADDED: OPENING PLAY STRATEGY FROM YOUR CODE
def institutional_opening_play(index, df):
    """OPENING PLAY: Compulsory entries from 9:15 to 9:45 based on overnight moves"""
    try:
        # Check if we're in opening play hours
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time = ist_now.time()
        
        if not (OPENING_START <= current_time <= OPENING_END):
            return None
            
        prev_high = float(ensure_series(df['High']).iloc[-2])
        prev_low = float(ensure_series(df['Low']).iloc[-2])
        prev_close = float(ensure_series(df['Close']).iloc[-2])
        current_price = float(ensure_series(df['Close']).iloc[-1])
    except Exception:
        return None
        
    volume = ensure_series(df['Volume'])
    vol_avg = volume.rolling(10).mean().iloc[-1] if len(volume) >= 10 else volume.mean()
    vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg > 0 else 1)
    
    # 🚨 OPENING PLAY LOGIC: Strong moves with volume confirmation
    if current_price > prev_high + 15 and vol_ratio > 1.3: 
        return "CE"
    if current_price < prev_low - 15 and vol_ratio > 1.3: 
        return "PE"
    if current_price > prev_close + 25 and vol_ratio > 1.2: 
        return "CE"
    if current_price < prev_close - 25 and vol_ratio > 1.2: 
        return "PE"
    
    return None

# --------- SIGNAL DEDUPLICATION AND COOLDOWN CHECK ---------
def can_send_signal(index, strike, option_type):
    global active_strikes, last_signal_time
    
    current_time = time.time()
    strike_key = f"{index}_{strike}_{option_type}"
    
    if strike_key in active_strikes:
        return False
        
    if index in last_signal_time:
        time_since_last = current_time - last_signal_time[index]
        if time_since_last < signal_cooldown:
            return False
    
    return True

def update_signal_tracking(index, strike, option_type, signal_id):
    global active_strikes, last_signal_time
    
    strike_key = f"{index}_{strike}_{option_type}"
    active_strikes[strike_key] = {
        'signal_id': signal_id,
        'timestamp': time.time(),
        'targets_hit': 0
    }
    
    last_signal_time[index] = time.time()

def update_signal_progress(signal_id, targets_hit):
    for strike_key, data in active_strikes.items():
        if data['signal_id'] == signal_id:
            active_strikes[strike_key]['targets_hit'] = targets_hit
            break

def clear_completed_signal(signal_id):
    global active_strikes
    active_strikes = {k: v for k, v in active_strikes.items() if v['signal_id'] != signal_id}

# --------- INSTITUTIONAL SIGNAL ANALYSIS ---------
def analyze_index_signal_institutional(index):
    """PURE INSTITUTIONAL LOGIC - Perfect entries only"""
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 20 or close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    # 🚨 MUST HAVE INSTITUTIONAL VOLUME FOR ALL STRATEGIES EXCEPT OPENING PLAY
    # Opening play can work with slightly lower volume due to overnight positioning
    
    # 🚨 CHECK OPENING PLAY FIRST (HIGHEST PRIORITY 9:15-9:45)
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time = ist_now.time()
    
    if OPENING_PLAY_ENABLED and (OPENING_START <= current_time <= OPENING_END):
        opening_signal = institutional_opening_play(index, df5)
        if opening_signal:
            # Opening play has slightly relaxed volume requirements
            volume_ok = is_institutional_volume(df5) or (ensure_series(df5['Volume']).iloc[-1] > ensure_series(df5['Volume']).rolling(20).mean().iloc[-1] * 1.8)
            if volume_ok and has_follow_through(df5, opening_signal):
                return opening_signal, df5, False, "opening_play"

    # 🚨 FOR OTHER STRATEGIES: STRICT INSTITUTIONAL VOLUME REQUIRED
    if not is_institutional_volume(df5):
        return None

    # 🚨 INSTITUTIONAL STRATEGIES (IN ORDER OF PRIORITY)
    institutional_strategies = [
        ("institutional_liquidity_grab", institutional_liquidity_grab),
        ("institutional_absorption", institutional_absorption),
        ("institutional_momentum_ignition", institutional_momentum_ignition),
        ("liquidity_sweeps", detect_liquidity_sweeps),
        ("liquidity_zone", liquidity_zone_entry)
    ]
    
    for strategy_name, strategy_func in institutional_strategies:
        signal = strategy_func(index, df5)
        if signal:
            # 🚨 ADDITIONAL CONFIRMATION: Must have price conviction and follow-through
            if has_price_conviction(df5) and has_follow_through(df5, signal):
                return signal, df5, False, strategy_name
    
    return None

# [REST OF THE CODE - monitoring, EOD reports, main loop remains the same]
# ... (continuing with the existing monitor_price_live, send_individual_signal_reports, send_signal, trade_thread, run_algo_parallel functions)

# --------- FIXED: ENHANCED TRADE MONITORING AND TRACKING ---------
active_trades = {}

def calculate_pnl(entry, max_price, targets, targets_hit, sl):
    try:
        if targets is None or len(targets) == 0:
            diff = max_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        
        if not isinstance(targets_hit, (list, tuple)):
            targets_hit = list(targets_hit) if targets_hit is not None else [False]*len(targets)
        if len(targets_hit) < len(targets):
            targets_hit = list(targets_hit) + [False] * (len(targets) - len(targets_hit))
        
        achieved_prices = [target for i, target in enumerate(targets) if targets_hit[i]]
        if achieved_prices:
            exit_price = achieved_prices[-1]
            diff = exit_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        else:
            if max_price <= sl:
                diff = sl - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
            else:
                diff = max_price - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
    except Exception:
        return "0"

def monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data):
    def monitoring_thread():
        global daily_signals
        
        last_high = entry
        weakness_sent = False
        in_trade = False
        entry_price_achieved = False
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        last_activity_time = time.time()
        signal_id = signal_data.get('signal_id')
        
        while True:
            current_time = time.time()
            
            # Check for inactivity (20 minutes)
            if not in_trade and (current_time - last_activity_time) > 1200:
                send_telegram(f"⏰ {symbol}: No activity for 20 minutes. Allowing new signals.", reply_to=thread_id)
                clear_completed_signal(signal_id)
                break
                
            if should_stop_trading():
                try:
                    final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                except Exception:
                    final_pnl = "0"
                signal_data.update({
                    "entry_status": "NOT_ENTERED" if not entry_price_achieved else "ENTERED",
                    "targets_hit": sum(targets_hit),
                    "max_price_reached": max_price_reached,
                    "zero_targets": sum(targets_hit) == 0,
                    "no_new_highs": max_price_reached <= entry,
                    "final_pnl": final_pnl
                })
                daily_signals.append(signal_data)
                clear_completed_signal(signal_id)
                break
                
            price = fetch_option_price(symbol)
            if price:
                last_activity_time = current_time
                price = round(price)
                
                if price > max_price_reached:
                    max_price_reached = price
                
                if not in_trade:
                    if price >= entry:
                        send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                        in_trade = True
                        entry_price_achieved = True
                        last_high = price
                        signal_data["entry_status"] = "ENTERED"
                else:
                    if price > last_high:
                        send_telegram(f"🚀 {symbol} making new high → {price}", reply_to=thread_id)
                        last_high = price
                    elif not weakness_sent and price < sl * 1.05:
                        send_telegram(f"⚡ {symbol} showing weakness near SL {sl}", reply_to=thread_id)
                        weakness_sent = True
                    
                    # Update signal progress
                    current_targets_hit = sum(targets_hit)
                    for i, target in enumerate(targets):
                        if price >= target and not targets_hit[i]:
                            send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                            targets_hit[i] = True
                            current_targets_hit = sum(targets_hit)
                            update_signal_progress(signal_id, current_targets_hit)
                    
                    # SL hit - allow immediate new signal
                    if price <= sl:
                        send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade. ALLOWING NEW SIGNAL.", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": sum(targets_hit),
                            "max_price_reached": max_price_reached,
                            "zero_targets": sum(targets_hit) == 0,
                            "no_new_highs": max_price_reached <= entry,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
                        
                    # 2nd target hit - allow new signals but continue monitoring
                    if current_targets_hit >= 2:
                        update_signal_progress(signal_id, current_targets_hit)
                    
                    # All targets hit - complete trade
                    if all(targets_hit):
                        send_telegram(f"🏆 {symbol}: ALL TARGETS HIT! Trade completed successfully!", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": len(targets),
                            "max_price_reached": max_price_reached,
                            "zero_targets": False,
                            "no_new_highs": False,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
            
            time.sleep(10)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

# --------- FIXED: WORKING EOD REPORT SYSTEM ---------
def send_individual_signal_reports():
    global daily_signals, all_generated_signals
    
    all_signals = daily_signals + all_generated_signals
    
    seen_ids = set()
    unique_signals = []
    for signal in all_signals:
        sid = signal.get('signal_id')
        if not sid:
            continue
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique_signals.append(signal)
    
    if not unique_signals:
        send_telegram("📊 END OF DAY REPORT\nNo signals generated today.")
        return
    
    send_telegram(f"🕒 END OF DAY SIGNAL REPORT - { (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y') }\n"
                  f"📈 Total Signals: {len(unique_signals)}\n"
                  f"─────────────────────────────")
    
    for i, signal in enumerate(unique_signals, 1):
        targets_hit_list = []
        if signal.get('targets_hit', 0) > 0:
            for j in range(signal.get('targets_hit', 0)):
                if j < len(signal.get('targets', [])):
                    targets_hit_list.append(str(signal['targets'][j]))
        
        targets_for_disp = signal.get('targets', [])
        while len(targets_for_disp) < 4:
            targets_for_disp.append('-')
        
        msg = (f"📊 SIGNAL #{i} - {signal.get('index','?')} {signal.get('strike','?')} {signal.get('option_type','?')}\n"
               f"─────────────────────────────\n"
               f"📅 Date: {(datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y')}\n"
               f"🕒 Time: {signal.get('timestamp','?')}\n"
               f"📈 Index: {signal.get('index','?')}\n"
               f"🎯 Strike: {signal.get('strike','?')}\n"
               f"🔰 Type: {signal.get('option_type','?')}\n"
               f"🏷️ Strategy: {signal.get('strategy','?')}\n\n"
               
               f"💰 ENTRY: ₹{signal.get('entry_price','?')}\n"
               f"🎯 TARGETS: {targets_for_disp[0]} // {targets_for_disp[1]} // {targets_for_disp[2]} // {targets_for_disp[3]}\n"
               f"🛑 STOP LOSS: ₹{signal.get('sl','?')}\n\n"
               
               f"📊 PERFORMANCE:\n"
               f"• Entry Status: {signal.get('entry_status', 'PENDING')}\n"
               f"• Targets Hit: {signal.get('targets_hit', 0)}/4\n")
        
        if targets_hit_list:
            msg += f"• Targets Achieved: {', '.join(targets_hit_list)}\n"
        
        msg += (f"• Max Price Reached: ₹{signal.get('max_price_reached', signal.get('entry_price','?'))}\n"
                f"• Final P&L: {signal.get('final_pnl', '0')} points\n\n"
                
                f"⚡ Fakeout: {'YES' if signal.get('fakeout') else 'NO'}\n"
                f"📈 Index Price at Signal: {signal.get('index_price','?')}\n"
                f"🆔 Signal ID: {signal.get('signal_id','?')}\n"
                f"─────────────────────────────")
        
        send_telegram(msg)
        time.sleep(1)
    
    total_pnl = 0.0
    successful_trades = 0
    for signal in unique_signals:
        pnl_str = signal.get("final_pnl", "0")
        try:
            if isinstance(pnl_str, str) and pnl_str.startswith("+"):
                total_pnl += float(pnl_str[1:])
                successful_trades += 1
            elif isinstance(pnl_str, str) and pnl_str.startswith("-"):
                total_pnl -= float(pnl_str[1:])
        except:
            pass
    
    summary_msg = (f"📈 DAY SUMMARY\n"
                   f"─────────────────────────────\n"
                   f"• Total Signals: {len(unique_signals)}\n"
                   f"• Successful Trades: {successful_trades}\n"
                   f"• Success Rate: {(successful_trades/len(unique_signals))*100:.1f}%\n"
                   f"• Total P&L: ₹{total_pnl:+.2f}\n"
                   f"─────────────────────────────")
    
    send_telegram(summary_msg)
    
    send_telegram("✅ END OF DAY REPORTS COMPLETED! See you tomorrow at 9:15 AM! 🚀")

# --------- UPDATED SIGNAL SENDING WITH ONE STEP AWAY STRIKES ---------
def send_signal(index, side, df, fakeout, strategy_key):
    global signal_counter, all_generated_signals
    
    signal_detection_price = float(ensure_series(df["Close"]).iloc[-1])
    strike = round_strike_one_step_away(index, signal_detection_price)
    
    if strike is None:
        send_telegram(f"⚠️ {index}: could not determine strike (price missing). Signal skipped.")
        return
        
    if not can_send_signal(index, strike, side):
        return
        
    symbol = get_option_symbol(index, EXPIRIES[index], strike, side)
    
    if symbol is None:
        print(f"❌ STRICT EXPIRY ENFORCEMENT: {index} {strike}{side} - Only {EXPIRIES[index]} allowed")
        return
    
    option_price = fetch_option_price(symbol)
    if not option_price: 
        return
    
    entry = round(option_price)
    
    high = ensure_series(df["High"])
    low = ensure_series(df["Low"])
    close = ensure_series(df["Close"])
    
    bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
    
    if side == "CE":
        if bull_liq:
            nearest_bull_zone = max([z for z in bull_liq if z is not None])
            price_gap = nearest_bull_zone - signal_detection_price
        else:
            price_gap = signal_detection_price * 0.008
        
        base_move = max(price_gap * 0.3, 40)
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 1.8),
            round(entry + base_move * 2.8),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.8)
        
    else:
        if bear_liq:
            nearest_bear_zone = min([z for z in bear_liq if z is not None])
            price_gap = signal_detection_price - nearest_bear_zone
        else:
            price_gap = signal_detection_price * 0.008
        
        base_move = max(price_gap * 0.3, 40)
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 1.8),
            round(entry + base_move * 2.8),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.8)
    
    targets_str = "//".join(str(t) for t in targets) + "++"
    
    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key.upper())
    
    signal_id = f"SIG{signal_counter:04d}"
    signal_counter += 1
    
    signal_data = {
        "signal_id": signal_id,
        "timestamp": (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M:%S"),
        "index": index,
        "strike": strike,
        "option_type": side,
        "strategy": strategy_name,
        "entry_price": entry,
        "targets": targets,
        "sl": sl,
        "fakeout": fakeout,
        "index_price": signal_detection_price,
        "entry_status": "PENDING",
        "targets_hit": 0,
        "max_price_reached": entry,
        "zero_targets": True,
        "no_new_highs": True,
        "final_pnl": "0"
    }
    
    update_signal_tracking(index, strike, side, signal_id)
    
    all_generated_signals.append(signal_data.copy())
    
    msg = (f"🟢 {index} {strike} {side}\n"
           f"SYMBOL: {symbol}\n"
           f"ABOVE {entry}\n"
           f"TARGETS: {targets_str}\n"
           f"SL: {sl}\n"
           f"FAKEOUT: {'YES' if fakeout else 'NO'}\n"
           f"STRATEGY: {strategy_name}\n"
           f"SIGNAL ID: {signal_id}")
         
    thread_id = send_telegram(msg)
    
    trade_id = f"{symbol}_{int(time.time())}"
    active_trades[trade_id] = {
        "symbol": symbol, 
        "entry": entry, 
        "sl": sl, 
        "targets": targets, 
        "thread": thread_id, 
        "status": "OPEN",
        "index": index,
        "signal_data": signal_data
    }
    
    monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data)

# --------- UPDATED TRADE THREAD WITH INSTITUTIONAL LOGIC ---------
def trade_thread(index):
    result = analyze_index_signal_institutional(index)
    
    if not result:
        return
        
    if len(result) == 4:
        side, df, fakeout, strategy_key = result
    else:
        side, df, fakeout = result
        strategy_key = "unknown"
    
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return
        
    send_signal(index, side, df, fakeout, strategy_key)

# --------- MAIN LOOP WITH INSTITUTIONAL LOGIC ---------
def run_algo_parallel():
    if not is_market_open(): 
        print("❌ Market closed - skipping iteration")
        return
        
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
            
        if not EOD_REPORT_SENT:
            time.sleep(15)
            send_telegram("📊 GENERATING COMPULSORY END-OF-DAY REPORT...")
            try:
                send_individual_signal_reports()
            except Exception as e:
                send_telegram(f"⚠️ EOD Report Error, retrying: {str(e)[:100]}")
                time.sleep(10)
                send_individual_signal_reports()
            EOD_REPORT_SENT = True
            send_telegram("✅ TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")
            
        return
        
    threads = []
    kept_indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    
    for index in kept_indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()

# --------- MAIN EXECUTION ---------
STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

initialize_strategy_tracking()

while True:
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        current_datetime_ist = ist_now
        
        market_open = is_market_open()
        
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market is currently closed. Algorithm waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            if current_time_ist >= dtime(15,30) and current_time_ist <= dtime(16,0) and not EOD_REPORT_SENT:
                send_telegram("📊 GENERATING COMPULSORY END-OF-DAY REPORT...")
                time.sleep(10)
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
                send_telegram("✅ EOD Report completed! Algorithm will resume tomorrow.")
            
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🚀 INSTITUTIONAL ALGO STARTED - 3 Indices Running\n"
                         "✅ Pure Institutional Logic - No Indicators\n"
                         "✅ One Step Away Strikes Only\n"
                         "✅ 2.5x Volume Confirmation Required\n"
                         "✅ Perfect Entries Only\n"
                         "✅ OPENING PLAY (9:15-9:45) Compulsory Entries\n"
                         "✅ STRICT EXPIRY ENFORCEMENT")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached! Preparing EOD Report...")
                STOP_SENT = True
                STARTED_SENT = False
            
            if not EOD_REPORT_SENT:
                send_telegram("📊 FINALIZING TRADES...")
                time.sleep(20)
                try:
                    send_individual_signal_reports()
                except Exception as e:
                    send_telegram(f"⚠️ EOD Report Error, retrying: {str(e)[:100]}")
                    time.sleep(10)
                    send_individual_signal_reports()
                EOD_REPORT_SENT = True
                send_telegram("✅ TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")
            
            time.sleep(60)
            continue
            
        run_algo_parallel()
        time.sleep(30)
        
    except Exception as e:
        error_msg = f"⚠️ Main loop error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)

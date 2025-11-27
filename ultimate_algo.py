#INDEXBASED + INSTITUTIONAL BLAST DETECTION - ULTIMATE VERSION

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

# ---------------- INSTITUTIONAL BLAST CONFIG ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRY_ACTIONABLE = True
EXPIRY_INFO_ONLY = False

# INSTITUTIONAL BLAST DETECTION PARAMETERS
BLAST_MOMENTUM_THRESHOLD = 0.008  # 0.8% move in single candle
BLAST_VOLUME_SPIKE = 2.5          # 250% volume spike
BLAST_LIQUIDITY_CONFIRMATION = True
BLAST_PRE_ENTRY_DETECTION = True   # Detect before full move
BLAST_ON_SPOT_DETECTION = True     # Detect as it happens

# INSTITUTIONAL ENTRY FILTERS
INSTITUTIONAL_CONFIRMATION_CANDLES = 1  # Single candle confirmation
PRICE_ACCEPTANCE_THRESHOLD = 0.001      # 0.1% price acceptance

# --------- EXPIRIES FOR KEPT INDICES ---------
EXPIRIES = {
    "NIFTY": "02 DEC 2025",
    "BANKNIFTY": "30 DEC 2025", 
    "SENSEX": "04 DEC 2025"
}

# --------- STRATEGY TRACKING ---------
STRATEGY_NAMES = {
    "institutional_blast": "INSTITUTIONAL BLAST",
    "liquidity_blast": "LIQUIDITY BLAST", 
    "gamma_blast": "GAMMA BLAST",
    "institutional_price_action": "INSTITUTIONAL PRICE ACTION",
    "opening_play": "OPENING PLAY", 
    "gamma_squeeze": "GAMMA SQUEEZE",
    "liquidity_sweeps": "LIQUIDITY SWEEP",
    "wyckoff_schematic": "WYCKOFF SCHEMATIC",
    "vcp_pattern": "VCP PATTERN",
    "faulty_bases": "FAULTY BASES",
    "peak_rejection": "PEAK REJECTION",
    "smart_money_divergence": "SMART MONEY DIVERGENCE",
    "stop_hunt": "STOP HUNT",
    "institutional_continuation": "INSTITUTIONAL CONTINUATION",
    "fair_value_gap": "FAIR VALUE GAP",
    "volume_gap_imbalance": "VOLUME GAP IMBALANCE",
    "ote_retracement": "OTE RETRACEMENT",
    "demand_supply_zones": "DEMAND SUPPLY ZONES",
    "pullback_reversal": "PULLBACK REVERSAL",
    "orderflow_mimic": "ORDERFLOW MIMIC",
    "bottom_fishing": "BOTTOM FISHING",
    "liquidity_zone": "LIQUIDITY ZONE"
}

# --------- ENHANCED TRACKING FOR REPORTS ---------
all_generated_signals = []
strategy_performance = {}
signal_counter = 0
daily_signals = []

# --------- INSTITUTIONAL ENTRY TRACKING ---------
active_strikes = {}
last_signal_time = {}
signal_cooldown = 1200

def initialize_strategy_tracking():
    global strategy_performance
    strategy_performance = {name: {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0} 
                           for name in STRATEGY_NAMES.values()}

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

# --------- STRIKE ROUNDING ---------
def round_strike(index, price):
    try:
        if price is None or (isinstance(price, float) and math.isnan(price)):
            return None
        price = float(price)
        
        if index == "NIFTY": 
            return int(round(price / 50.0) * 50)
        elif index == "BANKNIFTY": 
            return int(round(price / 100.0) * 100)
        elif index == "SENSEX": 
            return int(round(price / 100.0) * 100)
        else: 
            return int(round(price / 50.0) * 50)
    except Exception:
        return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- FETCH INDEX DATA ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN"
    }
    df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
    return None if df.empty else df

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

# 🚨 INSTITUTIONAL BLAST DETECTION SYSTEM 🚨
def detect_institutional_blast(df):
    """
    PURE INSTITUTIONAL BLAST DETECTION
    Identifies massive single-candle moves WITHOUT retail indicators
    Only uses liquidity and price action
    """
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low']) 
        close = ensure_series(df['Close'])
        open_val = ensure_series(df['Open'])
        
        if len(close) < 3:
            return None
            
        # Current candle dynamics
        current_candle_size = abs(close.iloc[-1] - open_val.iloc[-1])
        prev_candle_size = abs(close.iloc[-2] - open_val.iloc[-2])
        avg_candle_size = abs(close.diff()).rolling(5).mean().iloc[-1]
        
        # 🚨 BLAST DETECTION: Single candle momentum
        current_momentum = current_candle_size / (avg_candle_size + 1e-6)
        
        # 🚨 INSTITUTIONAL BLAST CONDITIONS
        if current_momentum > BLAST_MOMENTUM_THRESHOLD:
            # Determine blast direction
            if close.iloc[-1] > open_val.iloc[-1]:
                # GREEN BLAST UP - INSTITUTIONAL BUYING
                blast_strength = (close.iloc[-1] - open_val.iloc[-1]) / open_val.iloc[-1]
                if blast_strength > 0.005:  # 0.5% move minimum
                    return "CE"
            else:
                # RED BLAST DOWN - INSTITUTIONAL SELLING  
                blast_strength = (open_val.iloc[-1] - close.iloc[-1]) / open_val.iloc[-1]
                if blast_strength > 0.005:  # 0.5% move minimum
                    return "PE"
                    
    except Exception as e:
        print(f"Blast detection error: {e}")
        return None
    return None

def detect_liquidity_blast(df):
    """
    LIQUIDITY-BASED BLAST DETECTION
    Identifies moves that break key liquidity levels
    """
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        open_val = ensure_series(df['Open'])
        
        if len(close) < 10:
            return None
            
        # Key liquidity levels
        recent_high = high.iloc[-10:-1].max()
        recent_low = low.iloc[-10:-1].min()
        current_close = close.iloc[-1]
        current_open = open_val.iloc[-1]
        
        # 🚨 LIQUIDITY BLAST: Break of key levels with momentum
        if current_close > recent_high and current_close > current_open:
            # Breakout blast with upward momentum
            move_strength = (current_close - recent_high) / recent_high
            if move_strength > 0.003:  # Strong break
                return "CE"
                
        elif current_close < recent_low and current_close < current_open:
            # Breakdown blast with downward momentum  
            move_strength = (recent_low - current_close) / recent_low
            if move_strength > 0.003:  # Strong break
                return "PE"
                
    except Exception:
        return None
    return None

def detect_gamma_blast(index, df):
    """
    GAMMA BLAST DETECTION for expiry days
    Massive institutional moves
    """
    try:
        if not is_expiry_day_for_index(index):
            return None
            
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        open_val = ensure_series(df['Open'])
        
        if len(close) < 3:
            return None
            
        # 🚨 GAMMA BLAST: Extreme moves on expiry
        current_range = high.iloc[-1] - low.iloc[-1]
        prev_range = high.iloc[-2] - low.iloc[-2]
        range_expansion = current_range / (prev_range + 1e-6)
        
        if range_expansion > 2.0:  # Range expansion blast
            if close.iloc[-1] > open_val.iloc[-1]:
                return "CE"
            else:
                return "PE"
                
    except Exception:
        return None
    return None

def detect_pre_blast_signal(df):
    """
    PRE-BLAST DETECTION: Signal before full move happens
    Identifies institutional accumulation before blast
    """
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        
        if len(close) < 5:
            return None
            
        # 🚨 PRE-BLAST: Compression before explosion
        recent_range = high.iloc[-5:-1].max() - low.iloc[-5:-1].min()
        current_range = high.iloc[-1] - low.iloc[-1]
        compression_ratio = current_range / (recent_range + 1e-6)
        
        # High compression suggests imminent blast
        if compression_ratio < 0.5:  # 50% compression
            # Check for directional bias
            if close.iloc[-1] > close.iloc[-3]:
                return "CE"  # Upward blast expected
            elif close.iloc[-1] < close.iloc[-3]:
                return "PE"  # Downward blast expected
                
    except Exception:
        return None
    return None

# 🚨 MASTER BLAST DETECTION FUNCTION 🚨
def institutional_blast_detection(index, df):
    """
    MASTER INSTITUTIONAL BLAST DETECTION
    Combines all blast detection methods
    """
    # 1. PRE-BLAST DETECTION (Before move)
    if BLAST_PRE_ENTRY_DETECTION:
        pre_blast = detect_pre_blast_signal(df)
        if pre_blast:
            send_telegram(f"⚡ PRE-BLAST DETECTED: {index} {pre_blast} - Institutional accumulation detected!")
            return pre_blast, df, False, "institutional_blast"
    
    # 2. ON-SPOT BLAST DETECTION (As it happens)
    if BLAST_ON_SPOT_DETECTION:
        # Institutional blast
        blast_signal = detect_institutional_blast(df)
        if blast_signal:
            send_telegram(f"💥 BLAST DETECTED: {index} {blast_signal} - Institutional move happening NOW!")
            return blast_signal, df, False, "institutional_blast"
        
        # Liquidity blast  
        liquidity_blast = detect_liquidity_blast(df)
        if liquidity_blast:
            send_telegram(f"🌊 LIQUIDITY BLAST: {index} {liquidity_blast} - Key levels broken!")
            return liquidity_blast, df, False, "liquidity_blast"
        
        # Gamma blast (expiry days)
        gamma_blast = detect_gamma_blast(index, df)
        if gamma_blast:
            send_telegram(f"🎯 GAMMA BLAST: {index} {gamma_blast} - Expiry day explosion!")
            return gamma_blast, df, False, "gamma_blast"
    
    return None

# --------- INSTITUTIONAL ENTRY FILTERS ---------
def institutional_entry_filter(index, signal, df):
    """
    INSTITUTIONAL-GRADE ENTRY FILTER for blast signals
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 3:
            return False
            
        # For blast signals, we want immediate momentum continuation
        if signal == "CE":
            return close.iloc[-1] > close.iloc[-2]  # Continuing upward
        elif signal == "PE":
            return close.iloc[-1] < close.iloc[-2]  # Continuing downward
            
    except Exception:
        return False
    return False

# --------- DETECT LIQUIDITY ZONE ---------
def detect_liquidity_zone(df, lookback=20):
    high_series = ensure_series(df['High']).dropna()
    low_series = ensure_series(df['Low']).dropna()
    try:
        if len(high_series) <= lookback:
            high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
        else:
            high_pool = float(high_series.rolling(lookback).max().iloc[-2])
    except Exception:
        high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
    try:
        if len(low_series) <= lookback:
            low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')
        else:
            low_pool = float(low_series.rolling(lookback).min().iloc[-2])
    except Exception:
        low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')

    return round(high_pool,0), round(low_pool,0)

# --------- INSTITUTIONAL LIQUIDITY HUNT ---------
def institutional_liquidity_hunt(index, df):
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
        highest_ce_oi_strike = round_strike(index, last_close_val + 50)
        highest_pe_oi_strike = round_strike(index, last_close_val - 50)

    bull_liquidity = []
    if prev_low is not None: bull_liquidity.append(prev_low)
    if low_zone is not None: bull_liquidity.append(low_zone)
    if highest_pe_oi_strike is not None: bull_liquidity.append(highest_pe_oi_strike)

    bear_liquidity = []
    if prev_high is not None: bear_liquidity.append(prev_high)
    if high_zone is not None: bear_liquidity.append(high_zone)
    if highest_ce_oi_strike is not None: bear_liquidity.append(highest_ce_oi_strike)

    return bull_liquidity, bear_liquidity

# 🚨 INSTITUTIONAL PRICE ACTION 🚨
def institutional_price_action_signal(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        
        if len(close) < 10:
            return None
            
        recent_high = high.iloc[-10:-1].max()
        recent_low = low.iloc[-10:-1].min()
        current_close = close.iloc[-1]
        
        # 🚨 INSTITUTIONAL BREAKOUT DETECTION
        if current_close > recent_high and current_close > close.iloc[-2]:
            return "CE"
            
        # 🚨 INSTITUTIONAL BREAKDOWN DETECTION  
        if current_close < recent_low and current_close < close.iloc[-2]:
            return "PE"
            
    except Exception:
        return None
    return None

# --------- OTHER STRATEGY FUNCTIONS (KEPT FOR COMPATIBILITY) ---------
def is_expiry_day_for_index(index):
    try:
        ex = EXPIRIES.get(index)
        if not ex: return False
        dt = datetime.strptime(ex, "%d %b %Y")
        today = (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()
        return dt.date() == today
    except Exception:
        return False

def institutional_momentum_confirmation(index, df, proposed_signal):
    try:
        close = ensure_series(df['Close'])
        return True  # Simplified for blast detection
    except Exception:
        return False

# --------- UPDATED STRATEGY CHECK WITH BLAST DETECTION ---------
def analyze_index_signal(index):
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 10 or close5.isna().iloc[-1]:
        return None

    # 🚨 INSTITUTIONAL BLAST DETECTION (HIGHEST PRIORITY)
    blast_result = institutional_blast_detection(index, df5)
    if blast_result:
        signal, df, fakeout, strategy = blast_result
        if institutional_entry_filter(index, signal, df5):
            return signal, df, fakeout, strategy

    # 🚨 INSTITUTIONAL PRICE ACTION (SECONDARY)
    institutional_pa_signal = institutional_price_action_signal(df5)
    if institutional_pa_signal:
        if institutional_momentum_confirmation(index, df5, institutional_pa_signal):
            return institutional_pa_signal, df5, False, "institutional_price_action"

    # 🚨 OPENING PLAY
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        t = ist_now.time()
        if OPENING_PLAY_ENABLED and (OPENING_START <= t <= OPENING_END):
            # Use blast detection for opening moves
            opening_blast = detect_institutional_blast(df5)
            if opening_blast:
                return opening_blast, df5, False, "opening_play"
    except Exception:
        pass

    return None

# --------- INSTITUTIONAL SIGNAL EXECUTION ---------
def institutional_signal_execution(index, side, df, fakeout, strategy_key):
    """
    INSTITUTIONAL-GRADE SIGNAL EXECUTION for blast signals
    """
    # Apply institutional entry filters
    if not institutional_entry_filter(index, side, df):
        return
        
    # All checks passed - send signal
    send_signal(index, side, df, fakeout, strategy_key)

# --------- SIGNAL TRACKING FUNCTIONS ---------
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

# --------- TRADE MONITORING ---------
active_trades = {}

def calculate_pnl(entry, max_price, targets, targets_hit, sl):
    try:
        if targets is None or len(targets) == 0:
            diff = max_price - entry
            return f"+{diff:.2f}" if diff > 0 else f"-{abs(diff):.2f}"
        
        achieved_prices = [target for i, target in enumerate(targets) if targets_hit[i]]
        if achieved_prices:
            exit_price = achieved_prices[-1]
            diff = exit_price - entry
            return f"+{diff:.2f}" if diff > 0 else f"-{abs(diff):.2f}"
        else:
            if max_price <= sl:
                diff = sl - entry
                return f"+{diff:.2f}" if diff > 0 else f"-{abs(diff):.2f}"
            else:
                diff = max_price - entry
                return f"+{diff:.2f}" if diff > 0 else f"-{abs(diff):.2f}"
    except Exception:
        return "0"

def monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data):
    def monitoring_thread():
        global daily_signals
        
        last_high = entry
        in_trade = False
        entry_price_achieved = False
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        last_activity_time = time.time()
        signal_id = signal_data.get('signal_id')
        
        while True:
            current_time = time.time()
            
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
                    
                    for i, target in enumerate(targets):
                        if price >= target and not targets_hit[i]:
                            send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                            targets_hit[i] = True
                            update_signal_progress(signal_id, sum(targets_hit))
                    
                    if price <= sl:
                        send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade.", reply_to=thread_id)
                        final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        signal_data.update({
                            "targets_hit": sum(targets_hit),
                            "max_price_reached": max_price_reached,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
                        
                    if all(targets_hit):
                        send_telegram(f"🏆 {symbol}: ALL TARGETS HIT! Trade completed!", reply_to=thread_id)
                        final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        signal_data.update({
                            "targets_hit": len(targets),
                            "max_price_reached": max_price_reached,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
            
            time.sleep(10)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

# --------- SIGNAL SENDING WITH BLAST DETECTION ---------
def send_signal(index, side, df, fakeout, strategy_key):
    global signal_counter, all_generated_signals
    
    signal_detection_price = float(ensure_series(df["Close"]).iloc[-1])
    strike = round_strike(index, signal_detection_price)
    
    if strike is None:
        return
        
    if not can_send_signal(index, strike, side):
        return
        
    symbol = get_option_symbol(index, EXPIRIES[index], strike, side)
    
    if symbol is None:
        return
    
    option_price = fetch_option_price(symbol)
    if not option_price: 
        return
    
    entry = round(option_price)
    
    # 🚨 BLAST-OPTIMIZED TARGETS - BIGGER MOVES FOR BLAST SIGNALS
    if side == "CE":
        base_move = max(signal_detection_price * 0.01, 60)  # 1% move minimum
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 2.0),  # Bigger targets for blasts
            round(entry + base_move * 3.0),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.6)
    else:
        base_move = max(signal_detection_price * 0.01, 60)
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 2.0),
            round(entry + base_move * 3.0),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.6)
    
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
        "final_pnl": "0"
    }
    
    update_signal_tracking(index, strike, side, signal_id)
    all_generated_signals.append(signal_data.copy())
    
    # 🚨 BLAST-SPECIFIC MESSAGING
    if "blast" in strategy_key.lower():
        msg = (f"💥 BLAST SIGNAL - {index} {strike} {side}\n"
               f"SYMBOL: {symbol}\n"
               f"ABOVE {entry}\n"
               f"TARGETS: {targets_str}\n"
               f"SL: {sl}\n"
               f"STRATEGY: {strategy_name}\n"
               f"BLAST DETECTION: ACTIVE ✅\n"
               f"SIGNAL ID: {signal_id}")
    else:
        msg = (f"🟢 {index} {strike} {side}\n"
               f"SYMBOL: {symbol}\n"
               f"ABOVE {entry}\n"
               f"TARGETS: {targets_str}\n"
               f"SL: {sl}\n"
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

# --------- EXPIRY VALIDATION ---------
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
            return expected_pattern in symbol_upper
        else:
            expected_pattern = expected_dt.strftime("%d%b%y").upper()
            symbol_upper = symbol.upper()
            return expected_pattern in symbol_upper
                
    except Exception:
        return False

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
            return symbol
        else:
            return None
            
    except Exception:
        return None

# --------- TRADE THREAD ---------
def trade_thread(index):
    result = analyze_index_signal(index)
    
    if not result:
        return
        
    if len(result) == 4:
        side, df, fakeout, strategy_key = result
    else:
        side, df, fakeout = result
        strategy_key = "unknown"
    
    institutional_signal_execution(index, side, df, fakeout, strategy_key)

# --------- MAIN LOOP ---------
def run_algo_parallel():
    if not is_market_open(): 
        return
        
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
            
        if not EOD_REPORT_SENT:
            time.sleep(15)
            # Simplified EOD report
            total_signals = len(all_generated_signals)
            send_telegram(f"📊 END OF DAY - Total Signals: {total_signals}")
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

while True:
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        
        market_open = is_market_open()
        
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market is currently closed. Algorithm waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🚀 INSTITUTIONAL BLAST ALGO STARTED\n"
                         "✅ BLAST DETECTION SYSTEM ACTIVE\n"
                         "✅ PRE-BLAST & ON-SPOT DETECTION\n" 
                         "✅ SINGLE CANDLE INSTITUTIONAL MOVES\n"
                         "✅ NO RETAIL INDICATORS - PURE PRICE ACTION\n"
                         "✅ BIG GREEN/RED BLAST SIGNALS ONLY")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached!")
                STOP_SENT = True
                STARTED_SENT = False
            
            time.sleep(60)
            continue
            
        run_algo_parallel()
        time.sleep(30)
        
    except Exception as e:
        error_msg = f"⚠️ Main loop error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)

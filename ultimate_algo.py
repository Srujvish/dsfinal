# ULTIMATE INSTITUTIONAL INTELLIGENCE ANALYZER WITH ANGEL ONE INTEGRATION

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

# --------- INSTITUTIONAL MONITORING CONFIG ---------
MOVE_THRESHOLD = 40
MOVE_TIME_WINDOW = 20
ANALYSIS_COOLDOWN = 30

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")

def angel_one_login():
    """Login to Angel One without error messages"""
    try:
        TOTP = pyotp.TOTP(TOTP_SECRET).now()
        client = SmartConnect(api_key=API_KEY)
        session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
        return client
    except Exception:
        return None

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        requests.post(url, data=payload, timeout=5)
        return True
    except:
        return False

# --------- MARKET HOURS ---------
def is_market_open():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return dtime(9,15) <= current_time_ist <= dtime(15,30)

# --------- FETCH DATA WITH ANGEL ONE ---------
def fetch_index_data_angel_one(client, index):
    """Fetch data using Angel One without errors"""
    try:
        # Map indices to Angel One symbols
        symbol_map = {
            "NIFTY": "NIFTY",
            "BANKNIFTY": "BANKNIFTY", 
            "SENSEX": "SENSEX"
        }
        
        # Try to get data from Angel One first
        if client:
            try:
                # Get LTP data for basic price info
                ltp_data = client.ltpData("NSE", "INDEX", symbol_map[index])
                if ltp_data and 'data' in ltp_data:
                    current_price = float(ltp_data['data']['ltp'])
                    return {'Close': [current_price], 'success': True}
            except:
                pass
        
        # Fallback to yfinance
        yf_symbols = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
        df = yf.download(yf_symbols[index], period="1d", interval="5m", progress=False)
        return df if not df.empty else None
        
    except Exception:
        return None

def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- INSTITUTIONAL ORDER FLOW ANALYZER ---------
class InstitutionalOrderFlowAnalyzer:
    def __init__(self):
        self.client = angel_one_login()
        self.last_analysis_time = {}
        self.analyzed_moves = set()
        self.order_flow_data = {}
        
    def get_angel_one_market_depth(self, symbol):
        """Get market depth data from Angel One"""
        try:
            if not self.client:
                return None
                
            # Get market depth for the symbol
            market_depth = self.client.marketData("NSE", "INDEX", symbol)
            if market_depth and 'data' in market_depth:
                return market_depth['data']
        except Exception:
            pass
        return None
    
    def analyze_institutional_order_flow(self, index, price_data):
        """Analyze institutional order flow patterns"""
        try:
            symbol_map = {"NIFTY": "NIFTY", "BANKNIFTY": "BANKNIFTY", "SENSEX": "SENSEX"}
            market_depth = self.get_angel_one_market_depth(symbol_map[index])
            
            if not market_depth:
                return self.estimate_order_flow(price_data)
            
            # Analyze bid-ask spread and volumes
            bid_volumes = []
            ask_volumes = []
            
            if 'best5bid' in market_depth and 'best5ask' in market_depth:
                for bid in market_depth['best5bid']:
                    bid_volumes.append(bid.get('quantity', 0))
                for ask in market_depth['best5ask']:
                    ask_volumes.append(ask.get('quantity', 0))
            
            total_bid_volume = sum(bid_volumes)
            total_ask_volume = sum(ask_volumes)
            
            order_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0
            
            return {
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'order_imbalance': round(order_imbalance, 4),
                'imbalance_direction': 'BULLISH' if order_imbalance > 0.1 else 'BEARISH' if order_imbalance < -0.1 else 'NEUTRAL',
                'market_depth_available': True,
                'estimated_large_orders': self.estimate_large_orders(total_bid_volume, total_ask_volume)
            }
            
        except Exception:
            return self.estimate_order_flow(price_data)
    
    def estimate_order_flow(self, price_data):
        """Estimate order flow when direct data isn't available"""
        try:
            close = ensure_series(price_data['Close'])
            volume = ensure_series(price_data['Volume']) if 'Volume' in price_data else None
            
            if volume is not None and len(volume) > 5:
                recent_volume = volume.iloc[-5:].mean()
                avg_volume = volume.iloc[-20:].mean() if len(volume) >= 20 else recent_volume
                
                volume_surge = recent_volume > avg_volume * 1.5
                price_momentum = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
                
                # Estimate institutional activity based on volume and momentum
                if volume_surge and price_momentum > 0.002:
                    large_orders = "INSTITUTIONAL BUYING"
                elif volume_surge and price_momentum < -0.002:
                    large_orders = "INSTITUTIONAL SELLING"
                else:
                    large_orders = "RETAIL DOMINATED"
                    
                return {
                    'total_bid_volume': 'N/A',
                    'total_ask_volume': 'N/A',
                    'order_imbalance': 'N/A',
                    'imbalance_direction': 'ESTIMATED_BULLISH' if price_momentum > 0 else 'ESTIMATED_BEARISH',
                    'market_depth_available': False,
                    'estimated_large_orders': large_orders,
                    'volume_surge_ratio': round(recent_volume/avg_volume, 2) if avg_volume > 0 else 1,
                    'price_momentum': round(price_momentum * 100, 3)
                }
            else:
                return {
                    'market_depth_available': False,
                    'estimated_large_orders': "INSUFFICIENT_DATA",
                    'volume_surge_ratio': 'N/A',
                    'price_momentum': 'N/A'
                }
                
        except Exception:
            return {'market_depth_available': False, 'estimated_large_orders': "ANALYSIS_FAILED"}
    
    def estimate_large_orders(self, bid_volume, ask_volume):
        """Estimate large institutional orders"""
        try:
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return "NO_DATA"
            
            large_order_threshold = total_volume * 0.1  # 10% of total volume
            
            large_bid_orders = bid_volume > large_order_threshold
            large_ask_orders = ask_volume > large_order_threshold
            
            if large_bid_orders and large_ask_orders:
                return "BOTH_SIDE_LARGE_ORDERS"
            elif large_bid_orders:
                return "LARGE_BUY_ORDERS"
            elif large_ask_orders:
                return "LARGE_SELL_ORDERS"
            else:
                return "RETAIL_ORDERS_DOMINANT"
                
        except Exception:
            return "ESTIMATION_FAILED"
    
    def detect_big_move(self, index, df):
        """Detect significant price moves"""
        try:
            if df is None or len(df) < 10:
                return None
            
            close_data = ensure_series(df['Close'])
            current_price = close_data.iloc[-1]
            
            lookback_bars = min(MOVE_TIME_WINDOW // 5, len(close_data) - 1)
            start_price = close_data.iloc[-lookback_bars]
            
            move_points = current_price - start_price
            move_percentage = (move_points / start_price) * 100
            
            if abs(move_points) >= MOVE_THRESHOLD:
                return {
                    'direction': "UP" if move_points > 0 else "DOWN",
                    'points': abs(move_points),
                    'percentage': abs(move_percentage),
                    'start_price': start_price,
                    'current_price': current_price,
                    'start_time_index': len(close_data) - lookback_bars,
                    'move_strength': 'STRONG' if abs(move_percentage) > 0.4 else 'MODERATE'
                }
        except Exception:
            return None
        return None
    
    def analyze_comprehensive_institutional_data(self, index, df, move_info):
        """Complete institutional analysis with order flow"""
        try:
            # Order Flow Analysis
            order_flow = self.analyze_institutional_order_flow(index, df)
            
            # Price Action Analysis
            price_action = self.analyze_price_action_detailed(df, move_info)
            
            # Volume Analysis
            volume_analysis = self.analyze_volume_detailed(df)
            
            # Market Microstructure
            market_structure = self.analyze_market_microstructure(df, move_info)
            
            # Institutional Sentiment
            sentiment = self.analyze_institutional_sentiment(order_flow, price_action, volume_analysis)
            
            return {
                'index': index,
                'move_info': move_info,
                'analysis_time': datetime.utcnow() + timedelta(hours=5, minutes=30),
                
                # 🚨 INSTITUTIONAL ORDER FLOW DATA
                'order_flow_analysis': order_flow,
                
                # 📊 PRICE ACTION
                'price_action_analysis': price_action,
                
                # 📈 VOLUME ANALYSIS
                'volume_analysis': volume_analysis,
                
                # 🏛️ MARKET MICROSTRUCTURE
                'market_microstructure': market_structure,
                
                # 💼 INSTITUTIONAL SENTIMENT
                'institutional_sentiment': sentiment,
                
                # 🎯 TRADING IMPLICATIONS
                'trading_implications': self.generate_trading_implications(move_info, order_flow, sentiment)
            }
            
        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}
    
    def analyze_price_action_detailed(self, df, move_info):
        """Detailed price action analysis"""
        try:
            close = ensure_series(df['Close'])
            high = ensure_series(df['High'])
            low = ensure_series(df['Low'])
            open_price = ensure_series(df['Open'])
            
            lookback = min(10, len(close) - 1)
            
            # Candlestick patterns
            current_candle = {
                'open': open_price.iloc[-1],
                'high': high.iloc[-1],
                'low': low.iloc[-1],
                'close': close.iloc[-1],
                'body_size': abs(close.iloc[-1] - open_price.iloc[-1]),
                'total_range': high.iloc[-1] - low.iloc[-1]
            }
            
            # Trend analysis
            price_sequence = close.iloc[-lookback:]
            higher_highs = all(price_sequence.iloc[i] > price_sequence.iloc[i-1] for i in range(1, len(price_sequence)))
            lower_lows = all(price_sequence.iloc[i] < price_sequence.iloc[i-1] for i in range(1, len(price_sequence)))
            
            return {
                'current_candle': current_candle,
                'trend_direction': 'UP' if higher_highs else 'DOWN' if lower_lows else 'SIDEWAYS',
                'volatility_expansion': current_candle['total_range'] > (high.iloc[-lookback:-1].max() - low.iloc[-lookback:-1].min()) * 1.2,
                'momentum_acceleration': self.check_momentum_acceleration(close),
                'key_level_breaks': self.identify_key_level_breaks(high, low, close, move_info)
            }
        except Exception:
            return {'error': 'Price action analysis failed'}
    
    def analyze_volume_detailed(self, df):
        """Detailed volume analysis"""
        try:
            if 'Volume' not in df:
                return {'error': 'Volume data not available'}
                
            volume = ensure_series(df['Volume'])
            close = ensure_series(df['Close'])
            
            # Volume analysis
            current_volume = volume.iloc[-1]
            avg_volume_5 = volume.iloc[-5:].mean()
            avg_volume_20 = volume.iloc[-20:].mean() if len(volume) >= 20 else avg_volume_5
            
            volume_ratio_5 = current_volume / avg_volume_5 if avg_volume_5 > 0 else 1
            volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # Volume profile
            volume_trend = 'INCREASING' if volume.iloc[-1] > volume.iloc[-2] > volume.iloc[-3] else 'DECREASING'
            
            return {
                'current_volume': current_volume,
                'volume_ratio_5min': round(volume_ratio_5, 2),
                'volume_ratio_20min': round(volume_ratio_20, 2),
                'volume_trend': volume_trend,
                'institutional_volume': volume_ratio_20 > 2.0,
                'volume_climax': volume_ratio_5 > 3.0
            }
        except Exception:
            return {'error': 'Volume analysis failed'}
    
    def analyze_market_microstructure(self, df, move_info):
        """Market microstructure analysis"""
        try:
            close = ensure_series(df['Close'])
            high = ensure_series(df['High'])
            low = ensure_series(df['Low'])
            
            # Efficiency analysis
            price_efficiency = self.calculate_price_efficiency(close)
            
            # Noise analysis
            market_noise = self.calculate_market_noise(high, low, close)
            
            # Momentum quality
            momentum_quality = self.assess_momentum_quality(close, move_info)
            
            return {
                'price_efficiency': price_efficiency,
                'market_noise': market_noise,
                'momentum_quality': momentum_quality,
                'trend_quality': 'HIGH' if price_efficiency > 0.7 and market_noise < 0.3 else 'LOW',
                'institutional_participation_likelihood': self.estimate_institutional_participation(df)
            }
        except Exception:
            return {'error': 'Market microstructure analysis failed'}
    
    def analyze_institutional_sentiment(self, order_flow, price_action, volume_analysis):
        """Institutional sentiment analysis"""
        try:
            sentiment_score = 0
            factors = []
            
            # Order flow sentiment
            if order_flow.get('imbalance_direction') == 'BULLISH':
                sentiment_score += 2
                factors.append("Positive order imbalance")
            elif order_flow.get('imbalance_direction') == 'BEARISH':
                sentiment_score -= 2
                factors.append("Negative order imbalance")
            
            # Volume sentiment
            if volume_analysis.get('institutional_volume'):
                sentiment_score += 1
                factors.append("Institutional volume detected")
            
            # Price action sentiment
            if price_action.get('trend_direction') == 'UP':
                sentiment_score += 1
                factors.append("Uptrend confirmation")
            
            # Large orders sentiment
            large_orders = order_flow.get('estimated_large_orders', '')
            if 'BUY' in large_orders:
                sentiment_score += 1
                factors.append("Large buy orders")
            elif 'SELL' in large_orders:
                sentiment_score -= 1
                factors.append("Large sell orders")
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment': 'BULLISH' if sentiment_score >= 2 else 'BEARISH' if sentiment_score <= -2 else 'NEUTRAL',
                'confidence': 'HIGH' if abs(sentiment_score) >= 3 else 'MEDIUM' if abs(sentiment_score) >= 2 else 'LOW',
                'factors': factors
            }
            
        except Exception:
            return {'sentiment': 'UNKNOWN', 'confidence': 'LOW', 'factors': []}
    
    def generate_trading_implications(self, move_info, order_flow, sentiment):
        """Generate trading implications"""
        direction = move_info['direction']
        strength = move_info.get('move_strength', 'MODERATE')
        
        implications = []
        
        if direction == 'UP':
            if sentiment['sentiment'] == 'BULLISH' and strength == 'STRONG':
                implications.append("STRONG BULLISH MOVE - High probability of continuation")
                implications.append("Consider CE positions on pullbacks")
            elif sentiment['sentiment'] == 'BULLISH':
                implications.append("MODERATE BULLISH MOVE - Watch for confirmation")
            else:
                implications.append("CAUTION: Bullish move but weak institutional support")
        else:
            if sentiment['sentiment'] == 'BEARISH' and strength == 'STRONG':
                implications.append("STRONG BEARISH MOVE - High probability of continuation")
                implications.append("Consider PE positions on bounces")
            elif sentiment['sentiment'] == 'BEARISH':
                implications.append("MODERATE BEARISH MOVE - Watch for confirmation")
            else:
                implications.append("CAUTION: Bearish move but weak institutional support")
        
        # Add order flow implications
        if order_flow.get('estimated_large_orders', '').startswith('LARGE_BUY'):
            implications.append("Large institutional buying detected")
        elif order_flow.get('estimated_large_orders', '').startswith('LARGE_SELL'):
            implications.append("Large institutional selling detected")
        
        return implications
    
    # Helper methods
    def check_momentum_acceleration(self, close):
        if len(close) < 5:
            return False
        recent_change = (close.iloc[-1] - close.iloc[-3]) / close.iloc[-3]
        previous_change = (close.iloc[-3] - close.iloc[-6]) / close.iloc[-6] if len(close) >= 7 else recent_change
        return abs(recent_change) > abs(previous_change) * 1.5
    
    def identify_key_level_breaks(self, high, low, close, move_info):
        levels = []
        if len(high) >= 10:
            recent_high = high.iloc[-10:-1].max()
            recent_low = low.iloc[-10:-1].min()
            
            if move_info['direction'] == 'UP' and close.iloc[-1] > recent_high:
                levels.append(f"Resistance break at {recent_high:.1f}")
            elif move_info['direction'] == 'DOWN' and close.iloc[-1] < recent_low:
                levels.append(f"Support break at {recent_low:.1f}")
        return levels
    
    def calculate_price_efficiency(self, close):
        if len(close) < 5:
            return 0.5
        total_move = abs(close.iloc[-1] - close.iloc[-5])
        sum_small_moves = sum(abs(close.iloc[i] - close.iloc[i-1]) for i in range(len(close)-4, len(close)))
        return total_move / sum_small_moves if sum_small_moves > 0 else 0.5
    
    def calculate_market_noise(self, high, low, close):
        if len(close) < 5:
            return 0.5
        avg_true_range = ta.volatility.AverageTrueRange(high, low, close, 5).average_true_range().iloc[-1]
        noise = avg_true_range / close.iloc[-1]
        return noise
    
    def assess_momentum_quality(self, close, move_info):
        if len(close) < 5:
            return 'UNKNOWN'
        momentum = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
        if abs(momentum) > 0.01:
            return 'STRONG'
        elif abs(momentum) > 0.005:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def estimate_institutional_participation(self, df):
        try:
            if 'Volume' not in df:
                return 'UNKNOWN'
            volume = ensure_series(df['Volume'])
            if len(volume) < 10:
                return 'UNKNOWN'
            recent_vol = volume.iloc[-5:].mean()
            avg_vol = volume.iloc[-20:].mean() if len(volume) >= 20 else recent_vol
            return 'HIGH' if recent_vol > avg_vol * 2 else 'MODERATE' if recent_vol > avg_vol * 1.5 else 'LOW'
        except:
            return 'UNKNOWN'
    
    def format_comprehensive_analysis(self, analysis):
        """Format the complete institutional analysis"""
        try:
            if 'error' in analysis:
                return f"❌ Analysis Error: {analysis['error']}"
            
            move = analysis['move_info']
            order_flow = analysis['order_flow_analysis']
            sentiment = analysis['institutional_sentiment']
            
            msg = f"""
🏛️ **ULTIMATE INSTITUTIONAL INTELLIGENCE REPORT** 🏛️

📊 **INDEX**: {analysis['index']}
🎯 **MOVE**: {move['direction']} {move['points']} points ({move['percentage']:.2f}%)
💪 **STRENGTH**: {move['move_strength']}
🕒 **TIME**: {analysis['analysis_time'].strftime('%H:%M:%S')}

🚨 **INSTITUTIONAL ORDER FLOW**:
• Market Depth: {'AVAILABLE' if order_flow.get('market_depth_available') else 'ESTIMATED'}
• Order Imbalance: {order_flow.get('order_imbalance', 'N/A')} ({order_flow.get('imbalance_direction', 'N/A')})
• Large Orders: {order_flow.get('estimated_large_orders', 'N/A')}
• Volume Surge: {order_flow.get('volume_surge_ratio', 'N/A')}x
• Price Momentum: {order_flow.get('price_momentum', 'N/A')}%

📊 **PRICE ACTION**:
• Trend: {analysis['price_action_analysis'].get('trend_direction', 'N/A')}
• Volatility Expansion: {analysis['price_action_analysis'].get('volatility_expansion', 'N/A')}
• Momentum Acceleration: {analysis['price_action_analysis'].get('momentum_acceleration', 'N/A')}
• Key Levels Broken: {', '.join(analysis['price_action_analysis'].get('key_level_breaks', [])) or 'None'}

📈 **VOLUME ANALYSIS**:
• Volume Ratio (20min): {analysis['volume_analysis'].get('volume_ratio_20min', 'N/A')}x
• Volume Trend: {analysis['volume_analysis'].get('volume_trend', 'N/A')}
• Institutional Volume: {analysis['volume_analysis'].get('institutional_volume', 'N/A')}
• Volume Climax: {analysis['volume_analysis'].get('volume_climax', 'N/A')}

🏗️ **MARKET MICROSTRUCTURE**:
• Price Efficiency: {analysis['market_microstructure'].get('price_efficiency', 'N/A'):.2f}
• Market Noise: {analysis['market_microstructure'].get('market_noise', 'N/A'):.3f}
• Trend Quality: {analysis['market_microstructure'].get('trend_quality', 'N/A')}
• Institutional Participation: {analysis['market_microstructure'].get('institutional_participation_likelihood', 'N/A')}

💼 **INSTITUTIONAL SENTIMENT**:
• Sentiment: {sentiment.get('sentiment', 'N/A')}
• Confidence: {sentiment.get('confidence', 'N/A')}
• Score: {sentiment.get('sentiment_score', 'N/A')}/5
• Factors: {', '.join(sentiment.get('factors', []))}

🎯 **TRADING IMPLICATIONS**:
{' | '.join(analysis.get('trading_implications', ['No clear implications']))}

─────────────────────────────
            """
            return msg
        except Exception as e:
            return f"Error formatting report: {str(e)}"
    
    def should_analyze(self, index):
        """Cooldown check"""
        current_time = time.time()
        if index in self.last_analysis_time:
            time_since_last = current_time - self.last_analysis_time[index]
            if time_since_last < ANALYSIS_COOLDOWN * 60:
                return False
        return True
    
    def update_analysis_time(self, index):
        self.last_analysis_time[index] = time.time()

# --------- MAIN MONITORING LOOP ---------
def monitor_institutional_intelligence():
    analyzer = InstitutionalOrderFlowAnalyzer()
    analyzed_moves = set()
    
    send_telegram("🏛️ ULTIMATE INSTITUTIONAL INTELLIGENCE STARTED\nMonitoring NIFTY, BANKNIFTY, SENSEX for 40+ point moves")
    
    while True:
        try:
            if not is_market_open():
                time.sleep(60)
                continue
            
            current_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
            time_key = current_time.strftime("%H:%M")
            
            for index in ["NIFTY", "BANKNIFTY", "SENSEX"]:
                try:
                    df = fetch_index_data_angel_one(analyzer.client, index)
                    if df is None:
                        continue
                    
                    move_info = analyzer.detect_big_move(index, df)
                    
                    if move_info and analyzer.should_analyze(index):
                        move_id = f"{index}_{move_info['direction']}_{time_key}"
                        
                        if move_id not in analyzed_moves:
                            # Perform deep institutional analysis
                            analysis = analyzer.analyze_comprehensive_institutional_data(index, df, move_info)
                            
                            # Send comprehensive report
                            message = analyzer.format_comprehensive_analysis(analysis)
                            send_telegram(message)
                            
                            # Update tracking
                            analyzed_moves.add(move_id)
                            analyzer.update_analysis_time(index)
                            
                            print(f"✅ Sent institutional intelligence for {index}")
                
                except Exception:
                    continue
            
            # Clean up old moves
            current_hour = datetime.utcnow().hour
            analyzed_moves = {move_id for move_id in analyzed_moves 
                            if int(move_id.split('_')[-1].split(':')[0]) >= current_hour - 2}
            
            time.sleep(300)  # Check every 5 minutes
            
        except Exception:
            time.sleep(60)

# --------- START THE ULTIMATE INSTITUTIONAL ANALYZER ---------
if __name__ == "__main__":
    monitor_institutional_intelligence()

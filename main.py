import json
import hmac
import hashlib
import time
import threading
import urllib.request
import urllib.parse
import numpy as np
import websocket
import logging
import requests
import os
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Cấu hình logging chi tiết
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot_errors.log')
    ]
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Lấy cấu hình từ biến môi trường
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
# Cấu hình bot từ biến môi trường (dạng JSON)
bot_config_json = os.getenv('BOT_CONFIGS', '[]')
try:
    BOT_CONFIGS = json.loads(bot_config_json)
except Exception as e:
    logging.error(f"Lỗi phân tích cấu hình BOT_CONFIGS: {e}")
    BOT_CONFIGS = []

API_KEY = BINANCE_API_KEY
API_SECRET = BINANCE_SECRET_KEY

# ========== HÀM GỬI TELEGRAM VÀ XỬ LÝ LỖI ==========
def send_telegram(message, chat_id=None, reply_markup=None):
    """Gửi thông báo qua Telegram với xử lý lỗi chi tiết"""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Cấu hình Telegram Bot Token chưa được thiết lập")
        return
    
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not chat_id:
        logger.warning("Cấu hình Telegram Chat ID chưa được thiết lập")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code != 200:
            error_msg = response.text
            logger.error(f"Lỗi gửi Telegram ({response.status_code}): {error_msg}")
    except Exception as e:
        logger.error(f"Lỗi kết nối Telegram: {str(e)}")

# ========== HÀM TẠO MENU TELEGRAM ==========
def create_menu_keyboard():
    """Tạo menu 3 nút cho Telegram"""
    return {
        "keyboard": [
            [{"text": "📊 Danh sách Bot"}],
            [{"text": "➕ Thêm Bot"}, {"text": "⛔ Dừng Bot"}],
            [{"text": "💰 Số dư tài khoản"}, {"text": "📈 Vị thế đang mở"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False
    }

def create_cancel_keyboard():
    """Tạo bàn phím hủy"""
    return {
        "keyboard": [[{"text": "❌ Hủy bỏ"}]],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_symbols_keyboard():
    """Tạo bàn phím chọn cặp coin"""
    popular_symbols = ["SUIUSDT", "DOGEUSDT", "1000PEPEUSDT", "TRUMPUSDT", "XRPUSDT", "ADAUSDT"]
    keyboard = []
    row = []
    for symbol in popular_symbols:
        row.append({"text": symbol})
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([{"text": "❌ Hủy bỏ"}])
    
    return {
        "keyboard": keyboard,
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_leverage_keyboard():
    """Tạo bàn phím chọn đòn bẩy"""
    leverages = ["10", "20", "30", "50", "75", "100"]
    keyboard = []
    row = []
    for lev in leverages:
        row.append({"text": f"⚖️ {lev}x"})
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([{"text": "❌ Hủy bỏ"}])
    
    return {
        "keyboard": keyboard,
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

# ========== HÀM HỖ TRỢ API BINANCE VỚI XỬ LÝ LỖI CHI TIẾT ==========
def sign(query):
    try:
        return hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    except Exception as e:
        logger.error(f"Lỗi tạo chữ ký: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI SIGN:</b> {str(e)}")
        return ""

def binance_api_request(url, method='GET', params=None, headers=None):
    """Hàm tổng quát cho các yêu cầu API Binance với xử lý lỗi chi tiết"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if method.upper() == 'GET':
                if params:
                    query = urllib.parse.urlencode(params)
                    url = f"{url}?{query}"
                req = urllib.request.Request(url, headers=headers or {})
            else:
                data = urllib.parse.urlencode(params).encode() if params else None
                req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
            
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    return json.loads(response.read().decode())
                else:
                    logger.error(f"Lỗi API ({response.status}): {response.read().decode()}")
                    if response.status == 429:  # Rate limit
                        time.sleep(2 ** attempt)  # Exponential backoff
                    elif response.status >= 500:
                        time.sleep(1)
                    continue
        except urllib.error.HTTPError as e:
            logger.error(f"Lỗi HTTP ({e.code}): {e.reason}")
            if e.code == 429:  # Rate limit
                time.sleep(2 ** attempt)  # Exponential backoff
            elif e.code >= 500:
                time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"Lỗi kết nối API: {str(e)}")
            time.sleep(1)
    
    logger.error(f"Không thể thực hiện yêu cầu API sau {max_retries} lần thử")
    return None

def get_step_size(symbol):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        data = binance_api_request(url)
        if not data:
            return 0.001
            
        for s in data['symbols']:
            if s['symbol'] == symbol.upper():
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        return float(f['stepSize'])
    except Exception as e:
        logger.error(f"Lỗi lấy step size: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI STEP SIZE:</b> {symbol} - {str(e)}")
    return 0.001

def set_leverage(symbol, lev):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "leverage": lev,
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/leverage?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        response = binance_api_request(url, method='POST', headers=headers)
        if response and 'leverage' in response:
            return True
    except Exception as e:
        logger.error(f"Lỗi thiết lập đòn bẩy: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI ĐÒN BẨY:</b> {symbol} - {str(e)}")
    return False

def get_balance():
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        data = binance_api_request(url, headers=headers)
        if not data:
            return 0
            
        for asset in data['assets']:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
    except Exception as e:
        logger.error(f"Lỗi lấy số dư: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI SỐ DƯ:</b> {str(e)}")
    return 0

def place_order(symbol, side, qty):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "MARKET",
            "quantity": qty,
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        return binance_api_request(url, method='POST', headers=headers)
    except Exception as e:
        logger.error(f"Lỗi đặt lệnh: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI ĐẶT LỆNH:</b> {symbol} - {str(e)}")
    return None

def cancel_all_orders(symbol):
    try:
        ts = int(time.time() * 1000)
        params = {"symbol": symbol.upper(), "timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/allOpenOrders?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        binance_api_request(url, method='DELETE', headers=headers)
        return True
    except Exception as e:
        logger.error(f"Lỗi hủy lệnh: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI HỦY LỆNH:</b> {symbol} - {str(e)}")
    return False

def get_current_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'price' in data:
            return float(data['price'])
    except Exception as e:
        logger.error(f"Lỗi lấy giá: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI GIÁ:</b> {symbol} - {str(e)}")
    return 0

def get_positions(symbol=None):
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        if symbol:
            params["symbol"] = symbol.upper()
            
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v2/positionRisk?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        positions = binance_api_request(url, headers=headers)
        if not positions:
            return []
            
        if symbol:
            for pos in positions:
                if pos['symbol'] == symbol.upper():
                    return [pos]
            
        return positions
    except Exception as e:
        logger.error(f"Lỗi lấy vị thế: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI VỊ THẾ:</b> {symbol if symbol else ''} - {str(e)}")
    return []

# ========== TÍNH CHỈ BÁO KỸ THUẬT NÂNG CAO ==========
def calc_rsi(prices, period=14):
    try:
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1 + rs))
    except Exception as e:
        logger.error(f"Lỗi tính RSI: {str(e)}")
        return None

def calc_ema(prices, period):
    """Tính Exponential Moving Average (EMA)"""
    if len(prices) < period:
        return None
        
    # Tính SMA đầu tiên
    sma = np.mean(prices[:period])
    emas = [sma]
    
    # Hệ số nhân
    multiplier = 2 / (period + 1)
    
    # Tính EMA cho các điểm tiếp theo
    for price in prices[period:]:
        ema = (price - emas[-1]) * multiplier + emas[-1]
        emas.append(ema)
    
    return emas[-1]

def calc_macd(prices, fast=12, slow=26, signal=9):
    """Tính MACD và đường tín hiệu"""
    if len(prices) < slow + signal:
        return None, None
        
    ema_fast = calc_ema(prices, fast)
    ema_slow = calc_ema(prices, slow)
    
    if ema_fast is None or ema_slow is None:
        return None, None
        
    macd_line = ema_fast - ema_slow
    
    # Tính đường tín hiệu (EMA của MACD)
    # Lấy giá trị MACD cho signal_period
    if len(prices) >= slow + signal:
        # Chỉ lấy các giá trị MACD cần thiết
        macd_values = []
        for i in range(len(prices) - slow + 1):
            fast_ema = calc_ema(prices[i:i+fast], fast)
            slow_ema = calc_ema(prices[i:i+slow], slow)
            if fast_ema is not None and slow_ema is not None:
                macd_values.append(fast_ema - slow_ema)
        
        # Tính EMA của MACD cho đường tín hiệu
        macd_signal = calc_ema(macd_values[-signal:], signal)
    else:
        macd_signal = None
    
    return macd_line, macd_signal

def calc_bollinger_bands(prices, period=20, std_dev=2):
    """Tính Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
        
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band

def calc_stochastic(prices, lows, highs, period=14, k_period=3):
    """Tính Stochastic Oscillator"""
    if len(prices) < period + k_period or len(lows) < period or len(highs) < period:
        return None, None
        
    current_close = prices[-1]
    low_min = min(lows[-period:])
    high_max = max(highs[-period:])
    
    if high_max - low_min == 0:
        return None, None
        
    k = 100 * (current_close - low_min) / (high_max - low_min)
    
    # Tính %D (signal line)
    d_values = [k]
    for i in range(2, k_period+1):
        if len(prices) < period + i:
            continue
        prev_close = prices[-i]
        prev_low = min(lows[-period-i:-i] or [0])
        prev_high = max(highs[-period-i:-i] or [1])
        if prev_high - prev_low == 0:
            continue
        d_val = 100 * (prev_close - prev_low) / (prev_high - prev_low)
        d_values.append(d_val)
    
    d = np.mean(d_values) if d_values else None
    
    return k, d

def calc_vwma(prices, volumes, period=20):
    """Tính Volume Weighted Moving Average (VWMA)"""
    if len(prices) < period or len(volumes) < period:
        return None
        
    prices_slice = prices[-period:]
    volumes_slice = volumes[-period:]
    total_volume = sum(volumes_slice)
    
    if total_volume == 0:
        return None
        
    return sum(p * v for p, v in zip(prices_slice, volumes_slice)) / total_volume

def calc_atr(highs, lows, closes, period=14):
    """Tính Average True Range (ATR)"""
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return None
        
    tr = []
    for i in range(1, len(closes)):
        h = highs[i]
        l = lows[i]
        pc = closes[i-1]
        tr.append(max(h-l, abs(h-pc), abs(l-pc)))
    
    return np.mean(tr[-period:]) if tr else None

# ========== QUẢN LÝ WEBSOCKET HIỆU QUẢ VỚI KIỂM SOÁT LỖI ==========
class WebSocketManager:
    def __init__(self):
        self.connections = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
    def add_symbol(self, symbol, callback):
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self.connections:
                self._create_connection(symbol, callback)
                
    def _create_connection(self, symbol, callback):
        if self._stop_event.is_set():
            return
            
        # Sử dụng kênh kline 1 phút để lấy thêm dữ liệu
        stream = f"{symbol.lower()}@kline_1m"
        url = f"wss://fstream.binance.com/ws/{stream}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                kline = data.get('k', {})
                if kline and kline.get('x'):  # Chỉ xử lý khi nến đã đóng
                    close = float(kline['c'])
                    volume = float(kline['v'])
                    high = float(kline['h'])
                    low = float(kline['l'])
                    self.executor.submit(callback, close, volume, high, low)
            except Exception as e:
                logger.error(f"Lỗi xử lý tin nhắn WebSocket {symbol}: {str(e)}")
                
        def on_error(ws, error):
            logger.error(f"Lỗi WebSocket {symbol}: {str(error)}")
            if not self._stop_event.is_set():
                time.sleep(5)
                self._reconnect(symbol, callback)
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket đóng {symbol}: {close_status_code} - {close_msg}")
            if not self._stop_event.is_set() and symbol in self.connections:
                time.sleep(5)
                self._reconnect(symbol, callback)
                
        ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        
        self.connections[symbol] = {
            'ws': ws,
            'thread': thread,
            'callback': callback
        }
        logger.info(f"WebSocket bắt đầu cho {symbol} (kline_1m)")
        
    def _reconnect(self, symbol, callback):
        logger.info(f"Kết nối lại WebSocket cho {symbol}")
        self.remove_symbol(symbol)
        self._create_connection(symbol, callback)
        
    def remove_symbol(self, symbol):
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.connections:
                try:
                    self.connections[symbol]['ws'].close()
                except Exception as e:
                    logger.error(f"Lỗi đóng WebSocket {symbol}: {str(e)}")
                del self.connections[symbol]
                logger.info(f"WebSocket đã xóa cho {symbol}")
                
    def stop(self):
        self._stop_event.set()
        for symbol in list(self.connections.keys()):
            self.remove_symbol(symbol)

# ========== BOT CHÍNH VỚI CHIẾN LƯỢC GIAO DỊCH NÂNG CAO ==========
class IndicatorBot:
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager=None):
        self.symbol = symbol.upper()
        self.lev = lev
        self.percent = percent
        self.tp = tp
        self.sl = sl
        
        self.ws_manager = ws_manager
        self.status = "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.prices = []
        self.volumes = []
        self.highs = []
        self.lows = []
        self.closes = []
        self._stop = False
        self.position_open = False
        self.last_trade_time = 0
        self.position_check_interval = 60
        self.last_position_check = 0
        self.last_error_log_time = 0
        self.last_close_time = 0
        self.cooldown_period = 60
        self.max_position_attempts = 3
        self.position_attempt_count = 0
        self.dynamic_sl = sl  # SL động có thể thay đổi
        self.best_profit = 0  # Theo dõi lợi nhuận tốt nhất để trailing stop
        
        # Đăng ký với WebSocket Manager
        self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        
        # Tải dữ liệu lịch sử
        self._fetch_initial_data()
        
        # Bắt đầu thread chính
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.log(f"🟢 Bot khởi động cho {self.symbol}")

    def _fetch_initial_data(self, limit=200):
        """Tải dữ liệu nến lịch sử khi khởi động bot"""
        try:
            # Lấy dữ liệu nến 1 phút
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                "symbol": self.symbol,
                "interval": "1m",
                "limit": limit
            }
            data = binance_api_request(url, params=params)
            if not data:
                return

            # Xử lý dữ liệu nến
            for candle in data:
                self.closes.append(float(candle[4]))
                self.highs.append(float(candle[2]))
                self.lows.append(float(candle[3]))
                self.volumes.append(float(candle[5]))
                self.prices.append(float(candle[4]))  # Giá đóng cửa

            # Log thông báo
            self.log(f"Đã tải {len(data)} nến lịch sử")

        except Exception as e:
            self.log(f"Lỗi khi tải dữ liệu lịch sử: {str(e)}")

    def log(self, message):
        """Ghi log và gửi qua Telegram"""
        logger.info(f"[{self.symbol}] {message}")
        send_telegram(f"<b>{self.symbol}</b>: {message}")

    def log_signal_conditions(self, signal_type, conditions):
        """Ghi log chi tiết điều kiện tín hiệu"""
        message = f"📊 {self.symbol} TÍN HIỆU {signal_type}:\n"
        for i, cond in enumerate(conditions, 1):
            message += f"ĐK {i}: {'✅' if cond else '❌'}\n"
        self.log(message)

    def _handle_price_update(self, close, volume, high, low):
        if self._stop: 
            return
            
        self.prices.append(close)
        self.volumes.append(volume)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        # Giới hạn lịch sử để tiết kiệm bộ nhớ
        max_history = 200
        if len(self.prices) > max_history:
            self.prices = self.prices[-max_history:]
        if len(self.volumes) > max_history:
            self.volumes = self.volumes[-max_history:]
        if len(self.highs) > max_history:
            self.highs = self.highs[-max_history:]
        if len(self.lows) > max_history:
            self.lows = self.lows[-max_history:]
        if len(self.closes) > max_history:
            self.closes = self.closes[-max_history:]

    def _run(self):
        """Luồng chính quản lý bot với kiểm soát lỗi chặt chẽ"""
        while not self._stop:
            try:
                current_time = time.time()
                
                # Kiểm tra trạng thái vị thế định kỳ
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                
                # Xử lý logic giao dịch
                if not self.position_open and self.status == "waiting":
                    # Kiểm tra thời gian chờ sau khi đóng lệnh
                    if current_time - self.last_close_time < self.cooldown_period:
                        time.sleep(1)
                        continue
                    
                    signal = self.get_signal()
                    
                    if signal and current_time - self.last_trade_time > 60:
                        self.open_position(signal)
                        self.last_trade_time = current_time
                
                # Kiểm tra TP/SL cho vị thế đang mở
                if self.position_open and self.status == "open":
                    self.check_tp_sl()
                
                time.sleep(1)
                
            except Exception as e:
                if time.time() - self.last_error_log_time > 10:
                    self.log(f"Lỗi hệ thống: {str(e)}")
                    self.last_error_log_time = time.time()
                time.sleep(5)

    def stop(self):
        self._stop = True
        self.ws_manager.remove_symbol(self.symbol)
        try:
            cancel_all_orders(self.symbol)
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"Lỗi hủy lệnh: {str(e)}")
                self.last_error_log_time = time.time()
        self.log(f"🔴 Bot dừng cho {self.symbol}")

    def check_position_status(self):
        """Kiểm tra trạng thái vị thế từ API Binance với kiểm soát lỗi"""
        try:
            positions = get_positions(self.symbol)
            
            if not positions or len(positions) == 0:
                self.position_open = False
                self.status = "waiting"
                self.side = ""
                self.qty = 0
                self.entry = 0
                return
            
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    position_amt = float(pos.get('positionAmt', 0))
                    
                    if abs(position_amt) > 0:
                        self.position_open = True
                        self.status = "open"
                        self.side = "BUY" if position_amt > 0 else "SELL"
                        self.qty = position_amt
                        self.entry = float(pos.get('entryPrice', 0))
                        return
            
            self.position_open = False
            self.status = "waiting"
            self.side = ""
            self.qty = 0
            self.entry = 0
            
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"Lỗi kiểm tra vị thế: {str(e)}")
                self.last_error_log_time = time.time()

    def calc_trend_strength(self, period=20):
        """Tính sức mạnh xu hướng (độ dốc) trong N phiên"""
        if len(self.prices) < period:
            return 0
        x = np.arange(period)
        y = np.array(self.prices[-period:])
        slope = np.polyfit(x, y, 1)[0]
        return slope * 100  # Trả về % thay đổi/phiên

    def check_bullish_divergence(self):
        """Phát hiện phân kỳ tăng giữa giá và RSI"""
        if len(self.prices) < 10 or len(self.closes) < 10:
            return False
            
        # Tìm đáy gần nhất
        min_idx = np.argmin(self.closes[-10:])
        min_price = self.closes[-10:][min_idx]
        
        # So sánh với đáy trước đó
        prev_low = min(self.closes[-20:-10]) if len(self.closes) >= 20 else min_price
        
        # Tính RSI tương ứng
        rsi_current = calc_rsi(self.prices[-10:], 14)
        rsi_prev = calc_rsi(self.prices[-20:-10], 14) if len(self.prices) >= 20 else rsi_current
        
        # Phân kỳ tăng: Giá tạo đáy thấp hơn nhưng RSI cao hơn
        return min_price < prev_low and rsi_current > rsi_prev

    def check_bearish_divergence(self):
        """Phát hiện phân kỳ giảm giữa giá và RSI"""
        if len(self.prices) < 10 or len(self.closes) < 10:
            return False
            
        # Tìm đỉnh gần nhất
        max_idx = np.argmax(self.closes[-10:])
        max_price = self.closes[-10:][max_idx]
        
        # So sánh với đỉnh trước đó
        prev_high = max(self.closes[-20:-10]) if len(self.closes) >= 20 else max_price
        
        # Tính RSI tương ứng
        rsi_current = calc_rsi(self.prices[-10:], 14)
        rsi_prev = calc_rsi(self.prices[-20:-10], 14) if len(self.prices) >= 20 else rsi_current
        
        # Phân kỳ giảm: Giá tạo đỉnh cao hơn nhưng RSI thấp hơn
        return max_price > prev_high and rsi_current < rsi_prev

    def get_signal(self):
        """Tạo tín hiệu với bộ lọc 3 lớp và xác nhận khối lượng"""
        # Kiểm tra đủ dữ liệu
        min_data = max(100, 50)  # Chỉ cần 50 nến cho chiến lược mới
        if len(self.prices) < min_data:
            return None
        
        try:
            # === LỚP 1: XU HƯỚNG CHÍNH ===
            # EMA 50 và EMA 200 để xác định xu hướng dài hạn
            ema50 = calc_ema(self.prices, 50)
            ema200 = calc_ema(self.prices, 200)
            
            if ema50 is None or ema200 is None:
                return None
                
            trend_direction = 1 if ema50 > ema200 else -1  # 1: uptrend, -1: downtrend
            
            # Tính sức mạnh xu hướng
            trend_strength = self.calc_trend_strength(20)
            
            # Phân loại xu hướng
            strong_uptrend = trend_direction == 1 and trend_strength > 0.2
            strong_downtrend = trend_direction == -1 and trend_strength < -0.2
            neutral_market = abs(trend_strength) <= 0.2
            
            # === LỚP 2: CHỈ BÁO ĐỘNG LƯỢNG ===
            # RSI với vùng quá mua/quá bán điều chỉnh theo xu hướng
            rsi = calc_rsi(self.prices, 14)
            if rsi is None:
                return None
                
            # Điều chỉnh ngưỡng RSI cho tín hiệu mua
            rsi_buy_threshold = 40 if trend_direction == 1 else 30  # Giảm ngưỡng mua trong uptrend

            # Điều chỉnh ngưỡng RSI cho tín hiệu bán
            rsi_sell_threshold = 65 if trend_direction == -1 else 70  # Tăng ngưỡng bán trong downtrend
            
            # MACD với tín hiệu phân kỳ
            macd_line, macd_signal = calc_macd(self.prices, 12, 26, 9)
            
            # === LỚP 3: MÔ HÌNH GIÁ ===
            # Xác nhận breakout với Bollinger Bands
            upper_band, middle_band, lower_band = calc_bollinger_bands(self.prices, 20, 2)
            current_price = self.prices[-1]
            
            # === XÁC NHẬN KHỐI LƯỢNG ===
            # So sánh khối lượng hiện tại với trung bình
            if len(self.volumes) < 20:
                return None
                
            current_volume = self.volumes[-1]
            avg_volume = np.mean(self.volumes[-20:])
            volume_ok = current_volume > avg_volume * 1.3  # Khối lượng > 130% trung bình
            
            # === TÍN HIỆU MUA (BUY) ===
            buy_conditions = [
                # Điều kiện mua trong uptrend mạnh
                strong_uptrend and rsi < rsi_buy_threshold and current_price < middle_band,
                
                # Điều kiện mua trong downtrend (phản đảo)
                strong_downtrend and rsi < 30 and macd_line > macd_signal,
                
                # Điều kiện mua trong thị trường trung lập
                neutral_market and rsi < 35 and current_price < lower_band
            ]
            
            # === TÍN HIỆU BÁN (SELL) ===
            sell_conditions = [
                # Điều kiện bán trong downtrend mạnh
                strong_downtrend and rsi > rsi_sell_threshold and current_price > middle_band,
                
                # Điều kiện bán trong uptrend (phản đảo)
                strong_uptrend and rsi > 70 and macd_line < macd_signal,
                
                # Điều kiện bán trong thị trường trung lập
                neutral_market and rsi > 65 and current_price > upper_band
            ]
            
            # === QUYẾT ĐỊNH TÍN HIỆU ===
            if any(buy_conditions) and volume_ok:
                # Kiểm tra thêm phân kỳ tăng
                if self.check_bullish_divergence():
                    self.log_signal_conditions("MUA", buy_conditions)
                    return "BUY"
                    
            if any(sell_conditions) and volume_ok:
                # Kiểm tra thêm phân kỳ giảm
                if self.check_bearish_divergence():
                    self.log_signal_conditions("BÁN", sell_conditions)
                    return "SELL"
                    
            return None
            
        except Exception as e:
            self.log(f"Lỗi tạo tín hiệu: {str(e)}")
            return None

    def check_tp_sl(self):
        """Quản lý TP/SL thích ứng với biến động thị trường"""
        if not self.position_open:
            return
            
        try:
            current_price = self.prices[-1] if self.prices else get_current_price(self.symbol)
            if current_price <= 0:
                return
                
            # Tính % thay đổi giá
            price_change_pct = ((current_price - self.entry) / self.entry) * 100
            if self.side == "SELL":
                price_change_pct = -price_change_pct
                
            # Tính ATR để xác định biến động
            atr = calc_atr(self.highs, self.lows, self.closes, 14)
            if atr:
                # Điều chỉnh SL động dựa trên ATR
                atr_pct = (atr / self.entry) * 100
                dynamic_sl = max(1.0, min(self.sl, 2.0 * atr_pct))
            else:
                dynamic_sl = self.sl
                
            # Điều chỉnh SL theo hướng có lợi (Trailing Stop)
            if price_change_pct > 0:
                # Dịch chuyển SL lên khi có lợi nhuận
                new_sl_level = price_change_pct * 0.7  # Giữ 70% lợi nhuận
                if new_sl_level > self.best_profit:
                    self.best_profit = new_sl_level
                    self.dynamic_sl = max(self.dynamic_sl, -new_sl_level)
            
            # Kiểm tra TP/SL
            if price_change_pct >= self.tp:
                self.close_position(f"✅ Đạt TP {self.tp}%")
            elif price_change_pct <= -dynamic_sl:
                self.close_position(f"❌ Đạt SL {dynamic_sl:.2f}%")
                
        except Exception as e:
            self.log(f"Lỗi kiểm tra TP/SL: {str(e)}")

    def close_position(self, reason=""):
        """Đóng vị thế với số lượng chính xác, kiểm tra kết quả từ Binance"""
        try:
            # Kiểm tra lại trạng thái trước khi đóng
            self.check_position_status()
            if not self.position_open:
                return
                
            # Lấy thông tin vị thế MỚI NHẤT từ API
            positions = get_positions(self.symbol)
            if not positions:
                return
                
            # Tìm vị thế chính xác
            current_qty = 0
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    current_qty = float(pos.get('positionAmt', 0))
                    break
                    
            if current_qty == 0:
                self.position_open = False
                self.status = "waiting"
                return
                
            # Xác định hướng đóng
            close_side = "SELL" if current_qty > 0 else "BUY"
            close_qty = abs(current_qty)
            
            # Làm tròn số lượng CHÍNH XÁC theo step size của Binance
            step = get_step_size(self.symbol)
            if step > 0:
                # Tính toán chính xác số bước
                steps = close_qty / step
                # Làm tròn xuống (floor) để đảm bảo không vượt quá số lượng hiện có
                close_qty = math.floor(steps) * step
            
            # Đảm bảo số lượng tối thiểu
            min_qty = step
            if close_qty < min_qty:
                close_qty = abs(current_qty)  # Dùng số lượng gốc nếu quá nhỏ
            
            # Đặt lệnh đóng với số lượng CHÍNH XÁC
            res = place_order(self.symbol, close_side, close_qty)
            if res:
                executed_qty = float(res.get('executedQty', 0))
                
                # Kiểm tra xem đã đóng hết chưa
                if executed_qty >= abs(current_qty) * 0.99:  # Cho phép sai số 1%
                    # Thông báo thành công
                    price = float(res.get('avgPrice', 0))
                    message = (
                        f"⛔ <b>ĐÃ ĐÓNG VỊ THẾ {self.symbol}</b>\n"
                        f"📌 Lý do: {reason}\n"
                        f"🏷️ Giá ra: {price:.4f}\n"
                        f"📊 Khối lượng: {executed_qty}\n"
                        f"💵 Giá trị: {executed_qty * price:.2f} USDT"
                    )
                    self.log(message)
                    
                    # Cập nhật trạng thái NGAY LẬP TỨC
                    self.status = "waiting"
                    self.side = ""
                    self.qty = 0
                    self.entry = 0
                    self.position_open = False
                    self.last_trade_time = time.time()
                    self.last_close_time = time.time()
                else:
                    # Xử lý trường hợp đóng không hết
                    remaining = abs(current_qty) - executed_qty
                    self.log(f"⚠️ Đóng chưa hết! Còn lại: {remaining}, thử đóng phần còn lại")
                    
                    # Thử đóng phần còn lại
                    retry_qty = remaining
                    if step > 0:
                        retry_steps = retry_qty / step
                        retry_qty = math.floor(retry_steps) * step
                    
                    if retry_qty >= min_qty:
                        retry_res = place_order(self.symbol, close_side, retry_qty)
                        if retry_res:
                            total_executed = executed_qty + float(retry_res.get('executedQty', 0))
                            self.log(f"✅ Đã đóng thêm: {total_executed - executed_qty}, tổng: {total_executed}")
                            
                            # Cập nhật trạng thái nếu đóng thành công
                            if total_executed >= abs(current_qty) * 0.99:
                                self.status = "waiting"
                                self.side = ""
                                self.qty = 0
                                self.entry = 0
                                self.position_open = False
                        else:
                            self.log("❌ Lỗi khi đóng phần còn lại")
                    else:
                        self.log(f"⚠️ Số lượng còn lại quá nhỏ ({retry_qty}), không thể đóng")
            else:
                self.log(f"❌ Lỗi khi đặt lệnh đóng")
        except Exception as e:
            self.log(f"❌ Lỗi nghiêm trọng khi đóng lệnh: {str(e)}")

    def open_position(self, side):
        # Kiểm tra lại trạng thái trước khi vào lệnh
        self.check_position_status()
        
        if self.position_open:
            self.log(f"⚠️ Đã có vị thế mở, không vào lệnh mới")
            return
            
        try:
            # Hủy lệnh tồn đọng
            cancel_all_orders(self.symbol)
            
            # Đặt đòn bẩy
            if not set_leverage(self.symbol, self.lev):
                self.log(f"Không thể đặt đòn bẩy {self.lev}")
                return
            
            # Tính toán khối lượng
            balance = get_balance()
            if balance <= 0:
                self.log(f"Không đủ số dư USDT")
                return
            
            # Giới hạn % số dư sử dụng
            if self.percent > 100:
                self.percent = 100
            elif self.percent < 1:
                self.percent = 1
                
            usdt_amount = balance * (self.percent / 100)
            price = get_current_price(self.symbol)
            if price <= 0:
                self.log(f"Lỗi lấy giá")
                return
                
            step = get_step_size(self.symbol)
            if step <= 0:
                step = 0.001
            
            # Tính số lượng với đòn bẩy
            qty = (usdt_amount * self.lev) / price
            
            # Làm tròn số lượng theo step size (luôn làm tròn xuống)
            if step > 0:
                steps = qty / step
                qty = math.floor(steps) * step  # Luôn làm tròn xuống
            
            qty = max(qty, 0)
            qty = round(qty, 8)
            
            min_qty = step
            
            if qty < min_qty:
                self.log(f"⚠️ Số lượng quá nhỏ ({qty}), không đặt lệnh")
                return
                
            # Giới hạn số lần thử
            self.position_attempt_count += 1
            if self.position_attempt_count > self.max_position_attempts:
                self.log(f"⚠️ Đã đạt giới hạn số lần thử mở lệnh ({self.max_position_attempts})")
                self.position_attempt_count = 0
                return
                
            # Đặt lệnh
            res = place_order(self.symbol, side, qty)
            if not res:
                self.log(f"Lỗi khi đặt lệnh")
                return
                
            executed_qty = float(res.get('executedQty', 0))
            if executed_qty <= 0:
                self.log(f"Lệnh không khớp, số lượng thực thi: {executed_qty}")
                return

            # Cập nhật trạng thái
            self.entry = float(res.get('avgPrice', price))
            self.side = side
            self.qty = executed_qty if side == "BUY" else -executed_qty
            self.status = "open"
            self.position_open = True
            self.position_attempt_count = 0  # Reset số lần thử
            self.best_profit = 0  # Reset lợi nhuận tốt nhất
            
            # Thông báo qua Telegram
            message = (
                f"✅ <b>ĐÃ MỞ VỊ THẾ {self.symbol}</b>\n"
                f"📌 Hướng: {side}\n"
                f"🏷️ Giá vào: {self.entry:.4f}\n"
                f"📊 Khối lượng: {executed_qty}\n"
                f"💵 Giá trị: {executed_qty * self.entry:.2f} USDT\n"
                f"⚖️ Đòn bẩy: {self.lev}x\n"
                f"🎯 TP: {self.tp}% | 🛡️ SL: {self.dynamic_sl:.2f}%"
            )
            self.log(message)

        except Exception as e:
            self.position_open = False
            self.log(f"❌ Lỗi khi vào lệnh: {str(e)}")

# ========== QUẢN LÝ BOT CHẠY NỀN VÀ TƯƠNG TÁC TELEGRAM ==========
class BotManager:
    def __init__(self):
        self.ws_manager = WebSocketManager()
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.user_states = {}  # Lưu trạng thái người dùng
        self.admin_chat_id = TELEGRAM_CHAT_ID
        
        self.log("🟢 HỆ THỐNG BOT ĐÃ KHỞI ĐỘNG")
        
        # Bắt đầu thread kiểm tra trạng thái
        self.status_thread = threading.Thread(target=self._status_monitor, daemon=True)
        self.status_thread.start()
        
        # Bắt đầu thread lắng nghe Telegram
        self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
        self.telegram_thread.start()
        
        # Gửi menu chính khi khởi động
        if self.admin_chat_id:
            self.send_main_menu(self.admin_chat_id)

    def log(self, message):
        """Ghi log hệ thống và gửi Telegram"""
        logger.info(f"[SYSTEM] {message}")
        send_telegram(f"<b>SYSTEM</b>: {message}")

    def send_main_menu(self, chat_id):
        """Gửi menu chính cho người dùng"""
        welcome = (
            "🤖 <b>BOT GIAO DỊCH FUTURES BINANCE</b>\n\n"
            "Chọn một trong các tùy chọn bên dưới:"
        )
        send_telegram(welcome, chat_id, create_menu_keyboard())

    def add_bot(self, symbol, lev, percent, tp, sl, indicator_config=None):
        symbol = symbol.upper()
        if symbol in self.bots:
            self.log(f"⚠️ Đã có bot cho {symbol}")
            return False
            
        # Kiểm tra API key
        if not API_KEY or not API_SECRET:
            self.log("❌ Chưa cấu hình API Key và Secret Key!")
            return False
            
        try:
            # Kiểm tra kết nối API
            price = get_current_price(symbol)
            if price <= 0:
                self.log(f"❌ Không thể lấy giá cho {symbol}")
                return False
            
            # Kiểm tra vị thế hiện tại
            positions = get_positions(symbol)
            if positions and any(float(pos.get('positionAmt', 0)) != 0 for pos in positions):
                self.log(f"⚠️ Đã có vị thế mở cho {symbol} trên Binance")
                return False
            
            # Tạo bot mới
            bot = IndicatorBot(
                symbol, lev, percent, tp, sl, self.ws_manager
            )
            self.bots[symbol] = bot
            self.log(f"✅ Đã thêm bot: {symbol} | ĐB: {lev}x | %: {percent} | TP/SL: {tp}%/{sl}%")
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot {symbol}: {str(e)}")
            return False

    def stop_bot(self, symbol):
        symbol = symbol.upper()
        bot = self.bots.get(symbol)
        if bot:
            bot.stop()
            if bot.status == "open":
                bot.close_position("⛔ Dừng bot thủ công")
            self.log(f"⛔ Đã dừng bot cho {symbol}")
            del self.bots[symbol]
            return True
        return False

    def stop_all(self):
        self.log("⛔ Đang dừng tất cả bot...")
        for symbol in list(self.bots.keys()):
            self.stop_bot(symbol)
        self.ws_manager.stop()
        self.running = False
        self.log("🔴 Hệ thống đã dừng")

    def _status_monitor(self):
        """Kiểm tra và báo cáo trạng thái định kỳ"""
        while self.running:
            try:
                # Tính thời gian hoạt động
                uptime = time.time() - self.start_time
                hours, rem = divmod(uptime, 3600)
                minutes, seconds = divmod(rem, 60)
                uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                
                # Báo cáo số bot đang chạy
                active_bots = [s for s, b in self.bots.items() if not b._stop]
                
                # Báo cáo số dư tài khoản
                balance = get_balance()
                
                # Tạo báo cáo
                status_msg = (
                    f"📊 <b>BÁO CÁO HỆ THỐNG</b>\n"
                    f"⏱ Thời gian hoạt động: {uptime_str}\n"
                    f"🤖 Số bot đang chạy: {len(active_bots)}\n"
                    f"📈 Bot hoạt động: {', '.join(active_bots) if active_bots else 'Không có'}\n"
                    f"💰 Số dư khả dụng: {balance:.2f} USDT"
                )
                send_telegram(status_msg)
                
                # Log chi tiết
                for symbol, bot in self.bots.items():
                    if bot.status == "open":
                        status_msg = (
                            f"🔹 <b>{symbol}</b>\n"
                            f"📌 Hướng: {bot.side}\n"
                            f"🏷️ Giá vào: {bot.entry:.4f}\n"
                            f"📊 Khối lượng: {abs(bot.qty)}\n"
                            f"⚖️ Đòn bẩy: {bot.lev}x\n"
                            f"🎯 TP: {bot.tp}% | 🛡️ SL: {bot.dynamic_sl:.2f}%"
                        )
                        send_telegram(status_msg)
                
            except Exception as e:
                logger.error(f"Lỗi báo cáo trạng thái: {str(e)}")
            
            # Kiểm tra mỗi 6 giờ
            time.sleep(6 * 3600)

    def _telegram_listener(self):
        """Lắng nghe và xử lý tin nhắn từ Telegram"""
        last_update_id = 0
        
        while self.running:
            try:
                # Lấy tin nhắn mới
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={last_update_id+1}&timeout=30"
                response = requests.get(url, timeout=35)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        for update in data['result']:
                            update_id = update['update_id']
                            message = update.get('message', {})
                            chat_id = str(message.get('chat', {}).get('id'))
                            text = message.get('text', '').strip()
                            
                            # Chỉ xử lý tin nhắn từ admin
                            if chat_id != self.admin_chat_id:
                                continue
                            
                            # Cập nhật ID tin nhắn cuối
                            if update_id > last_update_id:
                                last_update_id = update_id
                            
                            # Xử lý tin nhắn
                            self._handle_telegram_message(chat_id, text)
                elif response.status_code == 409:
                    # Xử lý xung đột - chỉ có một instance của bot có thể lắng nghe
                    logger.error("Lỗi xung đột: Chỉ một instance bot có thể lắng nghe Telegram")
                    break
                
            except Exception as e:
                logger.error(f"Lỗi Telegram listener: {str(e)}")
                time.sleep(5)

    def _handle_telegram_message(self, chat_id, text):
        """Xử lý tin nhắn từ người dùng"""
        # Lưu trạng thái người dùng
        user_state = self.user_states.get(chat_id, {})
        current_step = user_state.get('step')
        
        # Xử lý theo bước hiện tại
        if current_step == 'waiting_symbol':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            else:
                symbol = text.upper()
                self.user_states[chat_id] = {
                    'step': 'waiting_leverage',
                    'symbol': symbol
                }
                send_telegram(f"Chọn đòn bẩy cho {symbol}:", chat_id, create_leverage_keyboard())
        
        elif current_step == 'waiting_leverage':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            elif 'x' in text:
                leverage = int(text.replace('⚖️', '').replace('x', '').strip())
                user_state['leverage'] = leverage
                user_state['step'] = 'waiting_percent'
                send_telegram(
                    f"📌 Cặp: {user_state['symbol']}\n⚖️ Đòn bẩy: {leverage}x\n\nNhập % số dư muốn sử dụng (1-100):",
                    chat_id,
                    create_cancel_keyboard()
                )
        
        elif current_step == 'waiting_percent':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            else:
                try:
                    percent = float(text)
                    if 1 <= percent <= 100:
                        user_state['percent'] = percent
                        user_state['step'] = 'waiting_tp'
                        send_telegram(
                            f"📌 Cặp: {user_state['symbol']}\n⚖️ ĐB: {user_state['leverage']}x\n📊 %: {percent}%\n\nNhập % Take Profit (ví dụ: 10):",
                            chat_id,
                            create_cancel_keyboard()
                        )
                    else:
                        send_telegram("⚠️ Vui lòng nhập % từ 1-100", chat_id)
                except:
                    send_telegram("⚠️ Giá trị không hợp lệ, vui lòng nhập số", chat_id)
        
        elif current_step == 'waiting_tp':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            else:
                try:
                    tp = float(text)
                    if tp > 0:
                        user_state['tp'] = tp
                        user_state['step'] = 'waiting_sl'
                        send_telegram(
                            f"📌 Cặp: {user_state['symbol']}\n⚖️ ĐB: {user_state['leverage']}x\n📊 %: {user_state['percent']}%\n🎯 TP: {tp}%\n\nNhập % Stop Loss (ví dụ: 5):",
                            chat_id,
                            create_cancel_keyboard()
                        )
                    else:
                        send_telegram("⚠️ TP phải lớn hơn 0", chat_id)
                except:
                    send_telegram("⚠️ Giá trị không hợp lệ, vui lòng nhập số", chat_id)
        
        elif current_step == 'waiting_sl':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            else:
                try:
                    sl = float(text)
                    if sl > 0:
                        # Thêm bot
                        symbol = user_state['symbol']
                        leverage = user_state['leverage']
                        percent = user_state['percent']
                        tp = user_state['tp']
                        
                        if self.add_bot(symbol, leverage, percent, tp, sl):
                            send_telegram(
                                f"✅ <b>ĐÃ THÊM BOT THÀNH CÔNG</b>\n\n"
                                f"📌 Cặp: {symbol}\n"
                                f"⚖️ Đòn bẩy: {leverage}x\n"
                                f"📊 % Số dư: {percent}%\n"
                                f"🎯 TP: {tp}%\n"
                                f"🛡️ SL: {sl}%",
                                chat_id,
                                create_menu_keyboard()
                            )
                        else:
                            send_telegram("❌ Không thể thêm bot, vui lòng kiểm tra log", chat_id, create_menu_keyboard())
                        
                        # Reset trạng thái
                        self.user_states[chat_id] = {}
                    else:
                        send_telegram("⚠️ SL phải lớn hơn 0", chat_id)
                except:
                    send_telegram("⚠️ Giá trị không hợp lệ, vui lòng nhập số", chat_id)
        
        # Xử lý các lệnh chính
        elif text == "📊 Danh sách Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id)
            else:
                message = "🤖 <b>DANH SÁCH BOT ĐANG CHẠY</b>\n\n"
                for symbol, bot in self.bots.items():
                    status = "🟢 Mở" if bot.status == "open" else "🟡 Chờ"
                    message += f"🔹 {symbol} | {status} | {bot.side}\n"
                send_telegram(message, chat_id)
        
        elif text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_symbol'}
            send_telegram("Chọn cặp coin:", chat_id, create_symbols_keyboard())
        
        elif text == "⛔ Dừng Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id)
            else:
                message = "⛔ <b>CHỌN BOT ĐỂ DỪNG</b>\n\n"
                keyboard = []
                row = []
                
                for i, symbol in enumerate(self.bots.keys()):
                    message += f"🔹 {symbol}\n"
                    row.append({"text": f"⛔ {symbol}"})
                    if len(row) == 2 or i == len(self.bots) - 1:
                        keyboard.append(row)
                        row = []
                
                keyboard.append([{"text": "❌ Hủy bỏ"}])
                
                send_telegram(
                    message, 
                    chat_id, 
                    {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True}
                )
        
        elif text.startswith("⛔ "):
            symbol = text.replace("⛔ ", "").strip().upper()
            if symbol in self.bots:
                self.stop_bot(symbol)
                send_telegram(f"⛔ Đã gửi lệnh dừng bot {symbol}", chat_id, create_menu_keyboard())
            else:
                send_telegram(f"⚠️ Không tìm thấy bot {symbol}", chat_id, create_menu_keyboard())
        
        elif text == "💰 Số dư tài khoản":
            try:
                balance = get_balance()
                send_telegram(f"💰 <b>SỐ DƯ KHẢ DỤNG</b>: {balance:.2f} USDT", chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Lỗi lấy số dư: {str(e)}", chat_id)
        
        elif text == "📈 Vị thế đang mở":
            try:
                positions = get_positions()
                if not positions:
                    send_telegram("📭 Không có vị thế nào đang mở", chat_id)
                    return
                
                message = "📈 <b>VỊ THẾ ĐANG MỞ</b>\n\n"
                for pos in positions:
                    position_amt = float(pos.get('positionAmt', 0))
                    if position_amt != 0:
                        symbol = pos.get('symbol', 'UNKNOWN')
                        entry = float(pos.get('entryPrice', 0))
                        side = "LONG" if position_amt > 0 else "SHORT"
                        pnl = float(pos.get('unRealizedProfit', 0))
                        
                        message += (
                            f"🔹 {symbol} | {side}\n"
                            f"📊 Khối lượng: {abs(position_amt):.4f}\n"
                            f"🏷️ Giá vào: {entry:.4f}\n"
                            f"💰 PnL: {pnl:.2f} USDT\n\n"
                        )
                
                send_telegram(message, chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Lỗi lấy vị thế: {str(e)}", chat_id)
        
        # Gửi lại menu nếu không có lệnh phù hợp
        elif text:
            self.send_main_menu(chat_id)

# ========== HÀM KHỞI CHẠY CHÍNH ==========
def main():
    # Khởi tạo hệ thống
    manager = BotManager()
    
    # Thêm các bot từ cấu hình
    if BOT_CONFIGS:
        for config in BOT_CONFIGS:
            # Xử lý cả cấu hình cũ và mới
            if isinstance(config, list):
                # Cấu hình cũ: [symbol, lev, percent, tp, sl]
                manager.add_bot(*config[:5])
            elif isinstance(config, dict):
                # Cấu hình mới với chỉ báo
                manager.add_bot(
                    config['symbol'],
                    config['lev'],
                    config['percent'],
                    config['tp'],
                    config['sl']
                )
    else:
        manager.log("⚠️ Không có cấu hình bot nào được tìm thấy!")
    
    # Thông báo số dư ban đầu
    try:
        balance = get_balance()
        manager.log(f"💰 SỐ DƯ BAN ĐẦU: {balance:.2f} USDT")
    except Exception as e:
        manager.log(f"⚠️ Lỗi lấy số dư ban đầu: {str(e)}")
    
    try:
        # Giữ chương trình chạy
        while manager.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        manager.log("👋 Nhận tín hiệu dừng từ người dùng...")
    except Exception as e:
        manager.log(f"⚠️ LỖI HỆ THỐNG NGHIÊM TRỌNG: {str(e)}")
    finally:
        manager.stop_all()

if __name__ == "__main__":
    main()

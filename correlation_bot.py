import ccxt.async_support as ccxt
import asyncio
import numpy as np

class CorrelationEngine:
    def __init__(self, symbol_a, symbol_b):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        self.lookback = 30 # Chu kỳ tính toán tương quan

    async def get_spread_series(self):
        # Lấy dữ liệu lịch sử để tính độ lệch chuẩn
        ohlcv_a = await self.exchange.fetch_ohlcv(self.symbol_a, timeframe='1m', limit=self.lookback)
        ohlcv_b = await self.exchange.fetch_ohlcv(self.symbol_b, timeframe='1m', limit=self.lookback)
        
        prices_a = np.array([x[4] for x in ohlcv_a])
        prices_b = np.array([x[4] for x in ohlcv_b])
        
        # Tính tỷ lệ (Ratio) giữa 2 cặp
        ratios = prices_a / prices_b
        return ratios

    def calculate_zscore(self, current_ratio, ratio_series):
        mean = np.mean(ratio_series)
        std = np.std(ratio_series)
        z_score = (current_ratio - mean) / std
        return z_score

    async def run(self):
        print(f"📡 Đang phân tích sự tương quan giữa {self.symbol_a} và {self.symbol_b}...")
        while True:
            ratios = await self.get_spread_series()
            current_ratio = ratios[-1]
            z = self.calculate_zscore(current_ratio, ratios)
            
            print(f"📊 Ratio: {current_ratio:.4f} | Z-Score: {z:.2f}")
            
            if z > 2:
                print("⚠️ ĐỘ LỆCH CAO: Cân nhắc SHORT A và LONG B (Mean Reversion)")
            elif z < -2:
                print("⚠️ ĐỘ LỆCH THẤP: Cân nhắc LONG A và SHORT B (Mean Reversion)")
                
            await asyncio.sleep(10)

if __name__ == "__main__":
    bot = CorrelationEngine('ETH/USDT', 'BTC/USDT')
    asyncio.run(bot.run())

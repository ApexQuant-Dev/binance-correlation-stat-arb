"""
Phase 1: Binance Futures Data Fetcher & Correlation Matrix
-----------------------------------------------------------
Open-Source Statistical Arbitrage Infrastructure (Public Goods)

This module provides the foundation for fetching historical candlestick data
from Binance Futures (public API) and calculating Pearson correlation matrices
across a basket of cryptocurrency pairs. No API keys are required for read-only
endpoints. All trading logic, risk parameters, and execution functions have been
removed to comply with open-source best practices and security guidelines.

Author: [Your GitHub Handle]
License: MIT
"""

import os
import asyncio
import json
import time
import logging
from datetime import datetime
from collections import deque
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Platform-specific settings
import sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.system("chcp 65001 > nul")

init(autoreset=True)
load_dotenv()

# ============================================================
# Logging Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataFetcher")

# ============================================================
# Public Configuration (Safe for Open Source)
# ============================================================
CONFIG = {
    "TIMEFRAME": '5m',
    "LOOKBACK_CANDLES": 100,            # Number of candles for historical analysis
    "TOP_VOLUME_LIMIT": 30,             # How many top volume coins to fetch
    "MIN_VOLUME_USDT": 5_000_000,       # Minimum 24h volume in USDT
    "CORRELATION_THRESHOLD": 0.7,       # Highlight pairs with correlation above this
    "PAIRS_FILE": "config_pairs.json",  # Optional: user-defined pairs list
}

# Default basket if no config file is found
DEFAULT_PAIRS = [
    ["btcusdt", "ethusdt"],
    ["solusdt", "avaxusdt"],
    ["linkusdt", "uniusdt"],
    ["arbusdt", "opusdt"],
    ["maticusdt", "suiusdt"],
    ["aptusdt", "seiusdt"],
]

# Symbol mapping for Binance Futures quirks
SYMBOL_MAPPING = {
    "shibusdt": "1000shibusdt",
}

# ============================================================
# Helper Functions
# ============================================================
def get_binance_symbol(sym: str) -> str:
    """Map local symbol to Binance Futures symbol."""
    return SYMBOL_MAPPING.get(sym, sym)

def load_pairs_from_file(filename: str) -> List[List[str]]:
    """Load trading pairs from JSON file or fallback to default."""
    try:
        with open(filename, 'r') as f:
            pairs = json.load(f)
            if pairs and isinstance(pairs, list):
                logger.info(f"Loaded {len(pairs)} pairs from {filename}")
                return pairs
    except FileNotFoundError:
        logger.warning(f"{filename} not found. Using default pairs.")
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
    return DEFAULT_PAIRS

# ============================================================
# Core Data Fetcher Class (Public API Only)
# ============================================================
class BinanceDataFetcher:
    """
    Fetches OHLCV data from Binance Futures public endpoints.
    No API keys are required for read-only market data.
    """

    def __init__(self):
        # Initialize exchange without any API keys (public endpoints only)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    async def fetch_top_volume_symbols(self, limit: int = 30) -> List[str]:
        """Fetch top USDT perpetual contracts by 24h volume."""
        logger.info(f"Fetching top {limit} symbols by volume...")
        await self.exchange.load_markets()
        tickers = await self.exchange.fetch_tickers()

        candidates = []
        for sym, ticker in tickers.items():
            if not sym.endswith('/USDT:USDT'):
                continue
            quote_volume = ticker.get('quoteVolume')
            if quote_volume is None or quote_volume < CONFIG["MIN_VOLUME_USDT"]:
                continue
            base = sym.split('/')[0].lower()
            candidates.append((base + 'usdt', quote_volume))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, _ in candidates[:limit]]
        logger.info(f"Selected {len(top_symbols)} symbols.")
        return top_symbols

    async def fetch_historical_closes(self, symbols: List[str]) -> Dict[str, np.ndarray]:
        """Fetch closing prices for a list of symbols."""
        logger.info(f"Fetching {CONFIG['LOOKBACK_CANDLES']} candles for {len(symbols)} symbols...")
        closes = {}
        timeframe = CONFIG["TIMEFRAME"]
        limit = CONFIG["LOOKBACK_CANDLES"]

        for i, sym in enumerate(symbols):
            binance_sym = get_binance_symbol(sym).upper().replace('USDT', '/USDT')
            try:
                ohlcv = await self.exchange.fetch_ohlcv(binance_sym, timeframe, limit=limit)
                if ohlcv and len(ohlcv) >= limit:
                    closes[sym] = np.array([c[4] for c in ohlcv])
                    logger.info(f"  [{i+1}/{len(symbols)}] {sym}: OK")
                else:
                    logger.warning(f"  [{i+1}/{len(symbols)}] {sym}: Insufficient data")
            except Exception as e:
                logger.error(f"  [{i+1}/{len(symbols)}] {sym}: Error - {e}")
            await asyncio.sleep(0.2)  # Rate limit

        return closes

    async def close(self):
        await self.exchange.close()

# ============================================================
# Correlation Analysis
# ============================================================
def compute_correlation_matrix(price_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute Pearson correlation matrix from price series."""
    if not price_dict:
        return pd.DataFrame()

    # Align all series to the same length (take the last N points)
    min_len = min(len(arr) for arr in price_dict.values())
    aligned_data = {}
    for sym, arr in price_dict.items():
        aligned_data[sym] = arr[-min_len:]

    df = pd.DataFrame(aligned_data)
    corr_matrix = df.corr(method='pearson')
    return corr_matrix

def print_correlation_heatmap(corr_matrix: pd.DataFrame, threshold: float = 0.7):
    """Print a color-coded correlation matrix to console."""
    if corr_matrix.empty:
        print(Fore.RED + "No correlation data available.")
        return

    symbols = corr_matrix.columns.tolist()
    print("\n" + Fore.CYAN + Style.BRIGHT + "=" * 80)
    print(f"{'PEARSON CORRELATION MATRIX':^80}")
    print("=" * 80)

    # Print header
    print(f"{'':<10}", end="")
    for sym in symbols:
        print(f"{sym[:8]:>10}", end="")
    print()

    # Print rows
    for sym1 in symbols:
        print(f"{sym1[:8]:<10}", end="")
        for sym2 in symbols:
            corr = corr_matrix.loc[sym1, sym2]
            if sym1 == sym2:
                color = Fore.WHITE
            elif corr >= threshold:
                color = Fore.GREEN
            elif corr <= -threshold:
                color = Fore.RED
            else:
                color = Fore.LIGHTBLACK_EX
            print(f"{color}{corr:>10.3f}{Style.RESET_ALL}", end="")
        print()

    print(Fore.CYAN + "=" * 80)
    print(Fore.YELLOW + f"Threshold: ±{threshold} | Green = positive, Red = negative correlation")

# ============================================================
# Main Async Entry Point
# ============================================================
async def main():
    print(Fore.CYAN + Style.BRIGHT + "=" * 80)
    print(" APEX QUANT - Phase 1: Data Infrastructure & Correlation Matrix")
    print("=" * 80)
    print(Fore.GREEN + "[INFO] This module uses PUBLIC Binance Futures endpoints only.")
    print(Fore.GREEN + "[INFO] No API keys are required. No trading will be executed.\n")

    fetcher = BinanceDataFetcher()

    try:
        # Step 1: Determine symbols to analyze
        pairs = load_pairs_from_file(CONFIG["PAIRS_FILE"])
        # Flatten unique symbols from pairs
        symbols = set()
        for p1, p2 in pairs:
            symbols.add(p1)
            symbols.add(p2)
        symbols = sorted(list(symbols))

        if not symbols:
            # Fallback: fetch top volume symbols automatically
            symbols = await fetcher.fetch_top_volume_symbols(limit=CONFIG["TOP_VOLUME_LIMIT"])
            # Re-generate pairs for display
            pairs = [[symbols[i], symbols[i+1]] for i in range(0, len(symbols)-1, 2)]

        print(Fore.YELLOW + f"Analyzing {len(symbols)} unique symbols:")
        for p in pairs:
            print(f"  {p[0]} / {p[1]}")

        # Step 2: Fetch historical price data
        price_data = await fetcher.fetch_historical_closes(symbols)

        # Step 3: Compute correlation matrix
        if price_data:
            corr_matrix = compute_correlation_matrix(price_data)
            print_correlation_heatmap(corr_matrix, CONFIG["CORRELATION_THRESHOLD"])

            # Optional: Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"correlation_matrix_{timestamp}.csv"
            corr_matrix.to_csv(csv_filename)
            print(Fore.GREEN + f"\n✅ Correlation matrix saved to {csv_filename}")
        else:
            print(Fore.RED + "❌ Failed to fetch price data for correlation analysis.")

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n[USER] Interrupted by user.")
    except Exception as e:
        logger.exception("Unexpected error in main()")
    finally:
        await fetcher.close()

if __name__ == "__main__":
    asyncio.run(main())
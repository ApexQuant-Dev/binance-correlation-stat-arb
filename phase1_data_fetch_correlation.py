"""
Phase 1: Binance Futures Data Fetcher & Correlation Matrix
-----------------------------------------------------------
Open-Source Statistical Arbitrage Infrastructure (Public Goods)

This module fetches historical candlestick data from Binance Futures public
REST endpoints and computes the Pearson correlation matrix for a given set
of trading pairs. No API keys or configuration files are required.

It uses custom DNS resolvers (Google & Cloudflare) to bypass any local DNS
issues on VPS or restricted networks.

Author: [Your GitHub Handle]
License: MIT
"""

import os
import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import aiohttp
from aiohttp.resolver import AsyncResolver
from colorama import init, Fore, Style

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.system("chcp 65001 > nul")

init(autoreset=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataFetcher")

# ============================================================
# PUBLIC CONFIGURATION (SAFE FOR OPEN SOURCE)
# ============================================================
CONFIG = {
    "TIMEFRAME": '5m',
    "LOOKBACK_CANDLES": 100,
    "TOP_VOLUME_LIMIT": 30,
    "MIN_VOLUME_USDT": 5_000_000,
    "CORRELATION_THRESHOLD": 0.7,
    "PAIRS_FILE": "config_pairs.json",
}

DEFAULT_PAIRS = [
    ["btcusdt", "ethusdt"],
    ["solusdt", "avaxusdt"],
    ["linkusdt", "uniusdt"],
    ["arbusdt", "opusdt"],
    ["maticusdt", "suiusdt"],
    ["aptusdt", "seiusdt"],
]

# Custom DNS servers to use when system DNS fails
CUSTOM_NAMESERVERS = ['8.8.8.8', '1.1.1.1', '8.8.4.4']


def load_pairs_from_file(filename: str) -> List[List[str]]:
    """Load trading pairs from a JSON file or fall back to defaults."""
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


class BinanceFuturesREST:
    """
    Lightweight REST client for Binance Futures public endpoints.
    Uses custom DNS resolvers to ensure connectivity on any VPS.
    """

    BASE_URL = "https://fapi.binance.com"

    def __init__(self, proxy: Optional[str] = None):
        self.proxy = proxy
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        # Create a custom resolver with public DNS servers
        resolver = AsyncResolver(nameservers=CUSTOM_NAMESERVERS)
        connector = aiohttp.TCPConnector(
            resolver=resolver,
            ssl=False,          # Disable SSL verification for compatibility
            force_close=True    # Avoid connection pool issues on Windows
        )
        self.session = aiohttp.ClientSession(connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_connection(self) -> bool:
        """Check connectivity to Binance Futures."""
        try:
            url = f"{self.BASE_URL}/fapi/v1/ping"
            async with self.session.get(url, proxy=self.proxy, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{self.BASE_URL}{endpoint}"
        async with self.session.get(url, params=params, proxy=self.proxy) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"HTTP {resp.status}: {text}")
            return await resp.json()

    async def get_top_volume_symbols(self, limit: int = 30) -> List[str]:
        """Fetch top USDT perpetuals by 24h quote volume."""
        data = await self._get("/fapi/v1/ticker/24hr")
        candidates = []
        for item in data:
            symbol = item['symbol']
            if not symbol.endswith('USDT'):
                continue
            quote_volume = float(item.get('quoteVolume', 0))
            if quote_volume < CONFIG["MIN_VOLUME_USDT"]:
                continue
            base = symbol[:-4].lower() + 'usdt'
            candidates.append((base, quote_volume))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in candidates[:limit]]

    async def fetch_klines(self, symbol: str, interval: str, limit: int) -> List[List]:
        """Fetch candlestick data for a single symbol."""
        params = {
            "symbol": symbol.upper().replace('USDT', 'USDT'),
            "interval": interval,
            "limit": limit
        }
        return await self._get("/fapi/v1/klines", params)

    async def fetch_closing_prices(self, symbols: List[str]) -> Dict[str, np.ndarray]:
        """Fetch closing prices for multiple symbols."""
        logger.info(f"Fetching {CONFIG['LOOKBACK_CANDLES']} candles for {len(symbols)} symbols...")
        closes = {}
        interval = CONFIG["TIMEFRAME"]
        limit = CONFIG["LOOKBACK_CANDLES"]

        for i, sym in enumerate(symbols):
            try:
                klines = await self.fetch_klines(sym, interval, limit)
                if len(klines) >= limit:
                    # close price is at index 4
                    closes[sym] = np.array([float(c[4]) for c in klines])
                    logger.info(f"  [{i+1}/{len(symbols)}] {sym}: OK")
                else:
                    logger.warning(f"  [{i+1}/{len(symbols)}] {sym}: Insufficient data")
            except Exception as e:
                logger.error(f"  [{i+1}/{len(symbols)}] {sym}: Error - {e}")
            await asyncio.sleep(0.2)
        return closes


def compute_correlation_matrix(price_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute Pearson correlation matrix from aligned price series."""
    if not price_dict:
        return pd.DataFrame()
    min_len = min(len(arr) for arr in price_dict.values())
    aligned = {sym: arr[-min_len:] for sym, arr in price_dict.items()}
    df = pd.DataFrame(aligned)
    return df.corr(method='pearson')


def print_correlation_heatmap(corr_matrix: pd.DataFrame, threshold: float = 0.7):
    """Display a color-coded correlation matrix in the terminal."""
    if corr_matrix.empty:
        print(Fore.RED + "No correlation data available.")
        return

    symbols = corr_matrix.columns.tolist()
    print("\n" + Fore.CYAN + Style.BRIGHT + "=" * 80)
    print(f"{'PEARSON CORRELATION MATRIX':^80}")
    print("=" * 80)

    # Header
    print(f"{'':<10}", end="")
    for sym in symbols:
        print(f"{sym[:8]:>10}", end="")
    print()

    # Rows
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
    print(Fore.YELLOW + f"Threshold: ±{threshold} | Green = positive, Red = negative")


async def main():
    print(Fore.CYAN + Style.BRIGHT + "=" * 80)
    print(" APEX QUANT - Phase 1: Data Infrastructure & Correlation Matrix")
    print("=" * 80)
    print(Fore.GREEN + "[INFO] Using public Binance Futures REST API only.")
    print(Fore.GREEN + "[INFO] No API keys or .env file are required.")
    print(Fore.GREEN + "[INFO] Custom DNS (8.8.8.8, 1.1.1.1) will bypass system DNS.\n")

    # Check for system proxy variables (if any) but not required
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
    if proxy:
        print(Fore.YELLOW + f"System proxy detected: {proxy}")

    async with BinanceFuturesREST(proxy=proxy) as client:
        print(Fore.CYAN + "Testing connection to Binance Futures...")
        if not await client.test_connection():
            print(Fore.RED + Style.BRIGHT + "=" * 80)
            print(Fore.RED + "ERROR: Cannot reach Binance Futures even with custom DNS.")
            print(Fore.RED + "=" * 80)
            print(Fore.YELLOW + "This is a network-level block. Possible solutions:")
            print(Fore.YELLOW + "  1. Use a VPN (most reliable).")
            print(Fore.YELLOW + "  2. Set a system proxy: set HTTP_PROXY=http://proxy:port")
            print(Fore.YELLOW + "  3. Contact your VPS provider.")
            print(Fore.RED + "=" * 80)
            return

        print(Fore.GREEN + "Connection successful.\n")

        # Determine symbols
        pairs = load_pairs_from_file(CONFIG["PAIRS_FILE"])
        symbols = sorted(list({p for pair in pairs for p in pair}))

        if not symbols:
            print(Fore.YELLOW + "No symbols provided. Fetching top volume symbols...")
            symbols = await client.get_top_volume_symbols(limit=CONFIG["TOP_VOLUME_LIMIT"])
            pairs = [[symbols[i], symbols[i+1]] for i in range(0, len(symbols)-1, 2)]

        print(Fore.YELLOW + f"Analyzing {len(symbols)} unique symbols:")
        for p in pairs:
            print(f"  {p[0]} / {p[1]}")

        # Fetch data and compute correlation
        price_data = await client.fetch_closing_prices(symbols)
        if price_data:
            corr_matrix = compute_correlation_matrix(price_data)
            print_correlation_heatmap(corr_matrix, CONFIG["CORRELATION_THRESHOLD"])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"correlation_matrix_{timestamp}.csv"
            corr_matrix.to_csv(csv_filename)
            print(Fore.GREEN + f"\n✅ Correlation matrix saved to {csv_filename}")
        else:
            print(Fore.RED + "❌ Failed to fetch price data.")


if __name__ == "__main__":
    asyncio.run(main())

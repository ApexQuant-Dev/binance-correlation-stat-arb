# 📉 Binance Correlation & Stat-Arb Suite

This repository contains a professional-grade statistical arbitrage engine designed to exploit mean-reversion opportunities between highly correlated crypto assets.

## 🧠 The Strategy: Z-Score Arbitrage
The system monitors the price ratio between two correlated assets (e.g., BTC/ETH). When the ratio deviates significantly from its historical mean (Z-Score > 2 or < -2), it signals a potential mean-reversion trade:
- **High Z-Score:** Asset A is overpriced relative to Asset B.
- **Low Z-Score:** Asset A is underpriced relative to Asset B.

## ⚡ Technical Highlights
- **Asynchronous Execution:** Handles real-time data streams with `asyncio`.
- **Statistical Modeling:** Uses `NumPy` for moving average and standard deviation calculations.
- **Production Ready:** Structured for easy integration with execution modules.

## 🛠️ Installation
`pip install ccxt numpy`

from pyalloq.optimizers.classical.markowitz import MarkowitzAllocator
from pyalloq.backtest.engine import WalkForwardEngine
import yfinance as yf

tickers = ["VTEC", "VCIG", "SYNA", "LAMR", "MITK"]

df = yf.download(tickers=tickers, start="2024-01-01", end="2026-06-01")["Close"]

allocator = MarkowitzAllocator(tickers=tickers)
engine = WalkForwardEngine(
    allocator=allocator,
    lookback_window=252,
    rebalance_freq="3ME",
)

historical_weights = engine.run(prices=df)
print("Rebalancing weights:")
print(historical_weights)


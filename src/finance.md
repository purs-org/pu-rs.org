# Financial Sidecar

Real-time context for xPU investment and procurement decisions.

## Stock prices (AI chip vendors)

| Ticker | Company | Role |
|---|---|---|
| NVDA | NVIDIA | GPU market leader |
| AMD | AMD | MI300X, CDNA competitor |
| AAPL | Apple | M-series, Metal ecosystem |
| INTC | Intel | Gaudi, Habana |
| GOOG | Google | TPU, custom silicon |
| AMZN | Amazon | Trainium, Inferentia |

## Device street prices

Tracking real-world prices (not MSRP) helps compute true cost-effectiveness:

| Device | MSRP | Street Price | Source |
|---|---|---|---|
| NVIDIA H100 SXM | $30,000 | Check latest | eBay, broker |
| NVIDIA A100 80GB | $10,000 | Check latest | eBay, broker |
| AMD MI300X | $15,000 | Check latest | AMD direct |
| Apple M4 Max (laptop) | $3,999 | Check latest | Apple Store |

## Commodity reference

| Symbol | Relevance |
|---|---|
| Gold (XAU) | Store-of-value benchmark |
| Oil (WTI) | Energy cost proxy |
| BTC | Crypto mining demand affects GPU pricing |
| USD/CNY | Huawei/Cambricon pricing |

*Price data updated weekly via `scripts/fetch_prices.py`.*

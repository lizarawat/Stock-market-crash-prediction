# Dataset Documentation & Reproducible Data Pipeline
**Project:** Stock Market Crash Prediction and Downside Risk Profiling on Indian Equities (NIFTY 50)  
**Course:** INT234: Predictive Analytics  
**Author:** Liza Rawat  
**Date:** Academic Session 2026–2027  

---

## 1. Overview and Ingestion Sources

This project uses daily historical financial time-series spanning **16 full calendar years (January 2010 to December 2025)**, comprising **3,928 daily trading sessions** on the National Stock Exchange of India (NSE) and complementary global macro exchanges.

All data is acquired programmatically via `yfinance` using standardized exchange symbology and cached locally in both Apache Parquet (`.parquet`) and Comma-Separated Values (`.csv`) formats inside `data/raw/`.

### Summary of Ingested Instruments

| Instrument Name | Exchange Ticker | Primary Source | Temporal Coverage | Trading Days | Primary Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NIFTY 50 Index** | `^NSEI` | NSE India / Yahoo Finance | 2010-01-04 to 2025-12-30 | 3,928 | Primary equity benchmark; defines crash target and core technical indicators |
| **India VIX** | `^INDIAVIX` | NSE India / Yahoo Finance | 2010-01-04 to 2025-12-30 | 3,928 | Domestic implied volatility; measures institutional market hedging demand |
| **S&P 500 Index** | `^GSPC` | US Markets / Yahoo Finance | 2010-01-04 to 2025-12-30 | 4,023 | Global equity risk sentiment; captures overnight cross-border contagion |
| **USD / INR Rate** | `INR=X` | Forex Interbank / Yahoo Finance | 2010-01-01 to 2025-12-30 | 4,164 | Exchange rate pressure; signals foreign capital inflows/outflows (FII/DII) |
| **WTI Crude Oil** | `CL=F` | NYMEX / Yahoo Finance | 2010-01-04 to 2025-12-30 | 4,023 | Global commodity cost proxy; key macroeconomic driver for India |
| **Gold Futures** | `GC=F` | COMEX / Yahoo Finance | 2010-01-04 to 2025-12-30 | 4,023 | Flight-to-safety asset during systemic market distress |
| **17 Blue-Chip Equities** | Diverse `.NS` tickers | NSE India / Yahoo Finance | 2010-01-04 to 2025-12-30 | 3,948 each | Cross-sectional universe across 7 core sectors for downside risk clustering |

### Constituent Equities Universe (7 Core Economic Sectors)
1. **Banking & Financial Services:** `HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS`, `KOTAKBANK.NS`, `AXISBANK.NS`
2. **Information Technology:** `TCS.NS`, `INFY.NS`, `WIPRO.NS`
3. **Oil, Gas & Energy:** `RELIANCE.NS`, `ONGC.NS`
4. **Fast-Moving Consumer Goods (FMCG):** `HINDUNILVR.NS`, `ITC.NS`
5. **Automobile:** `MARUTI.NS`
6. **Infrastructure & Engineering:** `LT.NS`
7. **Metals & Mining:** `TATASTEEL.NS`
8. **Telecom & Healthcare:** `BHARTIARTL.NS`, `SUNPHARMA.NS`

---

## 2. Variable Dictionary & Engineered Feature Definitions

The analytical matrix integrates 23 predictive features computed strictly using information available at or before trading day $t$. All target labels are computed strictly over forward horizons $[t+1, t+21]$.

| Feature Identifier | Formula / Construction | Econometric Interpretation | Syllabus Concept |
| :--- | :--- | :--- | :--- |
| `NIFTY_Return_1d` | $\ln(Close_t / Close_{t-1})$ | Daily log return of the NIFTY 50 index | Data Preprocessing (Unit I) |
| `NIFTY_Return_5d` | $\ln(Close_t / Close_{t-5})$ | Short-term weekly momentum | Feature Engineering (Unit I) |
| `NIFTY_Return_21d` | $\ln(Close_t / Close_{t-21})$ | Monthly return momentum | Feature Engineering (Unit I) |
| `NIFTY_Return_63d` | $\ln(Close_t / Close_{t-63})$ | Quarterly trend direction | Feature Engineering (Unit I) |
| `NIFTY_Vol_21d` | $\sigma_{t-20:t}(r) \times \sqrt{252}$ | Annualized 21-day realized historical volatility | Volatility Modeling (Unit I) |
| `NIFTY_Vol_63d` | $\sigma_{t-62:t}(r) \times \sqrt{252}$ | Annualized 63-day medium-term historical volatility | Volatility Modeling (Unit I) |
| `SMA_Ratio_20` | $Close_t / \text{SMA}_{20}(Close)$ | Short-term moving average stretch | Trend Following (Unit II) |
| `SMA_Ratio_50` | $Close_t / \text{SMA}_{50}(Close)$ | Medium-term moving average trend | Trend Following (Unit II) |
| `SMA_Ratio_200` | $Close_t / \text{SMA}_{200}(Close)$ | Long-term macro bull/bear boundary | Trend Following (Unit II) |
| `Dist_From_52w_High`| $(Close_t - \max_{252}(High)) / \max_{252}(High)$ | Distance from rolling 52-week peak (drawdown state) | Drawdown Analysis (Unit I) |
| `RSI_14` | $100 - (100 / (1 + \text{RS}_{14}))$ | Relative Strength Index (oversold/overbought) | Technical Oscillator (Unit I) |
| `MACD_Hist` | $(\text{MACD}_{12,26} - \text{Signal}_9) / Close_t$ | Normalized MACD histogram momentum | Momentum Divergence (Unit I) |
| `Volume_Ratio_21d` | $Volume_t / \text{SMA}_{21}(Volume)$ | Abnormal trading activity / institutional turnover | Liquidity Shock (Unit I) |
| `VIX_Level` | Closing level of `^INDIAVIX` | Domestic market expectation of 30-day volatility | Implied Volatility (Unit II) |
| `VIX_Return_5d` | $(VIX_t - VIX_{t-5}) / VIX_{t-5}$ | Rapid surge in domestic options market panic | Shock Indicator (Unit II) |
| `VIX_Return_21d` | $(VIX_t - VIX_{t-21}) / VIX_{t-21}$ | Monthly change in domestic market anxiety | Shock Indicator (Unit II) |
| `VIX_Ratio_63d` | $VIX_t / \text{SMA}_{63}(VIX)$ | Ratio of current VIX to 3-month baseline | Volatility Regime (Unit II) |
| `SP500_Return_1d` | $\ln(GSPC_{t-1} / GSPC_{t-2})$ | Lagged 1-day return of US S&P 500 (overnight) | Global Contagion (Unit II) |
| `SP500_Return_5d` | $\ln(GSPC_{t-1} / GSPC_{t-6})$ | Lagged 5-day return of US S&P 500 | Global Contagion (Unit II) |
| `USD_INR_Return_5d`| $(INR_{t-1} - INR_{t-6}) / INR_{t-6}$ | Short-term Rupee depreciation pace | Currency Pressure (Unit II) |
| `USD_INR_Return_21d`| $(INR_{t-1} - INR_{t-22}) / INR_{t-22}$ | Medium-term Rupee depreciation trend | Macroeconomic Risk (Unit II) |
| `CRUDE_Return_21d`| $\ln(CL_{t-1} / CL_{t-22})$ | Monthly return on WTI Crude Oil futures | Energy Cost Shock (Unit II) |
| `GOLD_Return_21d` | $\ln(GC_{t-1} / GC_{t-22})$ | Monthly return on Gold futures | Flight-to-Safety Proxy (Unit II) |
| **`Target_Forward_MDD`** | $\min_{\tau \in [1,21]} (Close_{t+\tau} - Close_t) / Close_t$ | Continuous maximum forward drawdown over 21 days | **Regression Target (Unit II)** |
| **`Target_Crash_Binary`**| $\mathbb{I}(\text{Target\_Forward\_MDD} \le -0.07)$ | Binary indicator for severe drawdown exceeding 7% | **Classification Target (Unit III)** |

---

## 3. Data Integrity, Alignment, and Non-Leakage Protocols

### 3.1 Non-Synchronous Trading Calendar Alignment
1. **Reference Time Base:** The NIFTY 50 trading calendar is the master index.
2. **International Assets:** US equities and global commodities operate on different public holidays than the NSE. International inputs are forward-filled (`ffill`) strictly up to day $t-1$. This guarantees that at the time of prediction on day $t$, no future information from foreign markets is consumed.

### 3.2 Look-Ahead Bias and Purging Protocol
- **Target Calculation:** The target $Y_t$ uses price values from $t+1$ to $t+21$.
- **Feature Calculation:** The feature vector $X_t$ uses prices strictly up to $t$.
- **Split Embargo:** Because target horizons overlap by 21 trading days, a 21-day buffer is purged between the training period (ending 2019-12-31) and validation period (starting 2020-01-01), and between validation and testing (2022-01-01).

---

## 4. Reproducible Download Instructions

To regenerate and update the raw dataset locally:
```bash
python -c "
import os, yfinance as yf, pandas as pd
raw_dir = r'data/raw'
os.makedirs(raw_dir, exist_ok=True)
tickers = {'NIFTY_50': '^NSEI', 'INDIA_VIX': '^INDIAVIX', 'SP500': '^GSPC', 
           'USD_INR': 'INR=X', 'CRUDE_OIL': 'CL=F', 'GOLD': 'GC=F'}
for name, sym in tickers.items():
    df = yf.download(sym, start='2010-01-01', end='2025-12-31', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.to_parquet(os.path.join(raw_dir, f'{name}.parquet'))
"
```

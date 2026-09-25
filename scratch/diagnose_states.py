import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# Download closing prices
sp500_close = yf.download("^GSPC", period="5y", interval="1d")["Close"]
tlt_close = yf.download("TLT", period="5y", interval="1d")["Close"]
gld_close = yf.download("GLD", period="5y", interval="1d")["Close"]

if isinstance(sp500_close, pd.DataFrame): sp500_close = sp500_close.squeeze()
if isinstance(tlt_close, pd.DataFrame): tlt_close = tlt_close.squeeze()
if isinstance(gld_close, pd.DataFrame): gld_close = gld_close.squeeze()

daily_return = sp500_close.pct_change()
vol_20 = daily_return.rolling(window=20).std()
mom_20 = sp500_close.pct_change(periods=20)
peak = sp500_close.cummax()
drawdown = (sp500_close - peak) / peak

tlt_return = tlt_close.pct_change()
gld_return = gld_close.pct_change()

corr_sp_tlt = daily_return.rolling(window=20).corr(tlt_return)
corr_sp_gld = daily_return.rolling(window=20).corr(gld_return)

feature_names = [
    "Daily_Return", "Rolling_Volatility_20", "Momentum_20",
    "Drawdown", "SP500_TLT_Corr_20", "SP500_GLD_Corr_20"
]

market_raw_df = pd.DataFrame({
    "Close": sp500_close,
    "Daily_Return": daily_return,
    "Rolling_Volatility_20": vol_20,
    "Momentum_20": mom_20,
    "Drawdown": drawdown,
    "SP500_TLT_Corr_20": corr_sp_tlt,
    "SP500_GLD_Corr_20": corr_sp_gld
})

market_clean_df = market_raw_df.dropna(subset=feature_names).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(market_clean_df[feature_names])

hmm_model = GaussianHMM(n_components=4, covariance_type="full", n_iter=200, random_state=42)
hmm_model.fit(X_scaled)
market_clean_df["HMM_State"] = hmm_model.predict(X_scaled)

print("=== 1. FULL DATASET OVERVIEW ===")
print(f"Total Date Range: {market_clean_df.index.min().strftime('%Y-%m-%d')} to {market_clean_df.index.max().strftime('%Y-%m-%d')}")
print(f"Total Observations: {len(market_clean_df)}")
print("\nOverall State Counts:")
print(market_clean_df["HMM_State"].value_counts().sort_index())

X = market_clean_df[feature_names]
y = market_clean_df["HMM_State"]

train_size = int(len(X) * 0.8)
train_df = market_clean_df.iloc[:train_size]
test_df = market_clean_df.iloc[train_size:]

print("\n=== 2. CHRONOLOGICAL SPLIT OVERVIEW ===")
print(f"Train Set: {train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')} (N={len(train_df)})")
print(train_df["HMM_State"].value_counts().sort_index())

print(f"\nTest Set: {test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')} (N={len(test_df)})")
print(test_df["HMM_State"].value_counts().sort_index())

print("\n=== 3. STATE DATE RANGES & MIN/MAX DATES ===")
for state in range(4):
    state_df = market_clean_df[market_clean_df["HMM_State"] == state]
    if len(state_df) > 0:
        min_d = state_df.index.min().strftime('%Y-%m-%d')
        max_d = state_df.index.max().strftime('%Y-%m-%d')
        print(f"State {state}: N={len(state_df):<4} | Min Date: {min_d} | Max Date: {max_d}")
        # Mean features per state
        means = state_df[feature_names].mean()
        print(f"   Mean Features -> Vol: {means['Rolling_Volatility_20']:.4f}, Mom: {means['Momentum_20']:.4f}, DD: {means['Drawdown']:.4f}")

print("\n=== 4. CONTINUOUS STATE REGIME BLOCKS ===")
market_clean_df["state_change"] = (market_clean_df["HMM_State"] != market_clean_df["HMM_State"].shift(1)).astype(int)
market_clean_df["block_id"] = market_clean_df["state_change"].cumsum()

blocks = market_clean_df.groupby("block_id").agg(
    State=("HMM_State", "first"),
    Start_Date=("HMM_State", lambda x: x.index.min().strftime('%Y-%m-%d')),
    End_Date=("HMM_State", lambda x: x.index.max().strftime('%Y-%m-%d')),
    Days=("HMM_State", "count")
).reset_index(drop=True)

print("Recent Blocks (showing last 15 blocks):")
print(blocks.tail(15).to_string(index=False))

import sys
import os
sys.path.insert(0, os.getcwd())

from app.utils.market_data import get_market_dataset, get_latest_market_snapshot
from app.utils.charts import (
    create_sp500_price_chart,
    create_daily_returns_chart,
    create_rolling_volatility_chart,
    create_drawdown_chart,
    create_cross_asset_correlation_chart
)

print("Fetching dataset...")
df = get_market_dataset(period="5y")
print(f"Loaded {len(df)} observations. Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

snapshot = get_latest_market_snapshot(df)
print("Latest S&P 500 snapshot:", snapshot)

print("Building all 5 charts...")
fig_price = create_sp500_price_chart(df)
fig_ret = create_daily_returns_chart(df)
fig_vol = create_rolling_volatility_chart(df)
fig_dd = create_drawdown_chart(df)
fig_corr = create_cross_asset_correlation_chart(df)

print("Overview page backend data & chart pipeline verified successfully!")

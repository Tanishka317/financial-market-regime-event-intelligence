import sys
import os
sys.path.insert(0, os.getcwd())

from app.utils.event_market import build_event_market_pipeline

print("Testing build_event_market_pipeline...")
res = build_event_market_pipeline(period="5y")

print(f"Total Merged Records: {res['total_news_count']}")
em_df = res["event_market_df"]

if not em_df.empty:
    print("Merged Columns:", em_df.columns.tolist())
    print("\nSample Merged Row:")
    sample_cols = ["published_at", "event_date", "headline", "sentiment_label", "sentiment_score", "event_type", "HMM_State", "next_day_return", "5_day_forward_return"]
    print(em_df[sample_cols].head(3).to_string(index=False))
    
    print("\n--- Event Statistics Table ---")
    print(res["event_stats_df"].to_string(index=False))
    
    print("\n--- Event x HMM State Crosstab ---")
    print(res["crosstab_df"].to_string(index=False))

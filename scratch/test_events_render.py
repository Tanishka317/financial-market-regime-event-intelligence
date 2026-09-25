import sys
import os
sys.path.insert(0, os.getcwd())

from app.utils.event_market import build_event_market_pipeline
from app.utils.charts import (
    create_sentiment_distribution_chart,
    create_event_type_distribution_chart,
    create_event_hmm_crosstab_heatmap
)

print("Testing News & Events backend execution...")
res = build_event_market_pipeline(period="5y")

print(f"Total News Items: {res['total_news_count']}")
em_df = res["event_market_df"]

if not em_df.empty:
    print("Sentiment Distribution:")
    print(em_df["sentiment_label"].value_counts().to_dict())
    
    print("\nEvent Categories:")
    print(em_df["event_type"].value_counts().to_dict())
    
    print("\nGenerating News & Event Charts...")
    fig1 = create_sentiment_distribution_chart(em_df["sentiment_label"].value_counts())
    fig2 = create_event_type_distribution_chart(res["event_stats_df"])
    fig3 = create_event_hmm_crosstab_heatmap(res["crosstab_df"])
    print("News & Events backend verified successfully!")

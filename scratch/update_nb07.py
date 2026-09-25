import json

nb_path = 'notebooks/07_event_market_analysis.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 0 markdown
cell0_src = nb['cells'][0]['source']
for i, line in enumerate(cell0_src):
    if "5-day forward cumulative returns" in line:
        cell0_src[i] = line.replace(
            "5-day forward cumulative returns (`5_day_forward_return`).",
            "5-trading-day forward return following the event date (`5_day_forward_return`)."
        )

# Update Cell 4 markdown
cell4_src = nb['cells'][4]['source']
for i, line in enumerate(cell4_src):
    if "calculate forward returns." in line:
        cell4_src[i] = line.replace(
            "calculate forward returns.",
            "calculate 5-trading-day forward returns following the event date."
        )

# Update Cell 5 code
cell5_src = nb['cells'][5]['source']
for i, line in enumerate(cell5_src):
    if '5_day_forward_return' in line and 'shift(-5)' in line:
        cell5_src[i] = 'market_clean_df["5_day_forward_return"] = (market_clean_df["Close"].shift(-5) / market_clean_df["Close"]) - 1\n'

# Update Cell 10 markdown
cell10_src = nb['cells'][10]['source']
for i, line in enumerate(cell10_src):
    if '5_day_forward_return' in line:
        cell10_src[i] = line.replace(
            "(`next_day_return`, `5_day_forward_return`)",
            "(`next_day_return`, `5_day_forward_return` — 5-trading-day forward return following the event date)"
        )

# Update Cell 13 code (Chart 3 title/labels)
cell13_src = nb['cells'][13]['source']
for i, line in enumerate(cell13_src):
    if "3. Average 5-Day Cumulative Forward S&P 500 Return" in line:
        cell13_src[i] = line.replace(
            "3. Average 5-Day Cumulative Forward S&P 500 Return by Event Category (%)",
            "3. Average 5-Trading-Day Forward S&P 500 Return by Event Category (%)"
        )

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook 07 successfully updated!")

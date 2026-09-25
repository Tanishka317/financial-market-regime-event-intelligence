import json

with open('notebooks/07_event_market_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    print(f"=== Cell {idx} ({cell['cell_type']}) ===")
    print(src[:300] + ('...' if len(src) > 300 else ''))
    print('-'*50)

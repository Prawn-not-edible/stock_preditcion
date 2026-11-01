# Phase6 Stock ID Embedding: controlled ablation

P2: Phase5 frozen no-ID shared model; P2_ID: identical P2 pool + stock-specific learned 5-day bias.
All alpha choices come from three historical development origins. Recent 8 windows are reused development data, NOT independent testing.

## Paired recent-window results

|Stock|Model|Pool|Alpha|Mean NMSE ↓|Median NMSE ↓|Baseline relative|Wins/Ties/Losses|
|---|---|---|---:|---:|---:|---:|---:|
|sh.601088|DLinear|P2|0.00|0.00038081|0.00036833|+0.0%|0/8/0|
|sh.601088|DLinear|P2_ID|0.25|0.00037654|0.00036607|+1.1%|5/0/3|
|sh.601088|TimeXer|P2|0.00|0.00038081|0.00036833|+0.0%|0/8/0|
|sh.601088|TimeXer|P2_ID|0.00|0.00038081|0.00036833|+0.0%|0/8/0|
|sh.601225|DLinear|P2|0.00|0.00042185|0.00035375|+0.0%|0/8/0|
|sh.601225|DLinear|P2_ID|0.00|0.00042185|0.00035375|+0.0%|0/8/0|
|sh.601225|TimeXer|P2|0.00|0.00042185|0.00035375|+0.0%|0/8/0|
|sh.601225|TimeXer|P2_ID|1.00|0.00047204|0.00057032|-11.9%|4/0/4|
|sz.000983|DLinear|P2|0.00|0.00068283|0.00056086|+0.0%|0/8/0|
|sz.000983|DLinear|P2_ID|0.00|0.00068283|0.00056086|+0.0%|0/8/0|
|sz.000983|TimeXer|P2|1.00|0.00063043|0.00066070|+7.7%|6/0/2|
|sz.000983|TimeXer|P2_ID|0.25|0.00064171|0.00058901|+6.0%|5/0/3|

## Phase5 incumbent comparison / selection

- sh.601088: Phase5 incumbent mean=0.00038081; Phase6 selection={'pool': 'Phase5-incumbent', 'reason': 'No P2_ID candidate cleared preset gates'}
- sh.601225: Phase5 incumbent mean=0.00042185; Phase6 selection={'pool': 'Phase5-incumbent', 'reason': 'No P2_ID candidate cleared preset gates'}
- sz.000983: Phase5 incumbent mean=0.00054003; Phase6 selection={'pool': 'Phase5-incumbent', 'reason': 'No P2_ID candidate cleared preset gates'}

## Limitations

1. ID correction is static per stock; cannot model state-dependent cross-stock interactions.
2. No-ID control is reused from Phase5 instead of retrained; compare seeds/inputs/configs for any differences.
3. Improvement in this repeatedly viewed recent period is not proof of out-of-sample performance.
4. Company actions in unadjusted data and survivorship bias remain as previously audited.
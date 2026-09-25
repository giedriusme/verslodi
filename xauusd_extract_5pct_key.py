import json
from pathlib import Path
p=Path('b_reentry_5pct_results/summary.json')
d=json.loads(p.read_text())
out={
 'focus_5pct_tp20_sl6': next(x for x in d['focus_5pct'] if x['tp2']==20.0 and x['sl2']==6.0),
 'trigger_train': d['trigger_only_tp20_sl6_train'],
 'trigger_oos': d['trigger_only_tp20_sl6_oos'],
 'trigger_full': d['trigger_only_tp20_sl6_full'],
}
Path('b_reentry_5pct_results/key_results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps(out,indent=2,allow_nan=False))

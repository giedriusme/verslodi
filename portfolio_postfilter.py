import json
from pathlib import Path
import pandas as pd

SRC=Path('portfolio_discovery_results/grid.csv'); OUT=Path('portfolio_postfilter_results');OUT.mkdir(exist_ok=True)
df=pd.read_csv(SRC)
# balanced score: reward train+OOS expectancy and PF, penalize DD; no hard original-user criteria
df['balance']=0.35*df.train_evR+0.55*df.oos_evR+0.08*(df.train_pf-1)+0.12*(df.oos_pf-1)-0.0025*df.oos_ddR
# Avoid tiny-sample curiosities
base=df[(df.train_n>=80)&(df.oos_n>=50)&(df.train_R>0)&(df.oos_R>0)].copy()
# one best config per family
family=base.sort_values(['oos_pos_years','balance'],ascending=False).drop_duplicates('family')
family.to_csv(OUT/'best_per_family.csv',index=False)
# Stable recent (2023-26) regardless of older filter
recent=base[(base.oos_pos_years>=3)&(base.oos_pf>=1.03)].sort_values(['oos_pos_years','balance'],ascending=False).drop_duplicates('family')
recent.to_csv(OUT/'recent_diverse.csv',index=False)
# Non-breakout complementary families
nonbreak=recent[~recent.family.str.contains('break',case=False,regex=False)].copy().sort_values('balance',ascending=False)
nonbreak.to_csv(OUT/'nonbreak_diverse.csv',index=False)
# TP10/SL25 specifically
tp10=base[base['exit'].eq('TP10_SL25')].sort_values(['oos_pos_years','balance'],ascending=False).drop_duplicates('family')
tp10.to_csv(OUT/'tp10_sl25.csv',index=False)
# Top 12 concise records
cols=['key','family','variant','exit','rr','horizon_min','train_n','train_R','train_evR','train_pf','train_pos_years','oos_n','oos_R','oos_evR','oos_wr','oos_pf','oos_ddR','oos_streak','oos_pos_years']+[f'R_{y}' for y in range(2018,2027)]+[f'n_{y}' for y in range(2023,2027)]
summary={'recent_top':recent.head(12)[cols].to_dict('records'),'nonbreak_top':nonbreak.head(10)[cols].to_dict('records'),'tp10_sl25_top':tp10.head(10)[cols].to_dict('records')}
with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps(summary,indent=2,allow_nan=False))

from pathlib import Path
import json
import pandas as pd
SRC=Path('hold24_results/grid.csv');OUT=Path('hold24_frontier_results');OUT.mkdir(exist_ok=True)
df=pd.read_csv(SRC)
base=df[(df.net_2023>0)&(df.net_2024>0)&(df.net_2025>0)&(df.net_2026>0)&(df.rr>=1)].copy()
base['distance']=(0.65-base.win_rate).clip(lower=0)*100 + (base.max_loss_streak-3).clip(lower=0)*2 - base.ev*.05
wr65=base[base.win_rate>=.65].sort_values(['max_loss_streak','ev'],ascending=[True,False])
s3=base[base.max_loss_streak<=3].sort_values(['win_rate','ev'],ascending=False)
s5=base[base.max_loss_streak<=5].sort_values(['win_rate','ev'],ascending=False)
closest=base.sort_values(['distance','win_rate','ev'],ascending=[True,False,False])
for name,x in [('wr65',wr65),('streak3',s3),('streak5',s5),('closest',closest)]:x.head(30).to_csv(OUT/f'{name}.csv',index=False)
summary={'base_count':len(base),'wr65_count':len(wr65),'streak3_count':len(s3),'streak5_count':len(s5),
         'max_wr':float(base.win_rate.max()) if len(base) else None,'min_streak_wr65':int(wr65.max_loss_streak.min()) if len(wr65) else None,
         'max_wr_streak3':float(s3.win_rate.max()) if len(s3) else None,'max_wr_streak5':float(s5.win_rate.max()) if len(s5) else None,
         'closest':closest.head(10).to_dict('records')}
with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in summary.items() if k!='closest'},indent=2));print(closest.head(10).to_string(index=False))

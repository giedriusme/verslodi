from pathlib import Path
import json
import pandas as pd

SRC=Path('outbox_results/grid.csv');OUT=Path('outbox_frontier_results');OUT.mkdir(exist_ok=True)
df=pd.read_csv(SRC)
base=df[(df.net>0)&(df.net_2023>0)&(df.net_2024>0)&(df.net_2025>0)&(df.net_2026>0)&(df.initial_rr>=1.0)].copy()
base['score']=base.win_rate*100 + base.ev + .2*base.pf - .002*base.max_dd
best_wr=base.sort_values(['win_rate','ev'],ascending=False).head(30)
best_s3=base[base.max_loss_streak<=3].sort_values(['win_rate','ev'],ascending=False).head(30)
best_s5=base[base.max_loss_streak<=5].sort_values(['win_rate','ev'],ascending=False).head(30)
for name,x in [('best_wr',best_wr),('best_streak3',best_s3),('best_streak5',best_s5)]:x.to_csv(OUT/f'{name}.csv',index=False)
summary={'base_count':len(base),'max_win_rate':float(base.win_rate.max()) if len(base) else None,
         'max_wr_streak3':float(best_s3.win_rate.max()) if len(best_s3) else None,
         'max_wr_streak5':float(best_s5.win_rate.max()) if len(best_s5) else None,
         'best_wr':best_wr.head(10).to_dict('records'),'best_streak3':best_s3.head(10).to_dict('records'),'best_streak5':best_s5.head(10).to_dict('records')}
with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)},indent=2))
print(best_wr.head(10).to_string(index=False))

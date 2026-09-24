from pathlib import Path
import json
import pandas as pd

SRC=Path('ml_oos_results/grid.csv')
OUT=Path('ml_oos_postfilter');OUT.mkdir(exist_ok=True)
df=pd.read_csv(SRC)
base=df[(df.freq_ok==True)&(df.net_2023>0)&(df.net_2024>0)&(df.net_2025>0)&(df.net_2026>0)&(df.rr>=1)].copy()
strict=base[(base.wr>=.65)&(base.streak<=3)].sort_values(['rr','wr','ev'],ascending=False)
near=base[(base.wr>=.65)&(base.streak.between(4,5))].sort_values(['rr','wr','ev'],ascending=False)
front=base.sort_values(['wr','ev'],ascending=False)
# Also show best low-streak regardless WR and best WR regardless streak.
s3=base[base.streak<=3].sort_values(['wr','ev'],ascending=False)
s5=base[base.streak<=5].sort_values(['wr','ev'],ascending=False)
for n,x in [('strict',strict),('near_4_5',near),('frontier',front),('streak3_frontier',s3),('streak5_frontier',s5)]:x.head(30).to_csv(OUT/f'{n}.csv',index=False)
summary={'base_count':len(base),'strict_count':len(strict),'near_count':len(near),'max_wr':float(base.wr.max()) if len(base) else None,'min_streak':int(base.streak.min()) if len(base) else None,'max_wr_streak3':float(s3.wr.max()) if len(s3) else None,'max_wr_streak5':float(s5.wr.max()) if len(s5) else None,'frontier':front.head(10).to_dict('records'),'streak3':s3.head(10).to_dict('records'),'streak5':s5.head(10).to_dict('records')}
with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)},indent=2))
print('\nFRONTIER\n',front.head(10).to_string(index=False))
print('\nS3\n',s3.head(10).to_string(index=False))

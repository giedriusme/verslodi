import json
from pathlib import Path
import pandas as pd

SRC=Path('hf_search_results/grid.csv'); OUT=Path('hf_frontier_results'); OUT.mkdir(exist_ok=True)
G=pd.read_csv(SRC)
base=(G.net_2023>0)&(G.net_2024>0)&(G.net_2025>0)&(G.net_2026>0)&(G.pf>1)&(G.rr>=1)
a=G[base&(G.max_loss_streak<=3)].sort_values(['win_rate','ev','n'],ascending=[False,False,False])
b=G[base&(G.max_loss_streak<=5)].sort_values(['win_rate','max_loss_streak','ev'],ascending=[False,True,False])
c=G[base&(G.win_rate>=.70)].sort_values(['max_loss_streak','win_rate','ev'],ascending=[True,False,False])
d=G[base&(G.tp_gt_sl==True)&(G.max_loss_streak<=3)].sort_values(['win_rate','ev'],ascending=[False,False])

def diverse(df,limit=10):
    out=[];seen=set()
    for _,r in df.iterrows():
        if r.family in seen:continue
        out.append(r);seen.add(r.family)
        if len(out)>=limit:break
    return pd.DataFrame(out)
for name,df in [('streak3_topwr',a),('streak5_topwr',b),('wr70_lowstreak',c),('tpgt_streak3',d)]:
    x=diverse(df);x.to_csv(OUT/(name+'.csv'),index=False)
summary={'streak3_count':len(a),'streak5_count':len(b),'wr70_count':len(c),'tpgt_streak3_count':len(d),
         'best_wr_streak3':float(a.win_rate.iloc[0]) if len(a) else None,'best_wr_streak5':float(b.win_rate.iloc[0]) if len(b) else None,
         'min_streak_wr70':int(c.max_loss_streak.min()) if len(c) else None,
         'streak3_top':diverse(a,10).to_dict('records') if len(a) else [],
         'streak5_top':diverse(b,10).to_dict('records') if len(b) else [],
         'wr70_lowstreak':diverse(c,10).to_dict('records') if len(c) else []}
with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)},indent=2))
if len(a): print('\nTOP STREAK<=3\n',diverse(a,10)[['family','variant','tp','sl','rr','n','win_rate','positive','negative','tp_count','sl_count','eod_count','net','pf','max_loss_streak','max_dd','n_2023','n_2024','n_2025','n_2026','net_2023','net_2024','net_2025','net_2026']].to_string(index=False))
if len(c): print('\nWR>=70 LOWEST STREAK\n',diverse(c,10)[['family','variant','tp','sl','rr','n','win_rate','positive','negative','tp_count','sl_count','eod_count','net','pf','max_loss_streak','max_dd','n_2023','n_2024','n_2025','n_2026','net_2023','net_2024','net_2025','net_2026']].to_string(index=False))

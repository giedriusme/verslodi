import json
from pathlib import Path
import pandas as pd

P=Path('rr_search_results')
G=pd.read_csv(P/'all_grid.csv')
base=(G.all_net>0)&(G.train_net>0)&(G.oos_net>0)&(G.all_win_rate>0.80)&(G.rr>=1.0)

outs={}
for min_n,min_oos in [(40,8),(30,5),(20,3),(10,2)]:
    for streak in [3,4,5]:
        q=G[base&(G.all_n>=min_n)&(G.oos_n>=min_oos)&(G.all_max_loss_streak<=streak)].copy()
        q=q.sort_values(['strict_tp_gt_sl','all_years_positive','positive_years','all_win_rate','all_n','all_ev'],ascending=False)
        key=f'n{min_n}_oos{min_oos}_streak{streak}'
        outs[key]={'count':len(q),'strict_tp_gt_sl':int(q.strict_tp_gt_sl.sum()) if len(q) else 0,'all_years_positive':int(q.all_years_positive.sum()) if len(q) else 0}
        if len(q):q.head(100).to_csv(P/f'{key}.csv',index=False)

# Efficient frontier for meaningful sample: RR>=1, profitable train/oos, n>=50, oos>=10.
f=G[(G.all_net>0)&(G.train_net>0)&(G.oos_net>0)&(G.rr>=1)&(G.all_n>=50)&(G.oos_n>=10)].copy()
# Best win rate among low streak groups
front=[]
for st in [3,4,5,6,7,8,10]:
    q=f[f.all_max_loss_streak<=st]
    if len(q):front.append(q.sort_values(['all_win_rate','all_n','all_ev'],ascending=False).iloc[0])
F=pd.DataFrame(front)
if len(F):F.to_csv(P/'frontier.csv',index=False)

# Best diverse candidates closest to requested criteria; prioritize low streak then win rate then RR.
f['distance']=(0.80-f.all_win_rate).clip(lower=0)*10 + (f.all_max_loss_streak-3).clip(lower=0)*0.5 + (1-f.rr).clip(lower=0)*10
f=f.sort_values(['distance','all_max_loss_streak','all_win_rate','all_ev'],ascending=[True,True,False,False])
picks=[];seen=set()
for _,r in f.iterrows():
    root=str(r.family).replace('_long','').replace('_short','')
    if root in seen:continue
    picks.append(r);seen.add(root)
    if len(picks)>=20:break
D=pd.DataFrame(picks)
if len(D):D.to_csv(P/'closest_diverse.csv',index=False)

with open(P/'filter_summary.json','w') as fp:
    json.dump({'threshold_counts':outs,'frontier':F.to_dict('records') if len(F) else [],'closest_diverse':D.to_dict('records') if len(D) else []},fp,indent=2,allow_nan=False)
print(json.dumps({'threshold_counts':outs},indent=2))

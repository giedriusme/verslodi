import json
from pathlib import Path
import pandas as pd
P=Path('b_reentry_reclaim_results')
q=pd.read_csv(P/'grid.csv')
# Only variants whose A+B train period had >=4 positive years.
r=q[q.ab_train_pos_years>=4].copy()
r4=r[r.ab_oos_pos_years==4].copy()
r4['growth_dd_score']=r4.ab_oos_final/(1+3*r4.ab_oos_dd)
r4.sort_values('ab_oos_final',ascending=False).head(15).to_csv(P/'top_oos_after_train_gate.csv',index=False)
r4.sort_values('growth_dd_score',ascending=False).head(15).to_csv(P/'top_growth_dd_after_train_gate.csv',index=False)
nb=q[(q.reclaim.between(1.5,3.0))&(q.wait.isin([60,120,240]))&(q.tp2.isin([12.,15.,20.]))&(q.sl2.isin([8.,10.,12.,15.]))].copy()
nb.sort_values(['reclaim','wait','tp2','sl2']).to_csv(P/'neighborhood.csv',index=False)
summary={
 'best_oos_after_train_gate':r4.sort_values('ab_oos_final',ascending=False).head(5).to_dict('records'),
 'best_growth_dd_after_train_gate':r4.sort_values('growth_dd_score',ascending=False).head(5).to_dict('records'),
 'robust_count':int(len(r4))
}
with open(P/'postfilter.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps(summary,indent=2,allow_nan=False))

import json
from pathlib import Path
import pandas as pd

SRC=Path('rr_search_results/all_grid.csv')
OUT=Path('rr70_filter_results'); OUT.mkdir(exist_ok=True)
G=pd.read_csv(SRC)
# Aggregate prefilter: full years need >=100 each => train total must be >=300; 2026 >50.
base=(G.train_n>=300)&(G.oos_n>50)&(G.all_net>0)&(G.train_net>0)&(G.oos_net>0)&(G.all_win_rate>=0.70)&(G.rr>=1.0)&(G.net_2023>0)&(G.net_2024>0)&(G.net_2025>0)&(G.net_2026>0)
q3=G[base&(G.all_max_loss_streak<=3)].copy()
q5=G[base&(G.all_max_loss_streak.between(4,5))].copy()
# Also show what breaks each criterion for diagnostics.
q_wr_rr=G[(G.train_n>=300)&(G.oos_n>50)&(G.all_net>0)&(G.train_net>0)&(G.oos_net>0)&(G.all_win_rate>=0.70)&(G.rr>=1.0)].copy()
q_years=G[(G.train_n>=300)&(G.oos_n>50)&(G.all_net>0)&(G.train_net>0)&(G.oos_net>0)&(G.all_win_rate>=0.70)&(G.rr>=1.0)&(G.net_2023>0)&(G.net_2024>0)&(G.net_2025>0)&(G.net_2026>0)].copy()
for d in [q3,q5,q_wr_rr,q_years]:
    if len(d): d.sort_values(['all_win_rate','all_ev','all_n'],ascending=[False,False,False],inplace=True)
q3.to_csv(OUT/'strict_streak3.csv',index=False)
q5.to_csv(OUT/'near_streak4_5.csv',index=False)
q_wr_rr.head(100).to_csv(OUT/'wr_rr_highfreq_top100.csv',index=False)
q_years.head(100).to_csv(OUT/'all_years_top100.csv',index=False)

def diverse(df,limit=20):
    out=[]; seen=set()
    for _,r in df.iterrows():
        fam=str(r.family)
        if fam in seen: continue
        out.append(r);seen.add(fam)
        if len(out)>=limit:break
    return pd.DataFrame(out)
D3=diverse(q3);D5=diverse(q5)
D3.to_csv(OUT/'diverse_streak3.csv',index=False);D5.to_csv(OUT/'diverse_streak4_5.csv',index=False)
summary={
 'grid_rows':int(len(G)),
 'wr_rr_highfreq_count':int(len(q_wr_rr)),
 'all_years_positive_count':int(len(q_years)),
 'strict_streak3_count':int(len(q3)),
 'near_streak4_5_count':int(len(q5)),
 'strict_families':int(q3.family.nunique()) if len(q3) else 0,
 'near_families':int(q5.family.nunique()) if len(q5) else 0,
 'strict_diverse':D3.to_dict('records') if len(D3) else [],
 'near_diverse':D5.to_dict('records') if len(D5) else []
}
with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in summary.items() if k not in ('strict_diverse','near_diverse')},indent=2))
if len(D3): print('\nSTRICT\n',D3[['family','variant','tp','sl','rr','train_n','oos_n','all_win_rate','all_positive','all_negative','all_tp_count','all_sl_count','all_eod_count','all_net','all_pf','all_max_loss_streak','net_2023','net_2024','net_2025','net_2026']].to_string(index=False))
if len(D5): print('\nNEAR\n',D5[['family','variant','tp','sl','rr','train_n','oos_n','all_win_rate','all_positive','all_negative','all_tp_count','all_sl_count','all_eod_count','all_net','all_pf','all_max_loss_streak','net_2023','net_2024','net_2025','net_2026']].to_string(index=False))

from pathlib import Path
import json
import pandas as pd

SRC=Path('hf_search_results/grid.csv')
OUT=Path('hf65_filter_results');OUT.mkdir(exist_ok=True)
df=pd.read_csv(SRC)

base=(df['rr']>=1.0)&(df['win_rate']>=0.65)&(df['net']>0)
for y in [2023,2024,2025]:
    base &= df[f'n_{y}']>=100
base &= df['n_2026']>50
for y in [2023,2024,2025,2026]:
    base &= df[f'net_{y}']>0

q=df[base].copy()
q['score']=q['ev'] + 0.25*q['pf'] - 0.002*q['max_dd'] + 0.15*(q['win_rate']-0.65)*100
q3=q[q['max_loss_streak']<=3].sort_values(['tp_gt_sl','score','n'],ascending=[False,False,False])
q45=q[(q['max_loss_streak']>=4)&(q['max_loss_streak']<=5)].sort_values(['tp_gt_sl','score','n'],ascending=[False,False,False])

def root_family(fam):
    root=str(fam)
    for token in ['_600','_660','_720','_780','_840','_900','_cont','_fade','_long','_short']:
        root=root.replace(token,'')
    return root

def diverse(x,limit=25):
    out=[];seen=set()
    for _,r in x.iterrows():
        root=root_family(r['family'])
        if root in seen: continue
        seen.add(root);out.append(r)
        if len(out)>=limit:break
    return pd.DataFrame(out)

d3=diverse(q3);d45=diverse(q45)
q3.to_csv(OUT/'strict_all.csv',index=False);q45.to_csv(OUT/'near_4_5_all.csv',index=False)
d3.to_csv(OUT/'strict_diverse.csv',index=False);d45.to_csv(OUT/'near_4_5_diverse.csv',index=False)
summary={'grid_rows':int(len(df)),'base_count':int(len(q)),'strict_count':int(len(q3)),'near_count':int(len(q45)),
         'strict_families':int(q3.family.nunique()) if len(q3) else 0,'near_families':int(q45.family.nunique()) if len(q45) else 0,
         'strict_top':d3.head(15).to_dict('records'),'near_top':d45.head(15).to_dict('records')}
with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in summary.items() if k not in ['strict_top','near_top']},indent=2))
if len(d3): print('\nSTRICT\n',d3.head(15).to_string(index=False))
if len(d45): print('\nNEAR\n',d45.head(15).to_string(index=False))

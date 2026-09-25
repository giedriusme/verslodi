import itertools, json
from pathlib import Path
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_compound_filter_search as s

OUT=Path('compound_finalists_results');OUT.mkdir(exist_ok=True)
CFG={
 'A_strong_20_12':('A',20.,12.,'close>=0.85'),
 'A_smooth_15_20':('A',15.,20.,'close>=0.85'),
 'A_highWR_6_12':('A',6.,12.,'ALL'),
 'A_highWR_bodyclose_6_12':('A',6.,12.,'body+close'),
 'B_star_20_25':('B',20.,25.,'mom>=0.15'),
 'B_highWR_5_20':('B',5.,20.,'ALL'),
 'B_highWRmom_5_20':('B',5.,20.,'mom>=0.15'),
 'C_30_20':('C',30.,20.,'ALL'),
 'C_30_20_ThuFri':('C',30.,20.,'Thu-Fri'),
}

def main():
    df=p.load(); C=p.candidates(df)
    rows_by={}
    for name,(eng,tp,sl,fn) in CFG.items():
        spec=s.ENGINES[eng];cs=C.get(spec['key'],[]);F=s.feature_frame(df,cs)
        rows=s.build_engine_rows(df,cs,F,eng,tp,sl,spec['h'])
        masks={n:m for n,m in s.filter_defs(F,tp)}
        rows_by[name]=s.apply_mask(rows,masks[fn])
    combos=[]
    A=['A_strong_20_12','A_smooth_15_20','A_highWR_6_12','A_highWR_bodyclose_6_12']
    B=['B_star_20_25','B_highWR_5_20','B_highWRmom_5_20']
    Cc=[None,'C_30_20','C_30_20_ThuFri']
    for a in A:
        combos.append((a,));
    for b in B:combos.append((b,))
    for c in Cc[1:]:combos.append((c,))
    for a,b in itertools.product(A,B):combos.append((a,b))
    for a,c in itertools.product(A,Cc[1:]):combos.append((a,c))
    for b,c in itertools.product(B,Cc[1:]):combos.append((b,c))
    for a,b,c in itertools.product(A,B,Cc):
        if c:combos.append((a,b,c))
    # de-dupe
    seen=set(); combos=[x for x in combos if not (x in seen or seen.add(x))]
    out=[]
    for combo in combos:
        rows=[]
        for n in combo:rows+=rows_by[n]
        tr=s.simulate(rows,s.TRAIN_YEARS);oo=s.simulate(rows,s.OOS_YEARS);full=s.simulate(rows,set(range(2018,2027)))
        rec={'portfolio':' + '.join(combo),'components':len(combo),'train_final':tr['final'],'train_dd':tr['maxdd'],'train_wr':tr['wr'],'train_pos_years':tr['positive_years'],
             'oos_final':oo['final'],'oos_multiple':oo['multiple'],'oos_dd':oo['maxdd'],'oos_wr':oo['wr'],'oos_trades':oo['executed'],'oos_pos_years':oo['positive_years'],
             'oos_max_day_loss':oo['max_day_loss'],'oos_max_day_risk':max([d['risk_frac'] for d in oo['daily']],default=0.0),
             'full_final':full['final'],'full_dd':full['maxdd']}
        for y,v in oo['years'].items():rec[f'ret_{y}']=v['return']
        out.append(rec)
    q=pd.DataFrame(out)
    q['growth_dd_score']=q.oos_multiple/(1+3*q.oos_dd)
    q.sort_values('oos_final',ascending=False).to_csv(OUT/'all.csv',index=False)
    q[q.oos_dd<=.50].sort_values('oos_final',ascending=False).to_csv(OUT/'dd50.csv',index=False)
    q[q.oos_dd<=.40].sort_values('oos_final',ascending=False).to_csv(OUT/'dd40.csv',index=False)
    q.sort_values('growth_dd_score',ascending=False).to_csv(OUT/'growth_dd.csv',index=False)
    summary={'top_dd50':q[q.oos_dd<=.50].sort_values('oos_final',ascending=False).head(10).to_dict('records'),
             'top_dd40':q[q.oos_dd<=.40].sort_values('oos_final',ascending=False).head(10).to_dict('records'),
             'top_growth_dd':q.sort_values('growth_dd_score',ascending=False).head(10).to_dict('records')}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))
if __name__=='__main__':main()

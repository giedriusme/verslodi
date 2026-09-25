import json, math, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_prevclose_regime_research as r

OUT=Path('prevclose_regime_fast_results'); OUT.mkdir(exist_ok=True)
YEARS=r.YEARS; TRAIN=r.TRAIN; OOS=r.OOS; TPS=r.TPS; SLS=r.SLS; HORIZONS=r.HORIZONS
START=r.START; VALUE=r.VALUE; RISK=r.RISK

def stats(arr, years, selected_years):
    m=np.isin(years,list(selected_years)); a=arr[m]
    if not len(a): return None
    gp=a[a>0].sum(); gl=-a[a<0].sum()
    yp={int(y):float(arr[years==y].sum()) for y in sorted(set(years[m]))}
    return {'n':int(len(a)),'net':float(a.sum()),'ev':float(a.mean()),'wr':float((a>0).mean()),
            'pf':float(gp/gl) if gl>0 else 999.,'positive_years':sum(v>0 for v in yp.values()),'yearpts':yp}

def compound(arr,years,selected_years,sl):
    eq=START; peak=eq; dd=0.; stalled=False
    for pts,y in zip(arr,years):
        if y not in selected_years: continue
        lot=r.lot_size(eq,sl)
        if lot<.01: stalled=True; continue
        eq=max(0.,eq+float(pts)*VALUE*lot); peak=max(peak,eq); dd=max(dd,(peak-eq)/peak if peak else 1.)
    return eq,dd,stalled

def main():
    df=p.load(); variants=r.build_signals(df); summary={}
    for name,signals in variants.items():
        print('variant',name,len(signals),flush=True)
        n=len(signals); years=np.array([s['year'] for s in signals],int)
        paths={h:[r.trade_path(df,int(s['entry_i']),s['side'],h) for s in signals] for h in HORIZONS}
        outcome_cache={}
        for tp,sl,h in itertools.product(TPS,SLS,HORIZONS):
            outcome_cache[(tp,sl,h)]=np.array([r.outcome(x,tp,sl)[0] for x in paths[h]],float)
        filters=r.make_filters(signals)
        candidates=[]
        for fname,fn in filters:
            fm=np.array([bool(fn(s)) for s in signals]); trmask=fm & np.isin(years,list(TRAIN))
            if trmask.sum()<120: continue
            for tp,sl,h in itertools.product(TPS,SLS,HORIZONS):
                arr=outcome_cache[(tp,sl,h)]; a=arr[trmask]
                gp=a[a>0].sum(); gl=-a[a<0].sum(); ev=float(a.mean()); pf=float(gp/gl) if gl>0 else 999.; wr=float((a>0).mean())
                py=0
                for y in TRAIN:
                    yy=arr[fm & (years==y)]
                    if len(yy) and yy.sum()>0: py+=1
                if ev<=0 or pf<=1 or py<3: continue
                score=ev*(1+min(pf,2))+.05*py
                candidates.append({'variant':name,'filter':fname,'tp':tp,'sl':sl,'h':h,'train_score':score,
                                   'train_n':int(len(a)),'train_ev':ev,'train_pf':pf,'train_wr':wr,'train_pos_years':py})
        tg=pd.DataFrame(candidates)
        if len(tg): tg=tg.sort_values('train_score',ascending=False)
        tg.to_csv(OUT/f'{name}_train_candidates.csv',index=False)
        outs=[]
        fdict=dict(filters)
        for rec in (tg.head(150).to_dict('records') if len(tg) else []):
            fm=np.array([bool(fdict[rec['filter']](s)) for s in signals]); arr=outcome_cache[(float(rec['tp']),float(rec['sl']),int(rec['h']))]
            oo=stats(arr[fm],years[fm],OOS); fu=stats(arr[fm],years[fm],set(YEARS)); eq,dd,st=compound(arr[fm],years[fm],OOS,float(rec['sl']))
            z=dict(rec); z.update({'oos_n':oo['n'],'oos_ev':oo['ev'],'oos_pf':oo['pf'],'oos_wr':oo['wr'],'oos_pos_years':oo['positive_years'],
                                  'oos_yearpts':json.dumps(oo['yearpts']),'oos_final':eq,'oos_dd':dd,'oos_stalled':st,
                                  'full_ev':fu['ev'],'full_pf':fu['pf'],'full_wr':fu['wr']})
            z['robust']=bool(oo['ev']>0 and oo['pf']>1 and oo['positive_years']>=3 and not st)
            z['robust_score']=math.log(max(eq/START,1e-9))-1.25*dd+max(0,oo['ev'])*(1+min(oo['pf'],2))
            outs.append(z)
        od=pd.DataFrame(outs)
        if len(od): od=od.sort_values('robust_score',ascending=False)
        od.to_csv(OUT/f'{name}_oos.csv',index=False)
        robust=od[od.robust].copy() if len(od) else pd.DataFrame()
        robust.to_csv(OUT/f'{name}_robust.csv',index=False)
        # Raw baseline by year for three settings.
        yr=[]
        fm=np.ones(n,dtype=bool)
        for tp,sl,h in [(30.,30.,480),(30.,15.,480),(20.,20.,480)]:
            arr=outcome_cache[(tp,sl,h)]
            for y in YEARS:
                yy=arr[years==y]
                if len(yy):
                    gp=yy[yy>0].sum(); gl=-yy[yy<0].sum(); yr.append({'tp':tp,'sl':sl,'h':h,'year':y,'n':len(yy),'net':yy.sum(),'ev':yy.mean(),'wr':(yy>0).mean(),'pf':gp/gl if gl>0 else 999.})
        pd.DataFrame(yr).to_csv(OUT/f'{name}_yearly_baselines.csv',index=False)
        summary[name]={'signals':n,'train_candidates':int(len(tg)),'robust_count':int(len(robust)),
                       'best':robust.head(12).to_dict('records') if len(robust) else [],
                       'top_train_then_oos':od.head(8).to_dict('records') if len(od) else []}
    with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2,allow_nan=True)
    print(json.dumps(summary,indent=2,allow_nan=True))

if __name__=='__main__': main()

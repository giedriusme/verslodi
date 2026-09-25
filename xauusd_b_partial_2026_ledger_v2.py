import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_partial_portfolio_final as pf
import xauusd_b_partial_exit_search as b

OUT=Path('b_partial_2026_ledger_v2'); OUT.mkdir(exist_ok=True)
START=300.0; VALUE=100.0; RISK=.10; YEAR=2026
TP1=4.0; TP2=25.0; SL=20.0; FRAC=.5; MODE='ORIG'; H=480

def lot_size(eq):
    raw=(eq*RISK)/(SL*VALUE)
    return round(max(0,math.floor((raw+1e-12)/.01))*.01,2)

def ts(df,i):
    if i is None or i<0 or i>=len(df): return ''
    return pd.Timestamp(df.iloc[int(i)].dt_local).isoformat()

def detailed_outcome(df,c,total_lot):
    base=b.base_path(df,c)
    fav,adv,mark=base
    a1=np.flatnonzero(fav>=TP1); bs=np.flatnonzero(adv>=SL)
    i1=int(a1[0]) if len(a1) else None; ib0=int(bs[0]) if len(bs) else None
    ei=int(c['entry_i'])
    rec={'date':c['date'],'side':c['side'],'entry_time':ts(df,ei),'entry_price':float(c['entry']),'lot':total_lot}
    sp=b.split_lots(total_lot,FRAC)
    if sp is None:
        # Validated fallback for 0.01 lot: full position targets TP2.
        a=np.flatnonzero(fav>=TP2); bb=np.flatnonzero(adv>=SL)
        ia=int(a[0]) if len(a) else None; ib=int(bb[0]) if len(bb) else None
        if ia is not None and (ib is None or ia<ib): pts=TP2; off=ia; state='TP25_FALLBACK'
        elif ib is not None: pts=-SL; off=ib; state='SL20_FALLBACK'
        else: pts=float(mark); off=len(fav)-1; state='TIME_FALLBACK'
        rec.update({'mode':'0.01 fallback -> full TP25','lot_tp4':0.0,'lot_runner':total_lot,'tp4_points':0.0,'runner_points':pts,'result_points_weighted':pts,'outcome':state,'exit_time':ts(df,ei+off)})
        return rec
    l1,l2=sp
    if ib0 is not None and (i1 is None or ib0<=i1):
        pts=-SL
        rec.update({'mode':'split','lot_tp4':l1,'lot_runner':l2,'tp4_points':pts,'runner_points':pts,'result_points_weighted':pts,'outcome':'SL20_BEFORE_TP4','exit_time':ts(df,ei+ib0)})
        return rec
    if i1 is None:
        pts=float(mark)
        rec.update({'mode':'split','lot_tp4':l1,'lot_runner':l2,'tp4_points':pts,'runner_points':pts,'result_points_weighted':pts,'outcome':'TIME_PRE_TP4','exit_time':ts(df,ei+len(fav)-1)})
        return rec
    a2=np.flatnonzero(fav[i1:]>=TP2); i2=int(a2[0])+i1 if len(a2) else None
    bs2=np.flatnonzero(adv[i1:]>=SL); ib=int(bs2[0])+i1 if len(bs2) else None
    if ib is not None and (i2 is None or ib<=i2): runner=-SL; off=ib; state='TP4 + RUNNER_SL20'
    elif i2 is not None: runner=TP2; off=i2; state='TP4 + TP25'
    else: runner=float(mark); off=len(fav)-1; state='TP4 + RUNNER_TIME'
    weighted=(TP1*l1+runner*l2)/total_lot
    rec.update({'mode':'split 50/50','lot_tp4':l1,'lot_runner':l2,'tp4_points':TP1,'runner_points':runner,'result_points_weighted':weighted,'outcome':state,'exit_time':ts(df,ei+off),'tp4_time':ts(df,ei+i1)})
    return rec

def main():
    df=p.load(); C=p.candidates(df); Bcs,Bbases=pf.build_b(df,C)
    pairs=[(c,base) for c,base in zip(Bcs,Bbases) if int(c['year'])==YEAR]
    eq=START; peak=eq; maxdd=0.0; rows=[]
    for c,base in pairs:
        bod=eq; lot=lot_size(bod)
        if lot<.01:
            rows.append({'date':c['date'],'start_balance':round(bod,2),'lot':0.0,'outcome':'SKIP_MINLOT','pnl_eur':0.0,'end_balance':round(eq,2)})
            continue
        d=detailed_outcome(df,c,lot)
        if d['mode'].startswith('0.01'):
            pnl=d['runner_points']*VALUE*lot
        else:
            pnl=(d['tp4_points']*VALUE*d['lot_tp4'])+(d['runner_points']*VALUE*d['lot_runner'])
        eq=max(0.0,eq+pnl); peak=max(peak,eq); maxdd=max(maxdd,(peak-eq)/peak if peak else 1.0)
        d.update({'start_balance':round(bod,2),'risk_eur_nominal':round(SL*VALUE*lot,2),'risk_pct_actual':round(100*SL*VALUE*lot/bod,2),'pnl_eur':round(pnl,2),'end_balance':round(eq,2)})
        rows.append(d)
    q=pd.DataFrame(rows)
    q.to_csv(OUT/'ledger_2026.csv',index=False)
    cols=['date','side','entry_time','lot','mode','lot_tp4','lot_runner','outcome','pnl_eur','start_balance','end_balance']
    q[[c for c in cols if c in q.columns]].to_csv(OUT/'ledger_2026_compact.csv',index=False)
    # Lot progression milestones.
    m=q[q.get('lot',pd.Series(dtype=float)).fillna(0).gt(0)].copy()
    milestones=[]; last=None
    for _,r in m.iterrows():
        if last is None or float(r['lot'])!=last:
            milestones.append({'date':r['date'],'balance_before':r['start_balance'],'new_lot':r['lot']})
            last=float(r['lot'])
    pd.DataFrame(milestones).to_csv(OUT/'lot_milestones.csv',index=False)
    summary={'start':START,'final':round(eq,2),'return_pct':round((eq/START-1)*100,2),'max_dd_pct':round(maxdd*100,2),'trades':int((q.lot.fillna(0)>=.01).sum()),'milestones':milestones,
             'rule':'Thu/Fri Asia breakout + mom>=0.15; 10% risk; SL20; if >=0.02 lot split 50% TP4 + 50% TP25; if 0.01 lot fallback full TP25.'}
    with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()

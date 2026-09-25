import json, math
from pathlib import Path
import numpy as np
import pandas as pd

import xauusd_portfolio_discovery as p
import xauusd_partial_portfolio_final as pf
import xauusd_b_partial_exit_search as b

OUT=Path('b_partial_2026_ledger'); OUT.mkdir(exist_ok=True)
START=300.0; RISK=.10; VALUE=100.0; LOT_STEP=.01
TP1=4.0; TP2=25.0; SL=20.0; FRAC=.5; MODE='ORIG'; H=480
YEAR=2026


def lot_size(eq):
    raw=(eq*RISK)/(SL*VALUE)
    return round(max(0,math.floor((raw+1e-12)/LOT_STEP))*LOT_STEP,2)


def ts(df,i):
    if i is None or i<0 or i>=len(df):return ''
    return pd.Timestamp(df.iloc[int(i)].dt_local).isoformat()


def full_detail(df,c,tp):
    i=int(c['entry_i']); e=float(c['entry']); f=df.iloc[i:min(len(df),i+H)]
    if c['side']=='LONG':
        fav=f.high_bid.to_numpy(float)-e; adv=e-f.low_bid.to_numpy(float); mark=float(f.iloc[-1].close_bid-e)
    else:
        fav=e-f.low_ask.to_numpy(float); adv=f.high_ask.to_numpy(float)-e; mark=float(e-f.iloc[-1].close_ask)
    a=np.flatnonzero(fav>=tp); z=np.flatnonzero(adv>=SL)
    ia=int(a[0]) if len(a) else None; iz=int(z[0]) if len(z) else None
    if ia is not None and (iz is None or ia<iz): return tp,'TP',i+ia
    if iz is not None:return -SL,'SL',i+iz
    return mark,'TIME',i+len(f)-1


def split_detail(df,c):
    base=b.base_path(df,c); p1,pr,state=b.partial_outcome(base,TP1,TP2,SL,MODE)
    i=int(c['entry_i']); e=float(c['entry']); f=df.iloc[i:min(len(df),i+H)]
    if c['side']=='LONG': fav=f.high_bid.to_numpy(float)-e; adv=e-f.low_bid.to_numpy(float)
    else: fav=e-f.low_ask.to_numpy(float); adv=f.high_ask.to_numpy(float)-e
    a1=np.flatnonzero(fav>=TP1); i1=int(a1[0]) if len(a1) else None
    if state=='SL':
        z=np.flatnonzero(adv>=SL); iz=int(z[0]); return p1,pr,state,i+iz,i+iz
    if state=='TIME_PRE_TP1': return p1,pr,state,i+len(f)-1,i+len(f)-1
    leg1_exit=i+i1
    a2=np.flatnonzero(fav[i1:]>=TP2); z2=np.flatnonzero(adv[i1:]>=SL)
    ia2=(i1+int(a2[0])) if len(a2) else None; iz2=(i1+int(z2[0])) if len(z2) else None
    if state=='TP2': runner_exit=i+ia2
    elif state=='TP1_STOP': runner_exit=i+iz2
    else: runner_exit=i+len(f)-1
    return p1,pr,state,leg1_exit,runner_exit


def main():
    df=p.load(); C=p.candidates(df); Bcs,Bbases=pf.build_b(df,C)
    cs=[c for c in Bcs if int(c['year'])==YEAR]
    eq=START; peak=eq; maxdd=0.; rows=[]
    for c in cs:
        bod=eq; lot=lot_size(bod)
        rec={'date':c['date'],'side':c['side'],'entry_time':ts(df,int(c['entry_i'])),'entry_price':float(c['entry']),
             'start_balance':round(bod,2),'lot':lot,'risk_pct_nominal':round((SL*VALUE*lot/bod*100),4) if bod>0 else 0.}
        if lot<.01:
            rec.update({'mode':'SKIP_MIN_LOT','result':'SKIP','pnl_eur':0.,'end_balance':round(eq,2)})
            rows.append(rec); continue
        sp=b.split_lots(lot,FRAC)
        if sp is None:
            pts,state,xi=full_detail(df,c,TP2); pnl=pts*VALUE*lot
            result=('TP25' if state=='TP' else 'SL20' if state=='SL' else f'TIME {pts:+.2f}')
            rec.update({'mode':'FULL_001_FALLBACK_TP25','leg1_lot':np.nan,'runner_lot':np.nan,'leg1_points':np.nan,'runner_points':np.nan,
                        'outcome':state,'result':result,'exit_time':ts(df,xi),'pnl_eur':round(pnl,2)})
        else:
            l1,l2=sp; p1,pr,state,x1,x2=split_detail(df,c); pnl=p1*VALUE*l1+pr*VALUE*l2
            if state=='SL': result='SL20 full'
            elif state=='TIME_PRE_TP1': result=f'TIME {p1:+.2f} full'
            else:
                rr='TP25' if state=='TP2' else 'SL20' if state=='TP1_STOP' else f'TIME {pr:+.2f}'
                result=f'TP4({l1:.2f}) + {rr}({l2:.2f})'
            rec.update({'mode':'SPLIT_50_50','leg1_lot':l1,'runner_lot':l2,'leg1_points':round(float(p1),4),'runner_points':round(float(pr),4),
                        'outcome':state,'result':result,'leg1_exit_time':ts(df,x1),'runner_exit_time':ts(df,x2),'exit_time':ts(df,x2),'pnl_eur':round(pnl,2)})
        eq=max(0.,bod+pnl); peak=max(peak,eq); maxdd=max(maxdd,(peak-eq)/peak if peak else 1.)
        rec['end_balance']=round(eq,2); rows.append(rec)
    q=pd.DataFrame(rows); q.to_csv(OUT/'ledger_2026.csv',index=False)
    compact=q[['date','side','start_balance','lot','mode','result','pnl_eur','end_balance']].copy(); compact.to_csv(OUT/'compact_2026.csv',index=False)
    summary={'system':'B Partial Balanced: Thu/Fri Asia breakout, momentum>=0.15, 10% risk, SL20. If lot>=0.02 split 50% TP4 + 50% TP25 with original SL20; if 0.01 lot, fallback full TP25.',
             'start':START,'final':round(eq,2),'return_pct':round((eq/START-1)*100,2),'max_dd_pct':round(maxdd*100,2),
             'trades':int((q['mode']!='SKIP_MIN_LOT').sum()),'split_trades':int((q['mode']=='SPLIT_50_50').sum()),'fallback_001_trades':int((q['mode']=='FULL_001_FALLBACK_TP25').sum()),
             'first_split_date':str(q.loc[q['mode']=='SPLIT_50_50','date'].iloc[0]) if (q['mode']=='SPLIT_50_50').any() else None,
             'max_lot':float(q['lot'].max())}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2)); print(compact.to_string(index=False))

if __name__=='__main__':main()

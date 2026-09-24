import json
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p

OUT=Path('final_portfolio_results');OUT.mkdir(exist_ok=True)

def metrics(r):
    r=r.sort_values(['entry_i','strategy']).reset_index(drop=True)
    a=r.R.to_numpy(float); gains=a[a>0].sum(); losses=-a[a<0].sum(); eq=np.cumsum(a); peak=np.maximum.accumulate(np.r_[0.,eq]); dd=peak[1:]-eq
    streak=cur=0
    for x in a:cur=cur+1 if x<0 else 0;streak=max(streak,cur)
    yd={}
    for y,g in r.groupby('year'):
        x=g.R.to_numpy(float); yd[int(y)]={'n':int(len(g)),'R':float(x.sum()),'wr':float((x>0).mean()),'pf':float(x[x>0].sum()/(-x[x<0].sum())) if (x<0).any() else 99.}
    return {'n':int(len(r)),'R':float(a.sum()),'evR':float(a.mean()),'wr':float((a>0).mean()),'pf':float(gains/losses) if losses>0 else 99.,'ddR':float(dd.max()),'streak':int(streak),'years':yd}

def main():
    df=p.load(); df=df.copy();df['global_i']=np.arange(len(df))
    rows=[]; ar=[];prev=None
    for day,g0 in df.groupby('date',sort=True):
        if pd.Timestamp(day).dayofweek>=5:continue
        g=g0.sort_values('timestamp').reset_index(drop=True);asia=g[(g['mod']>=60)&(g['mod']<540)]
        if len(asia)<420:
            if len(g):prev=float(g.iloc[-1].close_mid)
            continue
        AH=float(asia.high_mid.max());AL=float(asia.low_mid.min());AR=AH-AL;med=float(np.median(ar[-20:])) if len(ar)>=10 else np.nan;wd=int(g.iloc[0].wd)
        sig=[]
        # A compression breakout: Asia range <=70% of rolling 20-session median; first close outside 09-16
        if np.isfinite(med) and AR/med<=0.70:
            idx,side=p.first_break(g[(g['mod']>=540)&(g['mod']<960)],AH,AL)
            if side:
                c=p.mk(g,idx,side,'compression','r070');z=p.trade_result(df,c,15.,15.,720)
                if z:sig.append((c,z,'Compression breakout TP15/SL15 H12'))
        # B Thu/Fri breakout 10-16
        if wd in (3,4):
            idx,side=p.first_break(g[(g['mod']>=600)&(g['mod']<960)],AH,AL)
            if side:
                c=p.mk(g,idx,side,'thu_fri_break','all');z=p.trade_result(df,c,25.,20.,1440)
                if z:sig.append((c,z,'Thu/Fri Asia breakout TP25/SL20 H24'))
        # C previous-close continuation at 14:00
        if prev is not None:
            ss=g[g['mod']==840]
            if not ss.empty:
                idx=ss.index[0];px=float(ss.iloc[0].close_mid);side='LONG' if px>prev else 'SHORT';c=p.mk(g,idx,side,'prevclose14','all');z=p.trade_result(df,c,30.,25.,1440)
                if z:sig.append((c,z,'14:00 prev-close continuation TP30/SL25 H24'))
        for c,z,name in sig:
            rows.append({'date':z['date'],'year':z['year'],'strategy':name,'entry_i':c['entry_i'],'exit_i':c['entry_i']+z['hold']-1,'R':z['R'],'outcome':z['outcome'],'side':z['side']})
        ar.append(AR);prev=float(g.iloc[-1].close_mid)
    r=pd.DataFrame(rows);r.to_csv(OUT/'all_trades.csv',index=False)
    strat={name:metrics(g) for name,g in r.groupby('strategy')}
    # equal-risk portfolio daily: each engine has fixed 1/3 risk budget, zero on no-signal day
    piv=r.pivot_table(index='date',columns='strategy',values='R',aggfunc='sum').fillna(0.)
    for name in strat:
        if name not in piv:piv[name]=0.
    daily=piv[list(strat)].sum(axis=1)/3.0
    dates=pd.to_datetime(daily.index);pdaily=pd.DataFrame({'date':daily.index,'year':dates.year,'R':daily.values,'entry_i':np.arange(len(daily)),'strategy':'portfolio'})
    pm=metrics(pdaily)
    corr=piv.corr();corr.to_csv(OUT/'correlation.csv')
    # causal one-trade/day: earliest signal that day, then enforce no overlapping active trade
    one=[];active_until=-1
    for date,g in r.sort_values(['entry_i','strategy']).groupby('date',sort=True):
        gg=g[g.entry_i>active_until]
        if gg.empty:continue
        x=gg.sort_values(['entry_i','strategy']).iloc[0];one.append(x.to_dict());active_until=int(x.exit_i)
    one=pd.DataFrame(one);one.to_csv(OUT/'one_trade_meta.csv',index=False);om=metrics(one) if len(one) else None
    summary={'data_start':str(df.dt_local.iloc[0]),'data_end':str(df.dt_local.iloc[-1]),'strategies':strat,'equal_risk_portfolio':pm,'one_trade_meta':om,'correlation':corr.round(3).to_dict()}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':main()

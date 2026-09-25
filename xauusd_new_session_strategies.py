import json, math
from pathlib import Path
import numpy as np
import pandas as pd

import xauusd_portfolio_discovery as p

OUT=Path('new_session_strategy_results'); OUT.mkdir(exist_ok=True)
TRAIN=set(range(2018,2023)); OOS=set(range(2023,2027)); ALL=set(range(2018,2027))
TPS=[4.,5.,7.5,10.,12.5,15.,20.,25.,30.,40.,50.]
SLS=[5.,7.5,10.,12.5,15.,20.,25.,30.,40.,50.]
HORIZONS=[480,720,1440]


def trade(df,c,tp,sl,h):
    i=int(c['entry_i']); e=float(c['entry']); side=c['side']; f=df.iloc[i:min(len(df),i+h)]
    if f.empty:return None
    if side=='LONG':
        fav=f.high_bid.to_numpy(float)-e; adv=e-f.low_bid.to_numpy(float); mark=float(f.iloc[-1].close_bid-e)
    else:
        fav=e-f.low_ask.to_numpy(float); adv=f.high_ask.to_numpy(float)-e; mark=float(e-f.iloc[-1].close_ask)
    a=np.flatnonzero(fav>=tp); b=np.flatnonzero(adv>=sl)
    ia=int(a[0]) if len(a) else None; ib=int(b[0]) if len(b) else None
    if ia is not None and (ib is None or ia<ib): pnl=tp; out='TP'; hold=ia+1
    elif ib is not None: pnl=-sl; out='SL'; hold=ib+1
    else: pnl=mark; out='TIME'; hold=len(f)
    return {'date':c['date'],'year':c['year'],'side':side,'pnl':float(pnl),'R':float(pnl/sl),'outcome':out,'hold':hold}


def metrics(rows,years):
    r=pd.DataFrame(rows)
    if r.empty:return None
    r=r[r.year.isin(years)]
    if r.empty:return None
    R=r.R.to_numpy(float); eq=np.cumsum(R); peak=np.maximum.accumulate(np.r_[0.,eq]); dd=peak[1:]-eq
    gains=R[R>0].sum(); losses=-R[R<0].sum(); cur=mx=0
    for x in R:
        cur=cur+1 if x<0 else 0; mx=max(mx,cur)
    yd={}
    for y,g in r.groupby('year'):
        yd[int(y)]={'n':int(len(g)),'R':float(g.R.sum()),'wr':float((g.R>0).mean())}
    return {'n':int(len(r)),'R':float(R.sum()),'evR':float(R.mean()),'wr':float((R>0).mean()),
            'pf':float(gains/losses) if losses>0 else 99.,'ddR':float(dd.max()) if len(dd) else 0.,'streak':mx,'years':yd}


def build_signals(df):
    d=df.copy(); d['global_i']=np.arange(len(d))
    sig={'prev_0959_to_1000':[], 'prev_1000close_to_1001':[], 'london_us_meanrev':[]}
    prev_close=None
    for day,g0 in d.groupby('date',sort=True):
        g=g0.sort_values('timestamp')
        # Previous trading-day close must stay Friday through the weekend.
        if int(g.iloc[0].wd)>=5:
            continue
        date=str(day); year=int(g.iloc[0].year); wd=int(g.iloc[0].wd)
        if prev_close is not None:
            r959=g[g['mod']==599]; r1000=g[g['mod']==600]; r1001=g[g['mod']==601]
            if not r959.empty and not r1000.empty:
                s=r959.iloc[0]; e=r1000.iloc[0]; side='LONG' if float(s.close_mid)>prev_close else 'SHORT'
                sig['prev_0959_to_1000'].append({'date':date,'year':year,'wd':wd,'side':side,'entry_i':int(e.global_i),
                    'entry':float(e.open_ask if side=='LONG' else e.open_bid),'signal_px':float(s.close_mid),'prev_close':prev_close})
            if not r1000.empty and not r1001.empty:
                s=r1000.iloc[0]; e=r1001.iloc[0]; side='LONG' if float(s.close_mid)>prev_close else 'SHORT'
                sig['prev_1000close_to_1001'].append({'date':date,'year':year,'wd':wd,'side':side,'entry_i':int(e.global_i),
                    'entry':float(e.open_ask if side=='LONG' else e.open_bid),'signal_px':float(s.close_mid),'prev_close':prev_close})
        # London 10:00 open vs US 15:30 open: if US > London => SHORT, else LONG.
        lo=g[g['mod']==600]; us=g[g['mod']==930]
        if not lo.empty and not us.empty:
            L=float(lo.iloc[0].open_mid); U=float(us.iloc[0].open_mid); side='SHORT' if U>L else 'LONG'; e=us.iloc[0]
            sig['london_us_meanrev'].append({'date':date,'year':year,'wd':wd,'side':side,'entry_i':int(e.global_i),
                'entry':float(e.open_ask if side=='LONG' else e.open_bid),'london_open':L,'us_open':U,'disp':abs(U-L)})
        if len(g): prev_close=float(g.iloc[-1].close_mid)
    return sig


def score(m):
    pos=sum(1 for y in TRAIN if m['years'].get(y,{}).get('R',0)>0)
    return m['evR']*math.sqrt(max(1,m['n'])) + .18*(m['pf']-1) + .07*pos - .012*m['ddR']


def run_family(df,name,signals,thresholds=(0.0,)):
    rows=[]
    for th in thresholds:
        ss=[c for c in signals if float(c.get('disp',999999))>=th]
        for tp in TPS:
            for sl in SLS:
                for h in HORIZONS:
                    trds=[trade(df,c,tp,sl,h) for c in ss]; trds=[x for x in trds if x]
                    tr=metrics(trds,TRAIN); oo=metrics(trds,OOS); al=metrics(trds,ALL)
                    if not tr or not oo:continue
                    pos_tr=sum(1 for y in TRAIN if tr['years'].get(y,{}).get('R',0)>0)
                    pos_oo=sum(1 for y in OOS if oo['years'].get(y,{}).get('R',0)>0)
                    rec={'family':name,'threshold':th,'tp':tp,'sl':sl,'rr':tp/sl,'horizon_min':h,
                         'train_score':score(tr),'train_n':tr['n'],'train_R':tr['R'],'train_evR':tr['evR'],'train_wr':tr['wr'],'train_pf':tr['pf'],'train_ddR':tr['ddR'],'train_streak':tr['streak'],'train_pos_years':pos_tr,
                         'oos_n':oo['n'],'oos_R':oo['R'],'oos_evR':oo['evR'],'oos_wr':oo['wr'],'oos_pf':oo['pf'],'oos_ddR':oo['ddR'],'oos_streak':oo['streak'],'oos_pos_years':pos_oo,
                         'all_n':al['n'],'all_R':al['R'],'all_evR':al['evR'],'all_wr':al['wr'],'all_pf':al['pf'],'all_ddR':al['ddR']}
                    for y in range(2018,2027):
                        yy=al['years'].get(y,{}); rec[f'n_{y}']=yy.get('n',0); rec[f'R_{y}']=yy.get('R',0.); rec[f'wr_{y}']=yy.get('wr',np.nan)
                    rows.append(rec)
    q=pd.DataFrame(rows).sort_values('train_score',ascending=False)
    q.to_csv(OUT/f'{name}_grid.csv',index=False)
    return q


def shortlist(q,name):
    # Frozen candidate choice from TRAIN only; OOS is reported, not used to choose.
    eligible=q[(q.train_n>=400)&(q.train_pos_years>=4)&(q.train_pf>1.0)].copy()
    top=eligible.sort_values('train_score',ascending=False).head(15)
    top.to_csv(OUT/f'{name}_train_shortlist.csv',index=False)
    return top


def main():
    print('Loading data...',flush=True); df=p.load(); sig=build_signals(df)
    print({k:len(v) for k,v in sig.items()},flush=True)
    q1=run_family(df,'prev_0959_to_1000',sig['prev_0959_to_1000'])
    q2=run_family(df,'prev_1000close_to_1001',sig['prev_1000close_to_1001'])
    # exact strategy threshold=0 plus modest displacement filters as robustness variants
    q3=run_family(df,'london_us_meanrev',sig['london_us_meanrev'],thresholds=(0.,2.,5.,10.))
    t1=shortlist(q1,'prev_0959_to_1000'); t2=shortlist(q2,'prev_1000close_to_1001'); t3=shortlist(q3,'london_us_meanrev')
    exact3=shortlist(q3[q3.threshold==0].copy(),'london_us_meanrev_exact')
    def best(t):
        if t.empty:return None
        return t.iloc[0].replace({np.nan:None}).to_dict()
    summary={'signal_counts':{k:len(v) for k,v in sig.items()},
      'method':'TP/SL/horizon selected from 2018-2022 TRAIN only. 2023-2026 OOS shown only after shortlist. BID/ASK spread included; same-M1 TP+SL resolves to SL.',
      'best_train_prev_0959_entry_1000':best(t1),
      'best_train_prev_1000close_entry_1001':best(t2),
      'best_train_london_us_with_displacement_filters':best(t3),
      'best_train_london_us_exact_no_filter':best(exact3)}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=True)
    print(json.dumps(summary,indent=2,allow_nan=True),flush=True)

if __name__=='__main__':main()

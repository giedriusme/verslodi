import json, math, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p

OUT=Path('new_london_us_results'); OUT.mkdir(exist_ok=True)
YEARS=list(range(2018,2027)); TRAIN=set(range(2018,2023)); OOS=set(range(2023,2027));
TPS=[4.,5.,6.,8.,10.,12.,15.,20.,25.,30.]
SLS=[5.,8.,10.,12.,15.,20.,25.,30.]
HORIZONS=[120,240,480,1440]
VALUE=100.0; START=300.0; RISK=.10

def lot_size(eq,sl):
    raw=(eq*RISK)/(sl*VALUE)
    return round(max(0,math.floor((raw+1e-12)/.01))*.01,2)

def mid_close(r): return (float(r.close_bid)+float(r.close_ask))/2

def mid_open(r): return (float(r.open_bid)+float(r.open_ask))/2

def trade_path(df,entry_i,side,h):
    f=df.iloc[entry_i:min(len(df),entry_i+h)]
    if len(f)==0:return None
    if side=='LONG':
        e=float(df.iloc[entry_i].open_ask); fav=f.high_bid.to_numpy(float)-e; adv=e-f.low_bid.to_numpy(float); mark=float(f.iloc[-1].close_bid-e)
    else:
        e=float(df.iloc[entry_i].open_bid); fav=e-f.low_ask.to_numpy(float); adv=f.high_ask.to_numpy(float)-e; mark=float(e-f.iloc[-1].close_ask)
    return {'entry':e,'fav':fav,'adv':adv,'mark':mark}

def outcome(path,tp,sl):
    a=np.flatnonzero(path['fav']>=tp); b=np.flatnonzero(path['adv']>=sl)
    ia=int(a[0]) if len(a) else None; ib=int(b[0]) if len(b) else None
    if ib is not None and (ia is None or ib<=ia):return -sl,'SL'
    if ia is not None:return tp,'TP'
    return float(path['mark']),'TIME'

def build_daily_index(df):
    dt=pd.to_datetime(df.dt_local)
    dates=dt.dt.date.astype(str)
    hm=dt.dt.hour*60+dt.dt.minute
    groups={}
    for i,(d,m) in enumerate(zip(dates,hm)):
        groups.setdefault(d,{})[int(m)]=i
    # Prior trading-day close = last available M1 quote of prior local trading date.
    unique=[]; last_by={}
    for i,d in enumerate(dates):
        last_by[d]=i
    unique=sorted(last_by)
    prev={}
    for j,d in enumerate(unique):
        if j>0: prev[d]=last_by[unique[j-1]]
    return groups,prev

def build_signals(df):
    groups,prev=build_daily_index(df)
    sigs={'prevclose_0959_to_1000':[], 'prevclose_1000close_to_1001':[], 'london_us_same1530':[], 'london_us_1531':[]}
    for d,g in groups.items():
        y=int(d[:4])
        if y not in YEARS or d not in prev:continue
        pi=prev[d]; pc=mid_close(df.iloc[pi])
        # Variant 1: 09:59 close vs previous trading-day close -> enter 10:00 open.
        if 9*60+59 in g and 10*60 in g:
            si=g[9*60+59]; ei=g[10*60]; px=mid_close(df.iloc[si])
            if px!=pc:
                sigs['prevclose_0959_to_1000'].append({'date':d,'year':y,'side':'LONG' if px>pc else 'SHORT','entry_i':ei,'prev_close':pc,'signal_price':px})
        # Variant 2: 10:00 candle close -> enter 10:01 open.
        if 10*60 in g and 10*60+1 in g:
            si=g[10*60]; ei=g[10*60+1]; px=mid_close(df.iloc[si])
            if px!=pc:
                sigs['prevclose_1000close_to_1001'].append({'date':d,'year':y,'side':'LONG' if px>pc else 'SHORT','entry_i':ei,'prev_close':pc,'signal_price':px})
        # London open 10:00 vs US open 15:30. Contrarian: US>London => SHORT, US<London => LONG.
        if 10*60 in g and 15*60+30 in g:
            li=g[10*60]; ui=g[15*60+30]
            lo=mid_open(df.iloc[li]); uo=mid_open(df.iloc[ui])
            if uo!=lo:
                side='SHORT' if uo>lo else 'LONG'
                sigs['london_us_same1530'].append({'date':d,'year':y,'side':side,'entry_i':ui,'london_open':lo,'us_open':uo})
                if 15*60+31 in g:
                    sigs['london_us_1531'].append({'date':d,'year':y,'side':side,'entry_i':g[15*60+31],'london_open':lo,'us_open':uo})
    return sigs

def eval_combo(paths,signals,tp,sl,h,years,compound=False):
    vals=[]; wins=losses=times=0; eq=START; peak=eq; dd=0.; stalled=False
    yr={}
    for j,s in enumerate(signals):
        if s['year'] not in years:continue
        pts,state=outcome(paths[h][j],tp,sl)
        vals.append(pts)
        wins+=pts>0; losses+=pts<0; times+=state=='TIME'
        if compound:
            lot=lot_size(eq,sl)
            if lot<.01:
                stalled=True; continue
            pnl=pts*VALUE*lot; eq=max(0.,eq+pnl); peak=max(peak,eq); dd=max(dd,(peak-eq)/peak if peak else 1.)
            y=s['year']; yr.setdefault(y,{'start':eq-pnl,'end':eq})['end']=eq
    arr=np.array(vals,float)
    if len(arr)==0:return None
    grossp=arr[arr>0].sum(); grossl=-arr[arr<0].sum()
    return {'n':len(arr),'net_points':float(arr.sum()),'ev_points':float(arr.mean()),'wr':float((arr>0).mean()),'pf':float(grossp/grossl) if grossl>0 else 999.,'time_exits':times,
            'compound_final':eq if compound else None,'compound_dd':dd if compound else None,'stalled':stalled if compound else None}

def main():
    df=p.load(); sigs=build_signals(df)
    summary={'signal_counts':{k:len(v) for k,v in sigs.items()}}
    allrows=[]; picks={}
    for name,signals in sigs.items():
        print(name,len(signals),flush=True)
        paths={h:[trade_path(df,int(s['entry_i']),s['side'],h) for s in signals] for h in HORIZONS}
        trainrows=[]
        for tp,sl,h in itertools.product(TPS,SLS,HORIZONS):
            z=eval_combo(paths,signals,tp,sl,h,TRAIN)
            if not z or z['n']<250:continue
            # Train-only ranking rewards expectancy + PF and penalizes very low sample quality.
            score=z['ev_points']*(1+min(z['pf'],3)/3)
            trainrows.append({'strategy':name,'tp':tp,'sl':sl,'horizon_m1':h,'train_score':score,**{f'train_{k}':v for k,v in z.items() if k not in ('compound_final','compound_dd','stalled')}})
        tg=pd.DataFrame(trainrows).sort_values('train_score',ascending=False)
        tg.to_csv(OUT/f'{name}_train_grid.csv',index=False)
        # Only top train candidates are exposed to OOS.
        outs=[]
        for r in tg.head(60).to_dict('records'):
            tp=float(r['tp']); sl=float(r['sl']); h=int(r['horizon_m1'])
            oo=eval_combo(paths,signals,tp,sl,h,OOS); fu=eval_combo(paths,signals,tp,sl,h,set(YEARS)); comp=eval_combo(paths,signals,tp,sl,h,OOS,compound=True)
            rec=dict(r)
            rec.update({f'oos_{k}':v for k,v in oo.items() if k not in ('compound_final','compound_dd','stalled')})
            rec.update({'oos_compound_final':comp['compound_final'],'oos_compound_dd':comp['compound_dd'],'oos_stalled':comp['stalled'],'full_net_points':fu['net_points'],'full_ev_points':fu['ev_points'],'full_wr':fu['wr'],'full_pf':fu['pf']})
            # Robust preference: positive OOS expectancy/PF, then growth/DD.
            rec['robust_score']=(max(0.,oo['ev_points'])*(1+min(oo['pf'],3))) + (math.log(max(comp['compound_final']/START,1e-9)) if comp['compound_final'] else -20) - 1.5*comp['compound_dd']
            outs.append(rec)
        q=pd.DataFrame(outs).sort_values('robust_score',ascending=False)
        q.to_csv(OUT/f'{name}_oos_selected.csv',index=False)
        robust=q[(q.oos_ev_points>0)&(q.oos_pf>1.0)&(~q.oos_stalled)].copy()
        robust.to_csv(OUT/f'{name}_robust.csv',index=False)
        picks[name]=robust.head(10).to_dict('records')
        summary[name]={'best_robust':picks[name][:5],'robust_count':int(len(robust))}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=True)
    print(json.dumps(summary,indent=2,allow_nan=True))

if __name__=='__main__':main()

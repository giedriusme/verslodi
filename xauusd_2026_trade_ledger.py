import io
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

OUT = Path('trade_ledger_2026')
OUT.mkdir(exist_ok=True)
RAW = 'https://raw.githubusercontent.com/kevingtlin/Market-Data-Lab/main'
TZ = ZoneInfo('Europe/Vilnius')
# Need Nov-Dec 2025 to seed 20-day Asia median and prior-day close for Jan 2026.
MONTHS = [(2025,11),(2025,12)] + [(2026,m) for m in range(1,9)]

S = requests.Session()
S.headers.update({'User-Agent':'xauusd-2026-ledger/1.0'})


def month(side,y,m):
    u=f'{RAW}/xauusd/{side}/m1/xauusd_{side}_m1_{y:04d}_{m:02d}.csv'
    r=S.get(u,timeout=90)
    r.raise_for_status()
    d=pd.read_csv(io.StringIO(r.text))
    ren={c:f'{c}_{side}' for c in ['open','high','low','close']}
    return d.rename(columns=ren)


def load():
    parts=[]
    for y,m in MONTHS:
        b=month('bid',y,m); a=month('ask',y,m)
        d=b.merge(a,on='timestamp',how='inner')
        parts.append(d)
    df=pd.concat(parts,ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp')
    df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms',utc=True)
    df['local']=df['timestamp'].dt.tz_convert(TZ)
    df['date']=df['local'].dt.date.astype(str)
    df['year']=df['local'].dt.year.astype(int)
    df['wd']=df['local'].dt.dayofweek.astype(int)
    df['hour']=df['local'].dt.hour.astype(int)
    df['minute']=df['local'].dt.minute.astype(int)
    df['mod']=df['hour']*60+df['minute']
    for c in ['open','high','low','close']:
        df[f'{c}_mid']=(df[f'{c}_bid']+df[f'{c}_ask'])/2
    df=df.reset_index(drop=True)
    df['global_i']=np.arange(len(df))
    return df


def first_close_break(g,start,end,H,L):
    post=g[(g['mod']>=start)&(g['mod']<end)]
    for idx,r in post.iterrows():
        if float(r.close_mid)>H:
            return idx,'LONG'
        if float(r.close_mid)<L:
            return idx,'SHORT'
    return None,None


def mk_trade(g,signal_idx,side,strategy,tp,sl,horizon):
    loc=g.index.get_loc(signal_idx)
    if loc+1>=len(g): return None
    e=g.iloc[loc+1]
    return {
        'strategy':strategy,
        'entry_date':str(e['date']),
        'entry_time':e['local'].strftime('%H:%M'),
        'entry_i':int(e['global_i']),
        'side':side,
        'entry':float(e.open_ask if side=='LONG' else e.open_bid),
        'tp':float(tp),'sl':float(sl),'horizon':int(horizon)
    }


def evaluate(df,t):
    i=t['entry_i']; e=t['entry']; side=t['side']; tp=t['tp']; sl=t['sl']; h=t['horizon']
    f=df.iloc[i:min(len(df),i+h)]
    if f.empty: return None
    if side=='LONG':
        a=np.flatnonzero(f.high_bid.to_numpy(float)>=e+tp)
        b=np.flatnonzero(f.low_bid.to_numpy(float)<=e-sl)
        mark=float(f.iloc[-1].close_bid-e)
    else:
        a=np.flatnonzero(f.low_ask.to_numpy(float)<=e-tp)
        b=np.flatnonzero(f.high_ask.to_numpy(float)>=e+sl)
        mark=float(e-f.iloc[-1].close_ask)
    ia=int(a[0]) if len(a) else None
    ib=int(b[0]) if len(b) else None
    if ia is not None and (ib is None or ia<ib):
        pnl=tp; outcome='TP'; exit_i=i+ia
    elif ib is not None:
        pnl=-sl; outcome='SL'; exit_i=i+ib
    else:
        pnl=mark; outcome='TIME'; exit_i=int(f.index[-1])
    ex=df.iloc[exit_i]
    out=dict(t)
    out.update({'result':float(pnl),'outcome':outcome,'exit_date':str(ex['date']),'exit_time':ex['local'].strftime('%H:%M')})
    return out


def main():
    df=load()
    days=[]
    for day,g0 in df.groupby('date',sort=True):
        if pd.Timestamp(day).dayofweek<5:
            days.append((day,g0.sort_values('timestamp').reset_index(drop=True)))

    asia_hist=[]
    prev_close=None
    trades=[]

    for day,g in days:
        asia=g[(g['mod']>=60)&(g['mod']<540)]
        if len(asia)<420:
            if len(g): prev_close=float(g.iloc[-1].close_mid)
            continue
        AH=float(asia.high_mid.max()); AL=float(asia.low_mid.min()); AR=AH-AL
        med20=float(np.median(asia_hist[-20:])) if len(asia_hist)>=20 else np.nan
        wd=int(g.iloc[0].wd)

        if int(g.iloc[0].year)==2026:
            if np.isfinite(med20) and med20>0 and AR<=0.70*med20:
                idx,side=first_close_break(g,540,960,AH,AL)
                if side:
                    t=mk_trade(g,idx,side,'CORE A - Asia Compression Breakout',15,15,1440)
                    if t: trades.append(t)

            if wd in (3,4):
                idx,side=first_close_break(g,600,960,AH,AL)
                if side:
                    t=mk_trade(g,idx,side,'CORE B - Thu/Fri Asia Close Breakout',20,20,480)
                    if t: trades.append(t)

            ss=g[g['mod']==840]
            if prev_close is not None and not ss.empty:
                idx=int(ss.index[0]); px=float(ss.iloc[0].close_mid)
                if px!=prev_close:
                    side='LONG' if px>prev_close else 'SHORT'
                    t=mk_trade(g,idx,side,'Satellite C - 14:00 Previous Close Continuation',30,25,1440)
                    if t: trades.append(t)

        asia_hist.append(AR)
        prev_close=float(g.iloc[-1].close_mid)

    results=[]
    for t in trades:
        r=evaluate(df,t)
        if r: results.append(r)
    rr=pd.DataFrame(results).sort_values(['entry_date','strategy']).reset_index(drop=True)
    rr.to_csv(OUT/'trade_details.csv',index=False)

    names=[
      'CORE A - Asia Compression Breakout',
      'CORE B - Thu/Fri Asia Close Breakout',
      'Satellite C - 14:00 Previous Close Continuation'
    ]
    piv=rr.pivot_table(index='entry_date',columns='strategy',values='result',aggfunc='first').reindex(columns=names)
    piv=piv.sort_index()
    piv.to_csv(OUT/'calendar_results.csv')

    human=piv.copy()
    human.index=[pd.Timestamp(x).strftime('%m.%d') for x in human.index]
    def fmt(v):
        if pd.isna(v): return ''
        if abs(v-round(v))<1e-9: return f'{int(round(v)):+d}'
        return f'{v:+.2f}'
    for c in human.columns: human[c]=human[c].map(fmt)
    human.index.name='date'
    human.to_csv(OUT/'calendar_results_human.csv')

    summary=[]
    for n in names:
        x=rr[rr.strategy==n]
        summary.append({
          'strategy':n,'trades':len(x),'positive':int((x.result>0).sum()),'negative':int((x.result<0).sum()),
          'tp':int((x.outcome=='TP').sum()),'sl':int((x.outcome=='SL').sum()),'time':int((x.outcome=='TIME').sum()),
          'net_points':float(x.result.sum())
        })
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(human.to_string())

if __name__=='__main__':
    main()

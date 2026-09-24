import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_backtest as bt
from sklearn.ensemble import HistGradientBoostingClassifier

OUT=Path('intraday_oos_results');OUT.mkdir(exist_ok=True)
TIMES=[540,600,660,780,840,900]
TPSL=[(10.,10.),(15.,15.),(20.,20.),(15.,10.),(20.,15.),(25.,20.),(30.,25.)]
THRESHOLDS=np.arange(.50,.81,.02)


def feat(g,tm,prev):
    ss=g[g['mod']==tm]
    if ss.empty:return None
    idx=int(ss.index[0]);r=ss.iloc[0]
    asia=g[(g['mod']>=60)&(g['mod']<540)]
    if len(asia)<420:return None
    H=float(asia.high_mid.max());L=float(asia.low_mid.min());R=max(H-L,1e-6);O=float(asia.iloc[0].open_mid);C=float(asia.iloc[-1].close_mid);px=float(r.close_mid)
    x=[R,(C-O)/R,(px-L)/R, math.sin(2*math.pi*tm/1440),math.cos(2*math.pi*tm/1440)]
    for mins in [15,30,60,120,240]:
        w=g[(g['mod']>=tm-mins)&(g['mod']<tm)]
        if len(w)<max(8,int(mins*.6)):return None
        op=float(w.iloc[0].open_mid);cl=float(w.iloc[-1].close_mid);hi=float(w.high_mid.max());lo=float(w.low_mid.min());rr=max(hi-lo,1e-6)
        x += [cl-op,rr,(cl-op)/rr,(cl-lo)/rr]
    x += [float(px-r.ema20),float(px-r.ema60),float(px-r.ema240),float(r.ema20-r.ema60),float(r.ema60-r.ema240),float(r.rsi),float(r.z60),float(r.r60)]
    if prev:
        po,pc,ph,pl=prev;pr=max(ph-pl,1e-6);x += [pc-po,pr,px-pc,(pc-po)/pr,(px-pl)/pr]
    else:x += [0,0,0,0,0]
    wd=pd.Timestamp(g.iloc[0].local_date).dayofweek;x += [math.sin(2*math.pi*wd/5),math.cos(2*math.pi*wd/5)]
    return idx,np.asarray(x,float)

def outcome(g,pos,d,tp,sl):
    if pos+1>=len(g):return None
    e=g.iloc[pos+1];tm=int(e['mod']);end=min(1380,tm+480);f=g[(g['mod']>=tm)&(g['mod']<end)]
    if f.empty:return None
    entry=float(e.open_ask if d=='LONG' else e.open_bid)
    if d=='LONG':a=np.flatnonzero(f.high_bid.to_numpy(float)>=entry+tp);b=np.flatnonzero(f.low_bid.to_numpy(float)<=entry-sl);last=float(f.iloc[-1].close_bid-entry)
    else:a=np.flatnonzero(f.low_bid.to_numpy(float)<=entry-tp);b=np.flatnonzero(f.high_ask.to_numpy(float)>=entry+sl);last=float(entry-f.iloc[-1].close_ask)
    ia=int(a[0]) if len(a) else None;ib=int(b[0]) if len(b) else None
    if ia is not None and (ib is None or ia<ib):return 1,float(tp),'TP'
    if ib is not None:return 0,-float(sl),'SL'
    return int(last>0),last,'TIME'

def streak(vals):
    b=c=0
    for x in vals:
        if x<0:c+=1;b=max(b,c)
        else:c=0
    return b

def metrics(rows):
    if not rows:return None
    r=pd.DataFrame(rows).sort_values(['date','tm']).reset_index(drop=True);p=r.pnl.to_numpy(float);g=p[p>0].sum();l=-p[p<0].sum();eq=np.cumsum(p);pk=np.maximum.accumulate(np.r_[0.,eq]);dd=pk[1:]-eq
    yd={}
    for y,z in r.groupby('year'):
        q=z.pnl.to_numpy(float);yd[int(y)]={'n':len(z),'net':float(q.sum()),'pos':int((q>0).sum()),'neg':int((q<0).sum()),'tp':int((z.outcome=='TP').sum()),'sl':int((z.outcome=='SL').sum()),'time':int((z.outcome=='TIME').sum())}
    return {'n':len(r),'pos':int((p>0).sum()),'neg':int((p<0).sum()),'wr':float((p>0).mean()),'net':float(p.sum()),'ev':float(p.mean()),'pf':float(g/l) if l>0 else 99.,'dd':float(dd.max()),'streak':streak(p),'years':yd}

def simulate(samples,probl,probs,tp,sl,th,years):
    # at most one trade/day: first chronological signal whose confidence clears threshold
    rows=[];taken=set()
    for i,s in enumerate(samples):
        if s['year'] not in years or s['date'] in taken:continue
        conf=max(probl[i],probs[i])
        if conf<th:continue
        d='LONG' if probl[i]>=probs[i] else 'SHORT';o=outcome(s['g'],s['pos'],d,tp,sl)
        if o is None:continue
        rows.append({'date':s['date'],'year':s['year'],'tm':s['tm'],'pnl':o[1],'outcome':o[2],'confidence':conf,'direction':d});taken.add(s['date'])
    return metrics(rows)

def main():
    bt.YEARS_MONTHS=[(y,m) for y in range(2018,2027) for m in range(1,13) if (y<2026 or m<=8)]
    df=bt.load_data().sort_values('timestamp').reset_index(drop=True);df['mod']=(df.hour*60+df.minute).astype(int)
    df['ema20']=df.close_mid.ewm(span=20,adjust=False).mean();df['ema60']=df.close_mid.ewm(span=60,adjust=False).mean();df['ema240']=df.close_mid.ewm(span=240,adjust=False).mean()
    d=df.close_mid.diff();u=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean();df['rsi']=100-100/(1+u/dn.replace(0,np.nan));ma=df.close_mid.rolling(60,min_periods=45).mean();sd=df.close_mid.rolling(60,min_periods=45).std();df['z60']=(df.close_mid-ma)/sd.replace(0,np.nan);df['r60']=df.high_mid.rolling(60,min_periods=45).max()-df.low_mid.rolling(60,min_periods=45).min()
    samples=[];prev=None
    for day,g0 in df.groupby('local_date',sort=True):
        if pd.Timestamp(day).dayofweek>=5:continue
        g=g0.reset_index(drop=True)
        for tm in TIMES:
            z=feat(g,tm,prev)
            if z is not None:samples.append({'date':str(day),'year':int(g.iloc[0].year),'tm':tm,'pos':z[0],'x':z[1],'g':g})
        prev=(float(g.iloc[0].open_mid),float(g.iloc[-1].close_mid),float(g.high_mid.max()),float(g.low_mid.min()))
    X=np.vstack([s['x'] for s in samples]);Y=np.array([s['year'] for s in samples]);rows=[]
    train0=Y<=2020;val=(Y>=2021)&(Y<=2022);pre=Y<=2022;test=Y>=2023
    for tp,sl in TPSL:
        yl=[];ys=[];valid=[]
        for i,s in enumerate(samples):
            a=outcome(s['g'],s['pos'],'LONG',tp,sl);b=outcome(s['g'],s['pos'],'SHORT',tp,sl)
            if a is None or b is None:continue
            valid.append(i);yl.append(a[0]);ys.append(b[0])
        V=np.asarray(valid);XV=X[V];yv=Y[V];yl=np.asarray(yl);ys=np.asarray(ys)
        tr0=yv<=2020;va=(yv>=2021)&(yv<=2022);pr=yv<=2022;te=yv>=2023
        def model(seed):return HistGradientBoostingClassifier(max_depth=3,max_iter=140,learning_rate=.04,l2_regularization=3.,min_samples_leaf=35,random_state=seed)
        ml=model(17);ms=model(31);ml.fit(XV[tr0],yl[tr0]);ms.fit(XV[tr0],ys[tr0]);plv=ml.predict_proba(XV)[:,1];psv=ms.predict_proba(XV)[:,1]
        # choose threshold on 2021-22 only: frequency >=100/year average and best EV, with soft preference WR.
        choices=[]
        sub=[samples[i] for i in V]
        for th in THRESHOLDS:
            m=simulate(sub,plv,psv,tp,sl,float(th),{2021,2022})
            if not m:continue
            yd=m['years'];freq=yd.get(2021,{}).get('n',0)>=100 and yd.get(2022,{}).get('n',0)>=100
            if freq:choices.append((m['ev']+.15*(m['wr']-.5)*100,float(th),m))
        if not choices:continue
        choices.sort(key=lambda x:x[0],reverse=True);th=choices[0][1];valm=choices[0][2]
        # retrain all 2018-22, untouched test 2023-26
        ml=model(17);ms=model(31);ml.fit(XV[pr],yl[pr]);ms.fit(XV[pr],ys[pr]);pl=ml.predict_proba(XV)[:,1];ps=ms.predict_proba(XV)[:,1]
        tm=simulate(sub,pl,ps,tp,sl,th,{2023,2024,2025,2026})
        if not tm:continue
        yd=tm['years'];freq=all(yd.get(y,{}).get('n',0)>=100 for y in [2023,2024,2025]) and yd.get(2026,{}).get('n',0)>50;ally=all(yd.get(y,{}).get('net',-1)>0 for y in [2023,2024,2025,2026])
        row={'tp':tp,'sl':sl,'rr':tp/sl,'threshold':th,'val_wr':valm['wr'],'val_ev':valm['ev'],'freq_ok':freq,'all_years_positive':ally,**{k:v for k,v in tm.items() if k!='years'},**{f'n_{y}':yd.get(y,{}).get('n',0) for y in [2023,2024,2025,2026]},**{f'net_{y}':yd.get(y,{}).get('net',0) for y in [2023,2024,2025,2026]}}
        rows.append(row)
    R=pd.DataFrame(rows);R.to_csv(OUT/'results.csv',index=False)
    strict=R[(R.freq_ok)&(R.all_years_positive)&(R.wr>=.65)&(R.streak<=3)] if len(R) else R;near=R[(R.freq_ok)&(R.all_years_positive)&(R.wr>=.65)&(R.streak.between(4,5))] if len(R) else R;front=R[(R.freq_ok)&(R.all_years_positive)].sort_values(['wr','ev'],ascending=False) if len(R) else R
    summary={'tested':len(R),'strict_count':len(strict),'near_count':len(near),'strict':strict.to_dict('records'),'near':near.to_dict('records'),'frontier':front.to_dict('records')}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print('INTRADAY_OOS_DONE');print(R.to_string(index=False))
if __name__=='__main__':main()

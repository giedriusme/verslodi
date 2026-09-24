import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_backtest as bt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUT=Path('ml_oos_results');OUT.mkdir(exist_ok=True)
TRAIN_END=2022
TEST_YEARS=[2023,2024,2025,2026]
TPSL=[(10.,10.),(15.,15.),(20.,20.),(20.,15.),(25.,20.),(30.,25.)]
TIMES=[600,840]
THRESHOLDS=[.50,.52,.54,.56,.58,.60,.62,.64,.66,.68,.70]


def outcome_path(g,pos,d,tp,sl,horizon=720):
    if pos+1>=len(g):return None
    e=g.iloc[pos+1];f=g.iloc[pos+1:min(len(g),pos+1+horizon)]
    if f.empty:return None
    entry=float(e.open_ask if d=='LONG' else e.open_bid)
    if d=='LONG':
        a=np.flatnonzero(f.high_bid.to_numpy(float)>=entry+tp);b=np.flatnonzero(f.low_bid.to_numpy(float)<=entry-sl);last=float(f.iloc[-1].close_bid-entry)
    else:
        a=np.flatnonzero(f.low_bid.to_numpy(float)<=entry-tp);b=np.flatnonzero(f.high_bid.to_numpy(float)>=entry+sl);last=float(entry-f.iloc[-1].close_ask)
    ia=int(a[0]) if len(a) else None;ib=int(b[0]) if len(b) else None
    if ia is not None and (ib is None or ia<ib):return 1,float(tp),'TP'
    if ib is not None:return 0,-float(sl),'SL'
    return int(last>0),last,'TIME'

def features(g,tm,prev):
    ss=g[g['mod']==tm]
    if ss.empty:return None
    idx=int(ss.index[0]);r=ss.iloc[0]
    asia=g[(g['mod']>=60)&(g['mod']<540)]
    if len(asia)<420:return None
    AH=float(asia.high_mid.max());AL=float(asia.low_mid.min());AR=AH-AL
    if AR<=0:return None
    AO=float(asia.iloc[0].open_mid);AC=float(asia.iloc[-1].close_mid);px=float(r.close_mid)
    out=[AR,(AC-AO)/AR,(px-AL)/AR]
    for mins in [15,30,60,120,240]:
        w=g[(g['mod']>=tm-mins)&(g['mod']<tm)]
        if len(w)<max(8,int(mins*.6)):return None
        op=float(w.iloc[0].open_mid);cl=float(w.iloc[-1].close_mid);hi=float(w.high_mid.max());lo=float(w.low_mid.min());rr=max(hi-lo,1e-6)
        out += [cl-op,rr,(cl-op)/rr,(cl-lo)/rr]
    out += [float(r.close_mid-r.ema20),float(r.close_mid-r.ema60),float(r.close_mid-r.ema240),float(r.ema20-r.ema60),float(r.ema60-r.ema240),float(r.rsi),float(r.z60),float(r.r60)]
    if prev is None: out += [0.,0.,0.,0.]
    else:
        po,pc,ph,pl=prev;pr=max(ph-pl,1e-6);out += [pc-po,pr,px-pc,(pc-po)/pr]
    # cyclical weekday
    wd=pd.Timestamp(g.iloc[0].local_date).dayofweek;out += [math.sin(2*math.pi*wd/5),math.cos(2*math.pi*wd/5)]
    return idx,np.asarray(out,float),AR

def streak(pnls):
    cur=best=0
    for x in pnls:
        if x<0:cur+=1;best=max(best,cur)
        else:cur=0
    return best

def summarize(rows):
    r=pd.DataFrame(rows).sort_values('date').reset_index(drop=True);p=r.pnl.to_numpy(float)
    gains=p[p>0].sum();loss=-p[p<0].sum();eq=np.cumsum(p);peak=np.maximum.accumulate(np.r_[0.,eq]);dd=peak[1:]-eq
    yd={}
    for y,g in r.groupby('year'):
        py=g.pnl.to_numpy(float);yd[int(y)]={'n':len(g),'net':float(py.sum()),'pos':int((py>0).sum()),'neg':int((py<0).sum()),'tp':int((g.outcome=='TP').sum()),'sl':int((g.outcome=='SL').sum()),'time':int((g.outcome=='TIME').sum())}
    return {'n':len(r),'positive':int((p>0).sum()),'negative':int((p<0).sum()),'wr':float((p>0).mean()),'net':float(p.sum()),'ev':float(p.mean()),'pf':float(gains/loss) if loss>0 else 99.,'dd':float(dd.max()) if len(dd) else 0.,'streak':streak(p),'years':yd}

def main():
    # Extend source download backwards by overriding start/end globals in bt if available.
    bt.START=pd.Timestamp('2018-01-01',tz='UTC') if hasattr(bt,'START') else None
    # loader in original script uses constants; monkey-patch explicit date strings when present
    for name,val in [('START_MONTH','2018-01'),('START_DATE','2018-01-01')]:
        if hasattr(bt,name):setattr(bt,name,val)
    df=bt.load_data()
    # If loader stayed at 2023, fail clearly so workflow can be adjusted.
    if int(df.year.min())>2018:
        raise RuntimeError(f'Need pre-2023 data but loader starts {int(df.year.min())}')
    df=df.sort_values('timestamp').reset_index(drop=True);df['mod']=(df.hour*60+df.minute).astype(int)
    df['ema20']=df.close_mid.ewm(span=20,adjust=False).mean();df['ema60']=df.close_mid.ewm(span=60,adjust=False).mean();df['ema240']=df.close_mid.ewm(span=240,adjust=False).mean()
    d=df.close_mid.diff();u=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean();df['rsi']=100-100/(1+u/dn.replace(0,np.nan))
    ma=df.close_mid.rolling(60,min_periods=45).mean();sd=df.close_mid.rolling(60,min_periods=45).std();df['z60']=(df.close_mid-ma)/sd.replace(0,np.nan);df['r60']=df.high_mid.rolling(60,min_periods=45).max()-df.low_mid.rolling(60,min_periods=45).min()
    samples=[];prev=None
    for day,g0 in df.groupby('local_date',sort=True):
        if pd.Timestamp(day).dayofweek>=5:continue
        g=g0.reset_index(drop=True)
        for tm in TIMES:
            z=features(g,tm,prev)
            if z is None:continue
            pos,x,AR=z
            rec={'date':str(day),'year':int(g.iloc[0].year),'tm':tm,'x':x,'g':g,'pos':pos}
            samples.append(rec)
        prev=(float(g.iloc[0].open_mid),float(g.iloc[-1].close_mid),float(g.high_mid.max()),float(g.low_mid.min()))
    # Model each time and TP/SL separately. Two classifiers: P(long win), P(short win).
    results=[]
    for tm in TIMES:
      ss=[s for s in samples if s['tm']==tm]
      X=np.vstack([s['x'] for s in ss]);years=np.array([s['year'] for s in ss]);
      for tp,sl in TPSL:
        yl=[];ys=[]
        valid=[]
        for i,s in enumerate(ss):
            ol=outcome_path(s['g'],s['pos'],'LONG',tp,sl);os=outcome_path(s['g'],s['pos'],'SHORT',tp,sl)
            if ol is None or os is None:continue
            valid.append(i);yl.append(ol[0]);ys.append(os[0])
        V=np.asarray(valid);XV=X[V];Y=years[V];yl=np.asarray(yl);ys=np.asarray(ys)
        tr=Y<=TRAIN_END;te=Y>=2023
        if tr.sum()<500 or te.sum()<300:continue
        # shallow regularized boosting to avoid memorization
        ml=HistGradientBoostingClassifier(max_depth=3,max_iter=120,learning_rate=.05,l2_regularization=2.,min_samples_leaf=30,random_state=11)
        ms=HistGradientBoostingClassifier(max_depth=3,max_iter=120,learning_rate=.05,l2_regularization=2.,min_samples_leaf=30,random_state=29)
        ml.fit(XV[tr],yl[tr]);ms.fit(XV[tr],ys[tr])
        pl=ml.predict_proba(XV[te])[:,1];ps=ms.predict_proba(XV[te])[:,1];idxs=V[te]
        for th in THRESHOLDS:
            rows=[]
            for j,i0 in enumerate(idxs):
                if max(pl[j],ps[j])<th:continue
                dire='LONG' if pl[j]>=ps[j] else 'SHORT';s=ss[i0];o=outcome_path(s['g'],s['pos'],dire,tp,sl)
                rows.append({'date':s['date'],'year':s['year'],'pnl':o[1],'outcome':o[2],'direction':dire,'prob':float(max(pl[j],ps[j]))})
            if not rows:continue
            m=summarize(rows);yd=m['years'];freq=all(yd.get(y,{}).get('n',0)>=100 for y in [2023,2024,2025]) and yd.get(2026,{}).get('n',0)>50
            ally=all(yd.get(y,{}).get('net',-1)<=0 is False and yd.get(y,{}).get('net',-1)>0 for y in TEST_YEARS)
            results.append({'tm':tm,'tp':tp,'sl':sl,'rr':tp/sl,'threshold':th,'freq_ok':freq,'all_years_positive':ally,**{k:v for k,v in m.items() if k!='years'},**{f'n_{y}':yd.get(y,{}).get('n',0) for y in TEST_YEARS},**{f'net_{y}':yd.get(y,{}).get('net',0) for y in TEST_YEARS}})
    R=pd.DataFrame(results);R.to_csv(OUT/'grid.csv',index=False)
    strict=R[(R.freq_ok)&(R.all_years_positive)&(R.wr>=.65)&(R.streak<=3)] if len(R) else R
    near=R[(R.freq_ok)&(R.all_years_positive)&(R.wr>=.65)&(R.streak.between(4,5))] if len(R) else R
    frontier=R[(R.freq_ok)&(R.all_years_positive)].sort_values(['wr','ev'],ascending=False).head(30) if len(R) else R
    strict.to_csv(OUT/'strict.csv',index=False);near.to_csv(OUT/'near.csv',index=False);frontier.to_csv(OUT/'frontier.csv',index=False)
    summary={'rows':len(R),'strict_count':len(strict),'near_count':len(near),'frontier_top':frontier.head(10).to_dict('records'),'strict_top':strict.head(10).to_dict('records'),'near_top':near.head(10).to_dict('records')}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print('ML_OOS_DONE');print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)},indent=2));print(frontier.head(10).to_string(index=False))
if __name__=='__main__':main()

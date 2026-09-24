import json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_backtest as bt
OUT=Path('hold24_results');OUT.mkdir(exist_ok=True)
SCHEMES=[('10_10','fixed',10.,10.,1.),('15_15','fixed',15.,15.,1.),('20_20','fixed',20.,20.,1.),('20_15','fixed',20.,15.,20/15),('25_20','fixed',25.,20.,1.25),('30_25','fixed',30.,25.,1.2),('40_30','fixed',40.,30.,4/3),('A25_R1','asia',.25,1.),('A25_R125','asia',.25,1.25),('R60_R1','r60',1.,1.),('R60_R125','r60',1.,1.25)]

def params(c,s):
    if s[1]=='fixed':return s[2],s[3]
    scale=c[s[1]]
    if not np.isfinite(scale) or scale<=0:return None
    sl=float(np.clip(scale*s[2],5,40));return sl*s[3],sl

def make(df,tsns,pos,d,fam,var,AR,PR):
    if pos+1>=len(df):return None
    ep=pos+1;e=df.iloc[ep];end=int(np.searchsorted(tsns,tsns[ep]+24*3600*10**9,side='right'))
    fut=df.iloc[ep:end]
    if fut.empty:return None
    return {'date':str(e.local_date),'year':int(e.year),'family':fam,'variant':var,'direction':d,'entry':float(e.open_ask if d=='LONG' else e.open_bid),'asia':AR,'r60':float(e.r60) if pd.notna(e.r60) else np.nan,'prev':PR,
            'hi':fut.high_bid.to_numpy(float) if d=='LONG' else fut.high_ask.to_numpy(float),'lo':fut.low_bid.to_numpy(float) if d=='LONG' else fut.low_ask.to_numpy(float),'cl':fut.close_bid.to_numpy(float) if d=='LONG' else fut.close_ask.to_numpy(float)}

def evalc(c,s):
    ps=params(c,s)
    if ps is None:return None
    tp,sl=ps;e=c['entry'];hi=c['hi'];lo=c['lo'];cl=c['cl']
    if c['direction']=='LONG':a=np.flatnonzero(hi>=e+tp);b=np.flatnonzero(lo<=e-sl);last=float(cl[-1]-e)
    else:a=np.flatnonzero(lo<=e-tp);b=np.flatnonzero(hi>=e+sl);last=float(e-cl[-1])
    a=int(a[0]) if len(a) else None;b=int(b[0]) if len(b) else None
    if a is None and b is None:return last,'TIME',tp,sl
    if a is not None and (b is None or a<b):return tp,'TP',tp,sl
    return -sl,'SL',tp,sl

def metrics(cs,s):
    rows=[]
    for c in cs:
        x=evalc(c,s)
        if x:rows.append((c['date'],c['year'],*x))
    if not rows:return None
    r=pd.DataFrame(rows,columns=['date','year','pnl','outcome','tpv','slv']).sort_values('date').reset_index(drop=True);p=r.pnl.to_numpy(float)
    best=cur=0;bs=be=st=None
    for i,v in enumerate(p):
        if v<0:
            if cur==0:st=i
            cur+=1
            if cur>best:best=cur;bs=st;be=i
        else:cur=0
    gains=p[p>0].sum();loss=-p[p<0].sum();eq=np.cumsum(p);peak=np.maximum.accumulate(np.r_[0.,eq]);dd=peak[1:]-eq
    yd={}
    for y,g in r.groupby('year'):
        py=g.pnl.to_numpy(float);yd[int(y)]={'n':len(g),'net':float(py.sum()),'pos':int((py>0).sum()),'neg':int((py<0).sum())}
    return {'n':len(r),'positive':int((p>0).sum()),'negative':int((p<0).sum()),'win_rate':float((p>0).mean()),'tp_count':int((r.outcome=='TP').sum()),'sl_count':int((r.outcome=='SL').sum()),'time_count':int((r.outcome=='TIME').sum()),'net':float(p.sum()),'ev':float(p.mean()),'pf':float(gains/loss) if loss>0 else 99.,'max_dd':float(dd.max()),'max_loss_streak':best,'streak_start':str(r.iloc[bs].date) if bs is not None else None,'streak_end':str(r.iloc[be].date) if be is not None else None,'avg_tp':float(r.tpv.mean()),'avg_sl':float(r.slv.mean()),'years':yd}

def first_close(g,H,L,start=600,end=960):
    p=g[(g['mod']>=start)&(g['mod']<end)]
    for idx,r in p.iterrows():
        if r.close_mid>=H:return int(idx),'LONG'
        if r.close_mid<=L:return int(idx),'SHORT'
    return None,None

def main():
    df=bt.load_data().sort_values('timestamp').reset_index(drop=True);df['mod']=(df.hour*60+df.minute).astype(int)
    df['ema60']=df.close_mid.ewm(span=60,adjust=False).mean();df['ema240']=df.close_mid.ewm(span=240,adjust=False).mean();df['r60']=df.high_mid.rolling(60,min_periods=45).max()-df.low_mid.rolling(60,min_periods=45).min()
    tsns=df.timestamp.astype('int64').to_numpy();C={}
    def add(f,v,c):
        if c:C.setdefault((f,v),[]).append(c)
    prev=None;wp=list(itertools.combinations(range(5),2))
    for day,g in df.groupby('local_date',sort=True):
        if pd.Timestamp(day).dayofweek>=5:continue
        asia=g[(g['mod']>=60)&(g['mod']<540)]
        if len(asia)<420:
            prev=(float(g.iloc[0].open_mid),float(g.iloc[-1].close_mid),float(g.high_mid.max()),float(g.low_mid.min()));continue
        H=float(asia.high_mid.max());L=float(asia.low_mid.min());AR=H-L;PR=(prev[2]-prev[3]) if prev else np.nan;wd=int(pd.Timestamp(day).dayofweek)
        # close-confirmed Asia breakout 10-16
        idx,d=first_close(g,H,L,600,960)
        if d:
            c=make(df,tsns,idx,d,'asia_close_10_16','all',AR,PR);add('asia_close_10_16','all',c)
            for p in wp:
                if wd in p:add('asia_close_wdpair',f'{p[0]}-{p[1]}',c)
        # fixed time 10 and 14 signals
        for tm in [600,840]:
            ss=g[g['mod']==tm]
            if ss.empty:continue
            idx=int(ss.index[0]);r=ss.iloc[0];px=float(r.close_mid);e60=float(r.ema60);e240=float(r.ema240)
            # Asia midpoint continuation/fade
            d='LONG' if px>(H+L)/2 else 'SHORT';add(f'asia_mid_cont_{tm}','all',make(df,tsns,idx,d,f'asia_mid_cont_{tm}','all',AR,PR));add(f'asia_mid_fade_{tm}','all',make(df,tsns,idx,'SHORT' if d=='LONG' else 'LONG',f'asia_mid_fade_{tm}','all',AR,PR))
            # EMA trend
            if px>e60>e240:d2='LONG'
            elif px<e60<e240:d2='SHORT'
            else:d2=None
            if d2:add(f'ema_trend_{tm}','all',make(df,tsns,idx,d2,f'ema_trend_{tm}','all',AR,PR))
            # prev close continuation/fade
            if prev:
                gap=px-prev[1];dc='LONG' if gap>0 else 'SHORT';add(f'prevclose_cont_{tm}','all',make(df,tsns,idx,dc,f'prevclose_cont_{tm}','all',AR,PR));add(f'prevclose_fade_{tm}','all',make(df,tsns,idx,'SHORT' if dc=='LONG' else 'LONG',f'prevclose_fade_{tm}','all',AR,PR))
            # ensemble vote
            w=g[(g['mod']>=tm-60)&(g['mod']<tm)];mv=float(w.iloc[-1].close_mid-w.iloc[0].open_mid) if len(w)>=40 else 0
            votes=[1 if px>(H+L)/2 else -1,1 if px>e60 else -1,1 if mv>0 else -1]
            if prev:votes.append(1 if px>prev[1] else -1)
            score=sum(votes);de='LONG' if score>0 else 'SHORT';add(f'ensemble_{tm}','majority',make(df,tsns,idx,de,f'ensemble_{tm}','majority',AR,PR))
        # 09-10 momentum at 10
        w=g[(g['mod']>=540)&(g['mod']<600)];s=g[g['mod']==600]
        if len(w)>=40 and not s.empty:
            idx=int(s.index[0]);mv=float(w.iloc[-1].close_mid-w.iloc[0].open_mid);rng=float(w.high_mid.max()-w.low_mid.min())
            for frac in [.25,.4,.55]:
                if rng>0 and abs(mv)/rng>=frac:
                    dc='LONG' if mv>0 else 'SHORT';add('09_10_norm_cont',f'f={frac}',make(df,tsns,idx,dc,'09_10_norm_cont',f'f={frac}',AR,PR));add('09_10_norm_fade',f'f={frac}',make(df,tsns,idx,'SHORT' if dc=='LONG' else 'LONG','09_10_norm_fade',f'f={frac}',AR,PR))
        prev=(float(g.iloc[0].open_mid),float(g.iloc[-1].close_mid),float(g.high_mid.max()),float(g.low_mid.min()))
    freq={};counts=[]
    for k,cs in C.items():
        yc=pd.Series([c['year'] for c in cs]).value_counts().to_dict();counts.append({'family':k[0],'variant':k[1],**{f'n_{y}':int(yc.get(y,0)) for y in [2023,2024,2025,2026]}})
        if yc.get(2023,0)>=100 and yc.get(2024,0)>=100 and yc.get(2025,0)>=100 and yc.get(2026,0)>50:freq[k]=cs
    pd.DataFrame(counts).to_csv(OUT/'signal_counts.csv',index=False)
    rows=[]
    for (f,v),cs in freq.items():
        for s in SCHEMES:
            m=metrics(cs,s);yd=m['years'] if m else {}
            if not all(y in yd for y in [2023,2024,2025,2026]):continue
            row={'family':f,'variant':v,'exit':s[0],'rr':s[4] if s[1]=='fixed' else s[3],**{f'n_{y}':yd[y]['n'] for y in yd},**{f'net_{y}':yd[y]['net'] for y in yd}}
            for k2,val in m.items():
                if k2!='years':row[k2]=val
            rows.append(row)
    G=pd.DataFrame(rows);G.to_csv(OUT/'grid.csv',index=False)
    base=G[(G.win_rate>=.65)&(G.net_2023>0)&(G.net_2024>0)&(G.net_2025>0)&(G.net_2026>0)] if len(G) else G
    strict=base[base.max_loss_streak<=3].copy() if len(base) else base;near=base[(base.max_loss_streak>=4)&(base.max_loss_streak<=5)].copy() if len(base) else base
    if len(strict):strict=strict.sort_values(['rr','ev','pf'],ascending=False)
    if len(near):near=near.sort_values(['rr','ev','pf'],ascending=False)
    strict.to_csv(OUT/'strict.csv',index=False);near.to_csv(OUT/'near_4_5.csv',index=False)
    summary={'families':len(C),'frequent_families':len(freq),'grid_rows':len(G),'strict_count':len(strict),'near_count':len(near),'strict_top':strict.head(10).to_dict('records'),'near_top':near.head(10).to_dict('records')}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print('HOLD24_DONE');print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)},indent=2))
    if len(strict):print(strict.head(10).to_string(index=False))
    if len(near):print(near.head(10).to_string(index=False))
if __name__=='__main__':main()

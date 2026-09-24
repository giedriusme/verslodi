import io, json, math, itertools
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

OUT=Path('portfolio_discovery_results'); OUT.mkdir(exist_ok=True)
RAW='https://raw.githubusercontent.com/kevingtlin/Market-Data-Lab/main'
TZ='Europe/Vilnius'
MONTHS=[(y,m) for y in range(2018,2027) for m in range(1,13) if y<2026 or m<=8]

EXITS=[
 ('TP5_SL15',5.,15.),('TP10_SL25',10.,25.),('TP15_SL25',15.,25.),('TP10_SL20',10.,20.),('TP15_SL20',15.,20.),
 ('TP15_SL15',15.,15.),('TP20_SL20',20.,20.),('TP20_SL15',20.,15.),('TP25_SL20',25.,20.),
 ('TP30_SL25',30.,25.),('TP40_SL30',40.,30.)]
HORIZONS=[480,720,1440]  # 8h / 12h / 24h

S=requests.Session(); S.headers.update({'User-Agent':'xauusd-portfolio-discovery/1.0'})

def month(side,y,m):
    u=f'{RAW}/xauusd/{side}/m1/xauusd_{side}_m1_{y:04d}_{m:02d}.csv'
    r=S.get(u,timeout=90); r.raise_for_status(); d=pd.read_csv(io.StringIO(r.text))
    return d.rename(columns={c:f'{c}_{side}' for c in ['open','high','low','close']})

def load():
    fs=[]
    for i,(y,m) in enumerate(MONTHS,1):
        print(f'Download {y}-{m:02d} {i}/{len(MONTHS)}',flush=True)
        b=month('bid',y,m); a=month('ask',y,m)
        fs.append(b.merge(a,on='timestamp',how='inner'))
    d=pd.concat(fs,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    for c in ['open','high','low','close']:
        d[f'{c}_mid']=(d[f'{c}_bid']+d[f'{c}_ask'])/2
    dt=pd.to_datetime(d.timestamp,unit='ms',utc=True)
    d['dt_local']=dt.dt.tz_convert(TZ); d['date']=d.dt_local.dt.date; d['year']=d.dt_local.dt.year.astype(int)
    d['mod']=(d.dt_local.dt.hour*60+d.dt_local.dt.minute).astype(int); d['wd']=d.dt_local.dt.dayofweek.astype(int)
    d['ema20']=d.close_mid.ewm(span=20,adjust=False).mean(); d['ema60']=d.close_mid.ewm(span=60,adjust=False).mean(); d['ema240']=d.close_mid.ewm(span=240,adjust=False).mean()
    d['r60']=d.high_mid.rolling(60,min_periods=45).max()-d.low_mid.rolling(60,min_periods=45).min()
    return d

def mk(g,idx,side,fam,var,extra=None):
    loc=g.index.get_loc(idx)
    if loc+1>=len(g): return None
    e=g.iloc[loc+1]
    x={'date':str(g.iloc[0].date),'year':int(g.iloc[0].year),'wd':int(g.iloc[0].wd),'family':fam,'variant':var,'side':side,
       'entry_i':int(g.iloc[loc+1].global_i),'entry':float(e.open_ask if side=='LONG' else e.open_bid)}
    if extra: x.update(extra)
    return x

def first_break(post,H,L):
    for idx,r in post.iterrows():
        up=float(r.close_mid)>H; dn=float(r.close_mid)<L
        if up and not dn:return idx,'LONG'
        if dn and not up:return idx,'SHORT'
    return None,None

def first_sweep(post,H,L):
    for idx,r in post.iterrows():
        up=float(r.high_mid)>=H and float(r.close_mid)<H
        dn=float(r.low_mid)<=L and float(r.close_mid)>L
        if up and dn:continue
        if up:return idx,'SHORT'
        if dn:return idx,'LONG'
    return None,None

def break_retest(g,H,L,start,end,wait=90):
    post=g[(g.mod>=start)&(g.mod<end)]
    for idx,r in post.iterrows():
        if r.close_mid>H:
            loc=g.index.get_loc(idx); later=g.iloc[loc+1:min(len(g),loc+1+wait)]
            for j,x in later.iterrows():
                if x.low_mid<=H and x.close_mid>=H:return j,'LONG'
            return None,None
        if r.close_mid<L:
            loc=g.index.get_loc(idx); later=g.iloc[loc+1:min(len(g),loc+1+wait)]
            for j,x in later.iterrows():
                if x.high_mid>=L and x.close_mid<=L:return j,'SHORT'
            return None,None
    return None,None

def candidates(df):
    df=df.copy(); df['global_i']=np.arange(len(df))
    days=[]
    for day,g0 in df.groupby('date',sort=True):
        if pd.Timestamp(day).dayofweek<5: days.append((day,g0.sort_values('timestamp').reset_index(drop=True)))
    C={}; prev=None; ar_hist=[]
    def add(c):
        if c is not None:C.setdefault((c['family'],c['variant']),[]).append(c)
    for day,g in days:
        asia=g[(g.mod>=60)&(g.mod<540)]
        if len(asia)<420:
            if len(g): prev={'C':float(g.iloc[-1].close_mid),'H':float(g.high_mid.max()),'L':float(g.low_mid.min()),'R':float(g.high_mid.max()-g.low_mid.min())}
            continue
        AH=float(asia.high_mid.max()); AL=float(asia.low_mid.min()); AR=AH-AL; AO=float(asia.iloc[0].open_mid); AC=float(asia.iloc[-1].close_mid); AM=(AH+AL)/2
        med20=float(np.median(ar_hist[-20:])) if len(ar_hist)>=10 else np.nan
        extra={'asia_range':AR,'asia_ratio':AR/med20 if np.isfinite(med20) and med20>0 else np.nan}
        wd=int(g.iloc[0].wd)
        # fixed-time context helpers
        for tm in [600,840]:
            ss=g[g.mod==tm]
            if ss.empty: continue
            idx=ss.index[0]; r=ss.iloc[0]; px=float(r.close_mid)
            # Asia body continuation / fade
            if AC!=AO:
                side='LONG' if AC>AO else 'SHORT'; add(mk(g,idx,side,f'asia_body_cont_{tm}','all',extra)); add(mk(g,idx,'SHORT' if side=='LONG' else 'LONG',f'asia_body_fade_{tm}','all',extra))
            # Asia midpoint
            side='LONG' if px>AM else 'SHORT'; add(mk(g,idx,side,f'asia_mid_cont_{tm}','all',extra))
            # previous close continuation
            if prev:
                side='LONG' if px>prev['C'] else 'SHORT'; add(mk(g,idx,side,f'prevclose_cont_{tm}','all',extra))
            # trend pullback
            e20,e60,e240=float(r.ema20),float(r.ema60),float(r.ema240); rv=float(r.r60) if pd.notna(r.r60) else np.nan
            if np.isfinite(rv) and rv>0 and abs(px-e20)<=0.25*rv:
                if e20>e60>e240:add(mk(g,idx,'LONG',f'trend_pullback_{tm}','f025',extra))
                elif e20<e60<e240:add(mk(g,idx,'SHORT',f'trend_pullback_{tm}','f025',extra))
            # 60m impulse continuation/fade
            w=g[(g.mod>=tm-60)&(g.mod<tm)]
            if len(w)>=45:
                mv=float(w.iloc[-1].close_mid-w.iloc[0].open_mid); rg=float(w.high_mid.max()-w.low_mid.min())
                if rg>0 and abs(mv)/rg>=0.40:
                    side='LONG' if mv>0 else 'SHORT'; add(mk(g,idx,side,f'impulse60_cont_{tm}','s040',extra)); add(mk(g,idx,'SHORT' if side=='LONG' else 'LONG',f'impulse60_fade_{tm}','s040',extra))
        # Asia close breakout windows
        for st,en,label in [(540,840,'asia_break_09_14'),(600,960,'asia_break_10_16')]:
            idx,side=first_break(g[(g.mod>=st)&(g.mod<en)],AH,AL)
            if side:
                add(mk(g,idx,side,label,'all',extra))
                if wd in (3,4): add(mk(g,idx,side,label,'thu_fri',extra))
                if wd in (0,): add(mk(g,idx,side,label,'mon',extra))
        # Sweep / failed breakout
        idx,side=first_sweep(g[(g.mod>=540)&(g.mod<960)],AH,AL)
        if side:add(mk(g,idx,side,'asia_liquidity_sweep','all',extra))
        # Break -> retest
        idx,side=break_retest(g,AH,AL,600,900,90)
        if side:add(mk(g,idx,side,'asia_break_retest','w90',extra))
        # previous-day sweep
        if prev:
            idx,side=first_sweep(g[(g.mod>=540)&(g.mod<960)],prev['H'],prev['L'])
            if side:add(mk(g,idx,side,'prevday_liquidity_sweep','all',extra))
            # inside previous-day range -> Asia breakout
            if AH<prev['H'] and AL>prev['L']:
                idx,side=first_break(g[(g.mod>=540)&(g.mod<960)],AH,AL)
                if side:add(mk(g,idx,side,'inside_prevday_asia_break','all',extra))
        # compression / expansion regime
        if np.isfinite(med20) and med20>0:
            ratio=AR/med20
            if ratio<=0.70:
                idx,side=first_break(g[(g.mod>=540)&(g.mod<960)],AH,AL)
                if side:add(mk(g,idx,side,'asia_compression_break','r070',extra))
            if ratio>=1.40:
                ss=g[g.mod==600]
                if not ss.empty and AC!=AO:
                    side='SHORT' if AC>AO else 'LONG'; add(mk(g,ss.index[0],side,'asia_expansion_fade_10','r140',extra))
            if 0.80<=ratio<=1.20:
                ss=g[g.mod==840]
                if not ss.empty:
                    px=float(ss.iloc[0].close_mid); side='LONG' if px>AM else 'SHORT'; add(mk(g,ss.index[0],side,'normal_range_mid_cont_14','r080_120',extra))
        ar_hist.append(AR)
        prev={'C':float(g.iloc[-1].close_mid),'H':float(g.high_mid.max()),'L':float(g.low_mid.min()),'R':float(g.high_mid.max()-g.low_mid.min())}
    return C

def trade_result(df,c,tp,sl,horizon):
    i=c['entry_i']; e=c['entry']; side=c['side']; f=df.iloc[i:min(len(df),i+horizon)]
    if len(f)==0:return None
    if side=='LONG':
        a=np.flatnonzero(f.high_bid.to_numpy(float)>=e+tp); b=np.flatnonzero(f.low_bid.to_numpy(float)<=e-sl); mark=float(f.iloc[-1].close_bid-e)
    else:
        a=np.flatnonzero(f.low_ask.to_numpy(float)<=e-tp); b=np.flatnonzero(f.high_ask.to_numpy(float)>=e+sl); mark=float(e-f.iloc[-1].close_ask)
    ia=int(a[0]) if len(a) else None; ib=int(b[0]) if len(b) else None
    if ia is not None and (ib is None or ia<ib): pnl=tp; out='TP'; hold=ia+1
    elif ib is not None: pnl=-sl; out='SL'; hold=ib+1
    else: pnl=mark; out='TIME'; hold=len(f)
    return {'date':c['date'],'year':c['year'],'wd':c['wd'],'pnl':float(pnl),'R':float(pnl/sl),'outcome':out,'hold':hold,'side':side}

def met(rows,years=None):
    r=pd.DataFrame(rows)
    if r.empty:return None
    if years is not None:r=r[r.year.isin(years)]
    if r.empty:return None
    p=r.R.to_numpy(float); eq=np.cumsum(p); peak=np.maximum.accumulate(np.r_[0.,eq]); dd=peak[1:]-eq
    losses=-p[p<0].sum(); gains=p[p>0].sum(); streak=cur=0
    for v in p:
        cur=cur+1 if v<0 else 0; streak=max(streak,cur)
    yd={}
    for y,g in r.groupby('year'):
        yd[int(y)]={'n':int(len(g)),'R':float(g.R.sum()),'wr':float((g.R>0).mean()),'pf':float(g.loc[g.R>0,'R'].sum()/(-g.loc[g.R<0,'R'].sum())) if (g.R<0).any() else 99.}
    return {'n':int(len(r)),'R':float(p.sum()),'evR':float(p.mean()),'wr':float((p>0).mean()),'pf':float(gains/losses) if losses>0 else 99.,'ddR':float(dd.max()),'streak':int(streak),'years':yd}

def main():
    df=load(); C=candidates(df)
    all_rows=[]; trade_store={}
    train_years=list(range(2018,2023)); test_years=[2023,2024,2025,2026]
    for (fam,var),cs in C.items():
        for ex,tp,sl in EXITS:
            for h in HORIZONS:
                rows=[]
                for c in cs:
                    z=trade_result(df,c,tp,sl,h)
                    if z:rows.append(z)
                tr=met(rows,train_years); te=met(rows,test_years); whole=met(rows)
                if tr is None or te is None:continue
                pos_train=sum(1 for y in train_years if tr['years'].get(y,{}).get('R',-1)>0)
                pos_test=sum(1 for y in test_years if te['years'].get(y,{}).get('R',-1)>0)
                key=f'{fam}|{var}|{ex}|H{h}'
                score=(tr['evR']*math.sqrt(max(tr['n'],1))) + 0.20*(tr['pf']-1) + 0.08*pos_train - 0.015*tr['ddR']
                rec={'key':key,'family':fam,'variant':var,'exit':ex,'tp':tp,'sl':sl,'rr':tp/sl,'horizon_min':h,'train_score':score,
                     'train_n':tr['n'],'train_R':tr['R'],'train_evR':tr['evR'],'train_wr':tr['wr'],'train_pf':tr['pf'],'train_ddR':tr['ddR'],'train_streak':tr['streak'],'train_pos_years':pos_train,
                     'oos_n':te['n'],'oos_R':te['R'],'oos_evR':te['evR'],'oos_wr':te['wr'],'oos_pf':te['pf'],'oos_ddR':te['ddR'],'oos_streak':te['streak'],'oos_pos_years':pos_test,
                     'all_n':whole['n'],'all_R':whole['R'],'all_evR':whole['evR'],'all_pf':whole['pf']}
                for y in range(2018,2027):
                    yy=whole['years'].get(y,{}); rec[f'n_{y}']=yy.get('n',0);rec[f'R_{y}']=yy.get('R',0.);rec[f'wr_{y}']=yy.get('wr',np.nan)
                all_rows.append(rec); trade_store[key]=rows
    grid=pd.DataFrame(all_rows).sort_values('train_score',ascending=False)
    grid.to_csv(OUT/'grid.csv',index=False)
    # candidate shortlist chosen primarily from TRAIN, OOS only used as validation gate
    q=grid[(grid.train_n>=120)&(grid.train_pf>=1.03)&(grid.train_evR>0.01)&(grid.train_pos_years>=3)].copy()
    # one best exit per family/variant by train score
    q=q.sort_values('train_score',ascending=False).drop_duplicates(['family','variant'])
    q['robust_oos']=(q.oos_pf>=1.02)&(q.oos_evR>0)&(q.oos_pos_years>=3)
    robust=q[q.robust_oos].copy().sort_values(['oos_pos_years','oos_evR','train_score'],ascending=False)
    robust.to_csv(OUT/'robust_candidates.csv',index=False)
    # Greedy diversified basket: distinct families, low TRAIN daily correlation, robust OOS as validation
    chosen=[]; series={}
    for _,r in robust.iterrows():
        key=r.key
        rr=pd.DataFrame(trade_store[key]); rr=rr[rr.year.isin(train_years)]
        s=rr.groupby('date').R.sum()
        ok=True
        for k in chosen:
            a,b=s.align(series[k],join='outer',fill_value=0.)
            corr=float(a.corr(b)) if a.std()>0 and b.std()>0 else 0.
            if abs(corr)>0.45:ok=False;break
        if ok:
            chosen.append(key);series[key]=s
        if len(chosen)>=5:break
    # If robust set too small, fill from best train candidates that are at least OOS positive overall
    if len(chosen)<5:
        fill=q[(q.oos_R>0)&(q.oos_pf>1.0)].sort_values(['oos_pos_years','train_score'],ascending=False)
        for _,r in fill.iterrows():
            key=r.key
            if key in chosen or any(key.startswith(x.split('|')[0]+'|') for x in chosen):continue
            rr=pd.DataFrame(trade_store[key]); rr=rr[rr.year.isin(train_years)]; s=rr.groupby('date').R.sum(); ok=True
            for k in chosen:
                a,b=s.align(series[k],join='outer',fill_value=0.); corr=float(a.corr(b)) if a.std()>0 and b.std()>0 else 0.
                if abs(corr)>0.45:ok=False;break
            if ok:chosen.append(key);series[key]=s
            if len(chosen)>=5:break
    # correlation all-period and portfolio equal risk: each strategy risks 1/len(chosen) unit per signal
    daily={}
    for k in chosen:
        rr=pd.DataFrame(trade_store[k]); daily[k]=rr.groupby('date').R.sum()
    corr=pd.DataFrame(daily).fillna(0).corr() if daily else pd.DataFrame()
    corr.to_csv(OUT/'chosen_correlation.csv')
    d=pd.DataFrame(daily).fillna(0.) if daily else pd.DataFrame()
    if len(d.columns):
        port=d.mean(axis=1); dates=pd.to_datetime(port.index); pr=pd.DataFrame({'date':port.index,'year':dates.year,'R':port.values})
        pm=met(pr.to_dict('records')); pr.to_csv(OUT/'portfolio_daily.csv',index=False)
    else: pm=None
    chosen_rows=grid[grid.key.isin(chosen)].copy(); chosen_rows.to_csv(OUT/'chosen_strategies.csv',index=False)
    summary={'data_start':str(df.dt_local.iloc[0]),'data_end':str(df.dt_local.iloc[-1]),'candidate_families':len(C),'grid_rows':len(grid),'train_shortlist':len(q),'robust_oos_count':len(robust),
             'chosen':chosen_rows.to_dict('records'),'portfolio':pm,'correlation':corr.round(3).to_dict() if len(corr) else {}}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps({'candidate_families':len(C),'grid_rows':len(grid),'train_shortlist':len(q),'robust_oos_count':len(robust),'chosen':chosen,'portfolio':pm},indent=2,allow_nan=False))

if __name__=='__main__':main()

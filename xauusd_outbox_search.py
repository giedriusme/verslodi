import json
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import xauusd_backtest as bt

OUT=Path('outbox_results');OUT.mkdir(exist_ok=True)
YEARS=[2023,2024,2025,2026]

# exit schemes: fixed + volatility-normalized, all initial RR >= 1
FIXED=[('F10_10','fixed',10.,10.,1.0),('F15_15','fixed',15.,15.,1.0),('F20_20','fixed',20.,20.,1.0),
       ('F20_15','fixed',20.,15.,20/15),('F25_20','fixed',25.,20.,1.25),('F30_25','fixed',30.,25.,1.2),('F40_30','fixed',40.,30.,4/3)]
DYN=[('R60x075_R1','r60',.75,1.0),('R60x075_R125','r60',.75,1.25),('R60x1_R1','r60',1.,1.0),('R60x1_R125','r60',1.,1.25),
     ('ASIAx025_R1','asia',.25,1.0),('ASIAx025_R125','asia',.25,1.25),('ASIAx033_R1','asia',.33,1.0),('ASIAx033_R125','asia',.33,1.25),
     ('PREVx025_R1','prev',.25,1.0),('PREVx025_R125','prev',.25,1.25)]


def make_cand(g,idx,direction,family,variant,asia_range=np.nan,prev_range=np.nan):
    loc=g.index.get_loc(idx)
    if loc+1>=len(g):return None
    e=g.iloc[loc+1];fut=g.iloc[loc+1:]
    scale=float(e.r60) if pd.notna(e.r60) else np.nan
    return {'date':str(g.iloc[0].local_date),'year':int(g.iloc[0].year),'weekday':int(pd.Timestamp(g.iloc[0].local_date).dayofweek),
            'family':family,'variant':variant,'direction':direction,'entry':float(e.open_ask if direction=='LONG' else e.open_bid),
            'r60':scale,'asia':float(asia_range) if pd.notna(asia_range) else np.nan,'prev':float(prev_range) if pd.notna(prev_range) else np.nan,
            'hi':fut.high_bid.to_numpy(float) if direction=='LONG' else fut.high_ask.to_numpy(float),
            'lo':fut.low_bid.to_numpy(float) if direction=='LONG' else fut.low_ask.to_numpy(float),
            'cl':fut.close_bid.to_numpy(float) if direction=='LONG' else fut.close_ask.to_numpy(float)}

def exit_params(c,scheme):
    if scheme[1]=='fixed':return scheme[2],scheme[3]
    _,kind,mult,rr=scheme
    scale=c.get(kind,np.nan)
    if not np.isfinite(scale) or scale<=0:return None
    sl=float(np.clip(scale*mult,5.,40.));tp=float(sl*rr)
    return tp,sl

def evaluate(c,scheme):
    ps=exit_params(c,scheme)
    if ps is None:return None
    tp,sl=ps;e=c['entry'];hi=c['hi'];lo=c['lo'];cl=c['cl']
    if len(cl)==0:return None
    if c['direction']=='LONG':
        a=np.flatnonzero(hi>=e+tp);b=np.flatnonzero(lo<=e-sl);eod=float(cl[-1]-e)
    else:
        a=np.flatnonzero(lo<=e-tp);b=np.flatnonzero(hi>=e+sl);eod=float(e-cl[-1])
    a=int(a[0]) if len(a) else None;b=int(b[0]) if len(b) else None
    if a is None and b is None:return eod,'EOD',tp,sl
    if a is not None and (b is None or a<b):return tp,'TP',tp,sl
    return -sl,'SL',tp,sl

def metrics(cs,scheme):
    rows=[]
    for c in cs:
        x=evaluate(c,scheme)
        if x is not None:rows.append((c['date'],c['year'],x[0],x[1],x[2],x[3]))
    if not rows:return None
    r=pd.DataFrame(rows,columns=['date','year','pnl','outcome','tpv','slv']).sort_values('date').reset_index(drop=True)
    p=r.pnl.to_numpy(float);pos=int((p>0).sum());neg=int((p<0).sum());best=cur=0;bs=be=start=None
    for i,v in enumerate(p):
        if v<0:
            if cur==0:start=i
            cur+=1
            if cur>best:best=cur;bs=start;be=i
        else:cur=0
    gains=p[p>0].sum();losses=-p[p<0].sum();eq=np.cumsum(p);peak=np.maximum.accumulate(np.r_[0.,eq]);dd=peak[1:]-eq
    yd={}
    for y,gg in r.groupby('year'):
        py=gg.pnl.to_numpy(float);yd[int(y)]={'n':len(gg),'net':float(py.sum()),'pos':int((py>0).sum()),'neg':int((py<0).sum()),'tp':int((gg.outcome=='TP').sum()),'sl':int((gg.outcome=='SL').sum()),'eod':int((gg.outcome=='EOD').sum())}
    return {'n':len(r),'positive':pos,'negative':neg,'win_rate':pos/len(r),'tp_count':int((r.outcome=='TP').sum()),'sl_count':int((r.outcome=='SL').sum()),'eod_count':int((r.outcome=='EOD').sum()),
            'net':float(p.sum()),'ev':float(p.mean()),'pf':float(gains/losses) if losses>0 else 99.,'max_dd':float(dd.max()),'max_loss_streak':best,
            'streak_start':str(r.iloc[bs].date) if bs is not None else None,'streak_end':str(r.iloc[be].date) if be is not None else None,
            'avg_tp':float(r.tpv.mean()),'avg_sl':float(r.slv.mean()),'avg_rr':float((r.tpv/r.slv).mean()),'years':yd}

def first_close(post,H,L,buf=0.):
    for idx,r in post.iterrows():
        if float(r.close_mid)>=H+buf:return int(idx),'LONG'
        if float(r.close_mid)<=L-buf:return int(idx),'SHORT'
    return None,None

def first_sweep(post,H,L,buf=0.):
    for idx,r in post.iterrows():
        up=float(r.high_mid)>=H+buf and float(r.close_mid)<H
        dn=float(r.low_mid)<=L-buf and float(r.close_mid)>L
        if up and dn:continue
        if up:return int(idx),'SHORT'
        if dn:return int(idx),'LONG'
    return None,None

def breakout_retest(g,H,L,start,end,maxwait=90,buf=0.):
    post=g[(g['mod']>=start)&(g['mod']<end)]
    for idx,r in post.iterrows():
        if float(r.close_mid)>=H+buf:
            loc=g.index.get_loc(idx)
            later=g.iloc[loc+1:min(len(g),loc+1+maxwait)]
            for idx2,x in later.iterrows():
                if float(x.low_mid)<=H and float(x.close_mid)>=H:return int(idx2),'LONG'
            return None,None
        if float(r.close_mid)<=L-buf:
            loc=g.index.get_loc(idx);later=g.iloc[loc+1:min(len(g),loc+1+maxwait)]
            for idx2,x in later.iterrows():
                if float(x.high_mid)>=L and float(x.close_mid)<=L:return int(idx2),'SHORT'
            return None,None
    return None,None

def main():
    df=bt.load_data();df['mod']=(df.hour*60+df.minute).astype(int)
    # causal technical context
    df['ema20']=df.close_mid.ewm(span=20,adjust=False).mean();df['ema60']=df.close_mid.ewm(span=60,adjust=False).mean();df['ema240']=df.close_mid.ewm(span=240,adjust=False).mean()
    d=df.close_mid.diff();u=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean();df['rsi']=100-100/(1+u/dn.replace(0,np.nan))
    ma=df.close_mid.rolling(60,min_periods=45).mean();sd=df.close_mid.rolling(60,min_periods=45).std();df['z60']=(df.close_mid-ma)/sd.replace(0,np.nan)
    df['r60']=df.high_mid.rolling(60,min_periods=45).max()-df.low_mid.rolling(60,min_periods=45).min()
    df['ema60_lag30']=df.ema60.shift(30)
    days=[]
    for day,g in df.groupby('local_date',sort=True):
        if pd.Timestamp(day).dayofweek<5:days.append((day,g.sort_values('timestamp').reset_index(drop=True)))
    C={}
    def add(f,v,c):
        if c is not None:C.setdefault((f,v),[]).append(c)
    prev=None
    weekday_pairs=list(itertools.combinations(range(5),2))
    for day,g in days:
        asia=g[(g['mod']>=60)&(g['mod']<540)]
        if len(asia)<420:
            prev=(float(g.iloc[0].open_mid),float(g.iloc[-1].close_mid),float(g.high_mid.max()),float(g.low_mid.min()));continue
        AH=float(asia.high_mid.max());AL=float(asia.low_mid.min());AR=AH-AL;AO=float(asia.iloc[0].open_mid);AC=float(asia.iloc[-1].close_mid)
        PR=(prev[2]-prev[3]) if prev else np.nan;wd=int(pd.Timestamp(day).dayofweek)
        # 1. Asia close breakout + weekday-pair variants (high frequency pair ~=100/yr)
        for start,end,label in [(540,840,'asia_break_09_14'),(600,960,'asia_break_10_16')]:
            post=g[(g['mod']>=start)&(g['mod']<end)]
            idx,dire=first_close(post,AH,AL,0.)
            if dire:
                c=make_cand(g,idx,dire,label,'all',AR,PR);add(label,'all',c)
                for p in weekday_pairs:
                    if wd in p:add(label+'_wdpair',f'{p[0]}-{p[1]}',c)
            # close +1 confirmation
            idx,dire=first_close(post,AH,AL,1.)
            if dire:add(label+'_confirm1','all',make_cand(g,idx,dire,label+'_confirm1','all',AR,PR))
        # 2. liquidity sweep / failed break
        post=g[(g['mod']>=540)&(g['mod']<960)]
        for b in [0.,1.,2.]:
            idx,dire=first_sweep(post,AH,AL,b)
            if dire:add('asia_liquidity_sweep',f'b={b}',make_cand(g,idx,dire,'asia_liquidity_sweep',f'b={b}',AR,PR))
        # 3. breakout -> retest continuation
        for wait in [30,60,120]:
            idx,dire=breakout_retest(g,AH,AL,600,900,wait,0.)
            if dire:add('asia_break_retest',f'w={wait}',make_cand(g,idx,dire,'asia_break_retest',f'w={wait}',AR,PR))
        # 4. previous-day H/L sweep and breakout/retest
        if prev:
            PO,PC,PH,PL=prev
            post=g[(g['mod']>=540)&(g['mod']<960)]
            idx,dire=first_sweep(post,PH,PL,0.)
            if dire:add('prevday_liquidity_sweep','b=0',make_cand(g,idx,dire,'prevday_liquidity_sweep','b=0',AR,PR))
            idx,dire=breakout_retest(g,PH,PL,600,960,90,0.)
            if dire:add('prevday_break_retest','w=90',make_cand(g,idx,dire,'prevday_break_retest','w=90',AR,PR))
        # fixed-time features 10:00 and 14:00
        for tm in [600,840]:
            ss=g[g['mod']==tm]
            if ss.empty:continue
            idx=int(ss.index[0]);r=ss.iloc[0];px=float(r.close_mid);e20=float(r.ema20);e60=float(r.ema60);e240=float(r.ema240);rsi=float(r.rsi) if pd.notna(r.rsi) else 50.;z=float(r.z60) if pd.notna(r.z60) else 0.
            # last-hour momentum and last-15m candle structure
            w60=g[(g['mod']>=tm-60)&(g['mod']<tm)];w15=g[(g['mod']>=tm-15)&(g['mod']<tm)]
            mv60=float(w60.iloc[-1].close_mid-w60.iloc[0].open_mid) if len(w60)>=40 else 0.;rng60=float(w60.high_mid.max()-w60.low_mid.min()) if len(w60)>=40 else np.nan
            body15=float(w15.iloc[-1].close_mid-w15.iloc[0].open_mid) if len(w15)>=10 else 0.;range15=float(w15.high_mid.max()-w15.low_mid.min()) if len(w15)>=10 else np.nan
            # 5. 5-vote ensemble: Asia position, EMA trend, hour momentum, Asia body, previous-close relation
            votes=[]
            votes.append(1 if px>(AH+AL)/2 else -1)
            votes.append(1 if px>e60 else -1)
            votes.append(1 if mv60>0 else -1)
            votes.append(1 if AC>AO else -1)
            if prev:votes.append(1 if px>prev[1] else -1)
            score=sum(votes)
            for margin in [1,3]:
                if abs(score)>=margin:
                    dire='LONG' if score>0 else 'SHORT';add(f'ensemble_trend_{tm}',f'm={margin}',make_cand(g,idx,dire,f'ensemble_trend_{tm}',f'm={margin}',AR,PR))
            # 6. ensemble contrarian only if stretched z/RSI agrees
            if z>=1 and rsi>=55:add(f'stretch_fade_{tm}','z1_rsi55',make_cand(g,idx,'SHORT',f'stretch_fade_{tm}','z1_rsi55',AR,PR))
            elif z<=-1 and rsi<=45:add(f'stretch_fade_{tm}','z1_rsi55',make_cand(g,idx,'LONG',f'stretch_fade_{tm}','z1_rsi55',AR,PR))
            # 7. trend pullback: aligned trend but price near EMA20
            dist=abs(px-e20);vol=float(r.r60) if pd.notna(r.r60) else np.nan
            if np.isfinite(vol):
                for frac in [.15,.25,.35]:
                    if dist<=vol*frac:
                        if e20>e60>e240:add(f'trend_pullback_{tm}',f'f={frac}',make_cand(g,idx,'LONG',f'trend_pullback_{tm}',f'f={frac}',AR,PR))
                        elif e20<e60<e240:add(f'trend_pullback_{tm}',f'f={frac}',make_cand(g,idx,'SHORT',f'trend_pullback_{tm}',f'f={frac}',AR,PR))
            # 8. normalized impulse continuation/fade
            if np.isfinite(rng60) and rng60>0:
                strength=abs(mv60)/rng60
                for th in [.25,.4,.55]:
                    if strength>=th:
                        dc='LONG' if mv60>0 else 'SHORT';dfade='SHORT' if mv60>0 else 'LONG'
                        add(f'norm_impulse_cont_{tm}',f's={th}',make_cand(g,idx,dc,f'norm_impulse_cont_{tm}',f's={th}',AR,PR))
                        add(f'norm_impulse_fade_{tm}',f's={th}',make_cand(g,idx,dfade,f'norm_impulse_fade_{tm}',f's={th}',AR,PR))
            # 9. 15-min body continuation/fade if directional body dominates local range
            if np.isfinite(range15) and range15>0:
                ratio=abs(body15)/range15
                for th in [.35,.5,.65]:
                    if ratio>=th:
                        dc='LONG' if body15>0 else 'SHORT';dfade='SHORT' if body15>0 else 'LONG'
                        add(f'body15_cont_{tm}',f'r={th}',make_cand(g,idx,dc,f'body15_cont_{tm}',f'r={th}',AR,PR))
                        add(f'body15_fade_{tm}',f'r={th}',make_cand(g,idx,dfade,f'body15_fade_{tm}',f'r={th}',AR,PR))
            # 10. Asia expansion/compression regimes vs previous-day range
            if prev and PR>0:
                ratio=AR/PR
                if ratio<=.55:
                    dire='LONG' if px>(AH+AL)/2 else 'SHORT';add(f'asia_compress_pos_{tm}','r<=.55',make_cand(g,idx,dire,f'asia_compress_pos_{tm}','r<=.55',AR,PR))
                if ratio>=.8:
                    rel=(px-AL)/AR if AR>0 else .5
                    if rel>=.65:add(f'asia_expand_fade_{tm}','r>=.8',make_cand(g,idx,'SHORT',f'asia_expand_fade_{tm}','r>=.8',AR,PR))
                    elif rel<=.35:add(f'asia_expand_fade_{tm}','r>=.8',make_cand(g,idx,'LONG',f'asia_expand_fade_{tm}','r>=.8',AR,PR))
        # 11. 09-10 Asia fakeout/hold state specifically at 10
        w=g[(g['mod']>=540)&(g['mod']<600)];s10=g[g['mod']==600]
        if len(w)>=40 and not s10.empty:
            idx=int(s10.index[0]);px=float(s10.iloc[0].close_mid);up=w.high_mid.max()>AH;dn=w.low_mid.min()<AL
            if up and not dn:
                if px<AH:add('09_10_fakeout','upper',make_cand(g,idx,'SHORT','09_10_fakeout','upper',AR,PR))
                else:add('09_10_hold','upper',make_cand(g,idx,'LONG','09_10_hold','upper',AR,PR))
            elif dn and not up:
                if px>AL:add('09_10_fakeout','lower',make_cand(g,idx,'LONG','09_10_fakeout','lower',AR,PR))
                else:add('09_10_hold','lower',make_cand(g,idx,'SHORT','09_10_hold','lower',AR,PR))
            # 12. Asia body + 09-10 momentum confluence/divergence
            mv=float(w.iloc[-1].close_mid-w.iloc[0].open_mid);ab=AC-AO
            if ab*mv>0:
                dire='LONG' if mv>0 else 'SHORT';add('asia_morning_alignment','same',make_cand(g,idx,dire,'asia_morning_alignment','same',AR,PR))
            elif ab*mv<0:
                dire='LONG' if mv>0 else 'SHORT';add('asia_morning_divergence','morning_dir',make_cand(g,idx,dire,'asia_morning_divergence','morning_dir',AR,PR))
                add('asia_morning_divergence_fade','asia_dir',make_cand(g,idx,'SHORT' if dire=='LONG' else 'LONG','asia_morning_divergence_fade','asia_dir',AR,PR))
        prev=(float(g.iloc[0].open_mid),float(g.iloc[-1].close_mid),float(g.high_mid.max()),float(g.low_mid.min()))

    # frequency prefilter
    freq={};counts=[]
    for key,cs in C.items():
        yc=pd.Series([c['year'] for c in cs]).value_counts().to_dict();row={'family':key[0],'variant':key[1],**{f'n_{y}':int(yc.get(y,0)) for y in YEARS}};counts.append(row)
        if yc.get(2023,0)>=100 and yc.get(2024,0)>=100 and yc.get(2025,0)>=100 and yc.get(2026,0)>50:freq[key]=cs
    pd.DataFrame(counts).to_csv(OUT/'signal_counts.csv',index=False)
    schemes=FIXED+DYN;rows=[]
    for (fam,var),cs in freq.items():
        for scheme in schemes:
            m=metrics(cs,scheme)
            if not m:continue
            yd=m['years']
            if not all(y in yd for y in YEARS):continue
            row={'family':fam,'variant':var,'exit':scheme[0],'exit_type':scheme[1],'initial_rr':scheme[4] if scheme[1]=='fixed' else scheme[3],
                 **{f'n_{y}':yd[y]['n'] for y in YEARS},**{f'net_{y}':yd[y]['net'] for y in YEARS}}
            for k,v in m.items():
                if k!='years':row[k]=v
            rows.append(row)
    G=pd.DataFrame(rows);G.to_csv(OUT/'grid.csv',index=False)
    if len(G):
        base=G[(G.win_rate>=.65)&(G.net>0)&(G.net_2023>0)&(G.net_2024>0)&(G.net_2025>0)&(G.net_2026>0)]
        strict=base[base.max_loss_streak<=3].copy();near=base[(base.max_loss_streak>=4)&(base.max_loss_streak<=5)].copy()
        for X in [strict,near]:
            if len(X):X['score']=X.ev+.3*X.pf-.002*X.max_dd+.1*(X.win_rate-.65)*100
        if len(strict):strict=strict.sort_values(['initial_rr','score'],ascending=False)
        if len(near):near=near.sort_values(['initial_rr','score'],ascending=False)
    else:strict=near=G.copy()
    def diverse(X,limit=20):
        out=[];seen=set()
        for _,r in X.iterrows():
            root=str(r.family).replace('_600','').replace('_840','').replace('_09_14','').replace('_10_16','')
            if root in seen:continue
            seen.add(root);out.append(r)
            if len(out)>=limit:break
        return pd.DataFrame(out)
    ds=diverse(strict);dn=diverse(near)
    strict.to_csv(OUT/'strict.csv',index=False);near.to_csv(OUT/'near_4_5.csv',index=False);ds.to_csv(OUT/'diverse_strict.csv',index=False);dn.to_csv(OUT/'diverse_near.csv',index=False)
    summary={'generated_families':len(C),'frequent_families':len(freq),'grid_rows':len(G),'strict_count':len(strict),'near_count':len(near),'strict_families':int(strict.family.nunique()) if len(strict) else 0,'near_families':int(near.family.nunique()) if len(near) else 0,
             'strict_top':ds.head(10).to_dict('records'),'near_top':dn.head(10).to_dict('records')}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print('OUTBOX_DONE');print(json.dumps({k:v for k,v in summary.items() if k not in ['strict_top','near_top']},indent=2))
    if len(ds):print('\nSTRICT\n',ds.head(10).to_string(index=False))
    if len(dn):print('\nNEAR\n',dn.head(10).to_string(index=False))

if __name__=='__main__':main()

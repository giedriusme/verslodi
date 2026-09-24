import json
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_backtest as bt

OUT=Path('hf_search_results'); OUT.mkdir(exist_ok=True)
TPS=[5.,7.5,10.,12.5,15.,20.,25.,30.,35.,40.]
SLS=[5.,7.5,10.,12.5,15.,20.,25.,30.,35.,40.]


def make_cand(g, idx, direction, family, variant):
    loc=g.index.get_loc(idx)
    if loc+1>=len(g): return None
    e=g.iloc[loc+1]; fut=g.iloc[loc+1:]
    entry=float(e.open_ask if direction=='LONG' else e.open_bid)
    return {'date':str(g.iloc[0].local_date),'year':int(g.iloc[0].year),'family':family,'variant':variant,'direction':direction,'entry':entry,
            'hi':fut.high_bid.to_numpy(float) if direction=='LONG' else fut.high_ask.to_numpy(float),
            'lo':fut.low_bid.to_numpy(float) if direction=='LONG' else fut.low_ask.to_numpy(float),
            'cl':fut.close_bid.to_numpy(float) if direction=='LONG' else fut.close_ask.to_numpy(float)}


def evaluate(c,tp,sl):
    e=c['entry']; hi=c['hi']; lo=c['lo']; cl=c['cl']
    if len(cl)==0:return None
    if c['direction']=='LONG':
        a=np.flatnonzero(hi>=e+tp); b=np.flatnonzero(lo<=e-sl); eod=float(cl[-1]-e)
    else:
        a=np.flatnonzero(lo<=e-tp); b=np.flatnonzero(hi>=e+sl); eod=float(e-cl[-1])
    a=int(a[0]) if len(a) else None; b=int(b[0]) if len(b) else None
    if a is None and b is None:return eod,'EOD'
    if a is not None and (b is None or a<b):return float(tp),'TP'
    return -float(sl),'SL'


def metrics(cs,tp,sl):
    rows=[]
    for c in cs:
        x=evaluate(c,tp,sl)
        if x is not None:rows.append((c['date'],c['year'],x[0],x[1]))
    if not rows:return None
    r=pd.DataFrame(rows,columns=['date','year','pnl','outcome']).sort_values('date').reset_index(drop=True)
    p=r.pnl.to_numpy(float); pos=int((p>0).sum()); neg=int((p<0).sum())
    best=cur=0; bs=be=start=None
    for i,v in enumerate(p):
        if v<0:
            if cur==0:start=i
            cur+=1
            if cur>best:best=cur;bs=start;be=i
        else: cur=0
    gains=p[p>0].sum(); losses=-p[p<0].sum();eq=np.cumsum(p);peak=np.maximum.accumulate(np.r_[0.,eq]);dd=peak[1:]-eq
    yd={}
    for y,gg in r.groupby('year'):
        py=gg.pnl.to_numpy(float)
        yd[int(y)]={'n':int(len(gg)),'net':float(py.sum()),'positive':int((py>0).sum()),'negative':int((py<0).sum()),
                    'tp':int((gg.outcome=='TP').sum()),'sl':int((gg.outcome=='SL').sum()),'eod':int((gg.outcome=='EOD').sum())}
    return {'n':len(r),'positive':pos,'negative':neg,'win_rate':pos/len(r),'tp_count':int((r.outcome=='TP').sum()),
            'sl_count':int((r.outcome=='SL').sum()),'eod_count':int((r.outcome=='EOD').sum()),'net':float(p.sum()),'ev':float(p.mean()),
            'pf':float(gains/losses) if losses>0 else 99.,'max_dd':float(dd.max()) if len(dd) else 0.,'max_loss_streak':best,
            'loss_streak_start':str(r.iloc[bs].date) if bs is not None else None,'loss_streak_end':str(r.iloc[be].date) if be is not None else None,'years':yd}


def main():
    df=bt.load_data();df['mod']=(df.hour*60+df.minute).astype(int)
    # Intraday technical context, all causal.
    df['ema20']=df.close_mid.ewm(span=20,adjust=False).mean();df['ema60']=df.close_mid.ewm(span=60,adjust=False).mean();df['ema240']=df.close_mid.ewm(span=240,adjust=False).mean()
    delta=df.close_mid.diff();up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();dn=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean();rs=up/dn.replace(0,np.nan);df['rsi14']=100-(100/(1+rs))
    ma=df.close_mid.rolling(60,min_periods=45).mean();sd=df.close_mid.rolling(60,min_periods=45).std();df['z60']=(df.close_mid-ma)/sd.replace(0,np.nan)
    df['ema60_lag30']=df.ema60.shift(30)
    days=[]
    for day,g in df.groupby('local_date',sort=True):
        if pd.Timestamp(day).dayofweek<5: days.append((day,g.sort_values('timestamp').reset_index(drop=True)))
    C={}
    def add(f,v,c):
        if c is not None:C.setdefault((f,v),[]).append(c)
    prev=None
    for day,g in days:
        asia=g[(g['mod']>=60)&(g['mod']<540)]
        if len(asia)<420:
            prev=(float(g.iloc[0].open_mid),float(g.iloc[-1].close_mid),float(g.high_mid.max()),float(g.low_mid.min()))
            continue
        Aopen=float(asia.iloc[0].open_mid);Aclose=float(asia.iloc[-1].close_mid);AH=float(asia.high_mid.max());AL=float(asia.low_mid.min());AR=AH-AL;Amid=(AH+AL)/2
        for tm in [600,660,720,780,840,900]:
            ss=g[g['mod']==tm]
            if ss.empty: continue
            idx=int(ss.index[0]);r=ss.iloc[0];px=float(r.close_mid);e20=float(r.ema20);e60=float(r.ema60);e240=float(r.ema240);rsi=float(r.rsi14) if pd.notna(r.rsi14) else 50.;z=float(r.z60) if pd.notna(r.z60) else 0.;slope=float(r.ema60-r.ema60_lag30) if pd.notna(r.ema60_lag30) else 0.
            # EMA trend variants, intentionally frequent.
            if px>e60 and e60>e240:add(f'ema_trend_{tm}','basic',make_cand(g,idx,'LONG',f'ema_trend_{tm}','basic'))
            elif px<e60 and e60<e240:add(f'ema_trend_{tm}','basic',make_cand(g,idx,'SHORT',f'ema_trend_{tm}','basic'))
            if px>e20 and e20>e60 and slope>0:add(f'ema_slope_{tm}','aligned',make_cand(g,idx,'LONG',f'ema_slope_{tm}','aligned'))
            elif px<e20 and e20<e60 and slope<0:add(f'ema_slope_{tm}','aligned',make_cand(g,idx,'SHORT',f'ema_slope_{tm}','aligned'))
            # RSI momentum and fade.
            for th in [52.,55.,58.,60.]:
                if rsi>=th:
                    add(f'rsi_cont_{tm}',f'th={th}',make_cand(g,idx,'LONG',f'rsi_cont_{tm}',f'th={th}'))
                    add(f'rsi_fade_{tm}',f'th={th}',make_cand(g,idx,'SHORT',f'rsi_fade_{tm}',f'th={th}'))
                elif rsi<=100-th:
                    add(f'rsi_cont_{tm}',f'th={th}',make_cand(g,idx,'SHORT',f'rsi_cont_{tm}',f'th={th}'))
                    add(f'rsi_fade_{tm}',f'th={th}',make_cand(g,idx,'LONG',f'rsi_fade_{tm}',f'th={th}'))
            # Bollinger/z-score stretch continuation/fade.
            for th in [.5,.75,1.,1.25,1.5]:
                if z>=th:
                    add(f'z_cont_{tm}',f'z={th}',make_cand(g,idx,'LONG',f'z_cont_{tm}',f'z={th}'))
                    add(f'z_fade_{tm}',f'z={th}',make_cand(g,idx,'SHORT',f'z_fade_{tm}',f'z={th}'))
                elif z<=-th:
                    add(f'z_cont_{tm}',f'z={th}',make_cand(g,idx,'SHORT',f'z_cont_{tm}',f'z={th}'))
                    add(f'z_fade_{tm}',f'z={th}',make_cand(g,idx,'LONG',f'z_fade_{tm}',f'z={th}'))
            # Asia position continuation/fade at fixed times.
            if AR>0:
                rel=(px-AL)/AR
                for frac in [.55,.60,.65,.70,.75]:
                    if rel>=frac:
                        add(f'asia_pos_cont_{tm}',f'f={frac}',make_cand(g,idx,'LONG',f'asia_pos_cont_{tm}',f'f={frac}'))
                        add(f'asia_pos_fade_{tm}',f'f={frac}',make_cand(g,idx,'SHORT',f'asia_pos_fade_{tm}',f'f={frac}'))
                    elif rel<=1-frac:
                        add(f'asia_pos_cont_{tm}',f'f={frac}',make_cand(g,idx,'SHORT',f'asia_pos_cont_{tm}',f'f={frac}'))
                        add(f'asia_pos_fade_{tm}',f'f={frac}',make_cand(g,idx,'LONG',f'asia_pos_fade_{tm}',f'f={frac}'))
                # Midpoint + EMA confluence.
                if px>Amid and px>e60:add(f'asia_mid_ema_{tm}','cont',make_cand(g,idx,'LONG',f'asia_mid_ema_{tm}','cont'))
                elif px<Amid and px<e60:add(f'asia_mid_ema_{tm}','cont',make_cand(g,idx,'SHORT',f'asia_mid_ema_{tm}','cont'))
            # Current time vs previous day close.
            if prev is not None:
                po,pc,ph,pl=prev; gap=px-pc; pmove=pc-po
                for th in [1.,2.,3.,5.,8.]:
                    if abs(gap)>=th:
                        dc='LONG' if gap>0 else 'SHORT';dfade='SHORT' if gap>0 else 'LONG'
                        add(f'prevclose_cont_{tm}',f'th={th}',make_cand(g,idx,dc,f'prevclose_cont_{tm}',f'th={th}'))
                        add(f'prevclose_fade_{tm}',f'th={th}',make_cand(g,idx,dfade,f'prevclose_fade_{tm}',f'th={th}'))
                    if abs(pmove)>=th:
                        dc='LONG' if pmove>0 else 'SHORT';dfade='SHORT' if pmove>0 else 'LONG'
                        add(f'prevday_cont_{tm}',f'th={th}',make_cand(g,idx,dc,f'prevday_cont_{tm}',f'th={th}'))
                        add(f'prevday_fade_{tm}',f'th={th}',make_cand(g,idx,dfade,f'prevday_fade_{tm}',f'th={th}'))
        # Session momentum fixed windows, plus confluence with EMA at end.
        for a,b,label in [(480,540,'08_09'),(540,600,'09_10'),(480,600,'08_10'),(600,660,'10_11'),(720,780,'12_13'),(780,840,'13_14'),(840,900,'14_15')]:
            w=g[(g['mod']>=a)&(g['mod']<b)]
            if len(w)<max(40,int((b-a)*.7)):continue
            mv=float(w.iloc[-1].close_mid-w.iloc[0].open_mid);idx=int(w.index[-1]);rend=w.iloc[-1];e60=float(rend.ema60);px=float(rend.close_mid)
            for th in [1.,2.,3.,5.,8.,10.]:
                if abs(mv)>=th:
                    dc='LONG' if mv>0 else 'SHORT';dfade='SHORT' if mv>0 else 'LONG'
                    add(label+'_cont',f'th={th}',make_cand(g,idx,dc,label+'_cont',f'th={th}'))
                    add(label+'_fade',f'th={th}',make_cand(g,idx,dfade,label+'_fade',f'th={th}'))
                    aligned=(dc=='LONG' and px>e60) or (dc=='SHORT' and px<e60)
                    if aligned:add(label+'_ema_cont',f'th={th}',make_cand(g,idx,dc,label+'_ema_cont',f'th={th}'))
        # Asia body direction from Asia open to close, entry 10:00.
        s10=g[g['mod']==600]
        if not s10.empty:
            idx=int(s10.index[0]);amove=Aclose-Aopen
            for th in [1.,2.,3.,5.,8.]:
                if abs(amove)>=th:
                    dc='LONG' if amove>0 else 'SHORT';dfade='SHORT' if amove>0 else 'LONG'
                    add('asia_body_cont_10',f'th={th}',make_cand(g,idx,dc,'asia_body_cont_10',f'th={th}'))
                    add('asia_body_fade_10',f'th={th}',make_cand(g,idx,dfade,'asia_body_fade_10',f'th={th}'))
            # At 10 already outside Asia range = hold/continuation vs fade.
            px=float(s10.iloc[0].close_mid)
            for buf in [0.,1.,2.,3.,5.]:
                if px>=AH+buf:
                    add('asia_outside_10_cont',f'b={buf}',make_cand(g,idx,'LONG','asia_outside_10_cont',f'b={buf}'))
                    add('asia_outside_10_fade',f'b={buf}',make_cand(g,idx,'SHORT','asia_outside_10_fade',f'b={buf}'))
                elif px<=AL-buf:
                    add('asia_outside_10_cont',f'b={buf}',make_cand(g,idx,'SHORT','asia_outside_10_cont',f'b={buf}'))
                    add('asia_outside_10_fade',f'b={buf}',make_cand(g,idx,'LONG','asia_outside_10_fade',f'b={buf}'))
        prev=(float(g.iloc[0].open_mid),float(g.iloc[-1].close_mid),float(g.high_mid.max()),float(g.low_mid.min()))

    # Only strategy variants frequent enough BEFORE testing TP/SL.
    frequent={}
    count_rows=[]
    for key,cs in C.items():
        yc=pd.Series([c['year'] for c in cs]).value_counts().to_dict()
        row={'family':key[0],'variant':key[1],**{f'n_{y}':int(yc.get(y,0)) for y in [2023,2024,2025,2026]}}
        count_rows.append(row)
        if yc.get(2023,0)>=100 and yc.get(2024,0)>=100 and yc.get(2025,0)>=100 and yc.get(2026,0)>50: frequent[key]=cs
    pd.DataFrame(count_rows).to_csv(OUT/'signal_counts.csv',index=False)

    rows=[]
    for (fam,var),cs in frequent.items():
        for tp in TPS:
            for sl in SLS:
                if tp<sl:continue
                m=metrics(cs,tp,sl)
                if not m:continue
                yd=m['years']
                if not all(y in yd for y in [2023,2024,2025,2026]):continue
                row={'family':fam,'variant':var,'tp':tp,'sl':sl,'rr':tp/sl,'tp_gt_sl':tp>sl,
                     'n_2023':yd[2023]['n'],'n_2024':yd[2024]['n'],'n_2025':yd[2025]['n'],'n_2026':yd[2026]['n'],
                     'net_2023':yd[2023]['net'],'net_2024':yd[2024]['net'],'net_2025':yd[2025]['net'],'net_2026':yd[2026]['net']}
                for k,v in m.items():
                    if k!='years':row[k]=v
                rows.append(row)
    G=pd.DataFrame(rows);G.to_csv(OUT/'grid.csv',index=False)
    if len(G):
        base=(G.win_rate>=.70)&(G.net_2023>0)&(G.net_2024>0)&(G.net_2025>0)&(G.net_2026>0)&(G.pf>1)&(G.rr>=1)
        q3=G[base&(G.max_loss_streak<=3)].copy();q5=G[base&G.max_loss_streak.between(4,5)].copy()
        for d in [q3,q5]:
            if len(d):d.sort_values(['tp_gt_sl','win_rate','ev','n'],ascending=[False,False,False,False],inplace=True)
    else:q3=q5=G.copy()
    q3.to_csv(OUT/'strict.csv',index=False);q5.to_csv(OUT/'near_4_5.csv',index=False)
    def diverse(d,limit=20):
        out=[];seen=set()
        for _,r in d.iterrows():
            if r.family in seen:continue
            out.append(r);seen.add(r.family)
            if len(out)>=limit:break
        return pd.DataFrame(out)
    D3=diverse(q3);D5=diverse(q5);D3.to_csv(OUT/'diverse_strict.csv',index=False);D5.to_csv(OUT/'diverse_near.csv',index=False)
    summary={'generated_families':len(C),'frequent_families':len(frequent),'grid_rows':len(G),'strict_count':len(q3),'near_count':len(q5),'strict_family_count':q3.family.nunique() if len(q3) else 0,'near_family_count':q5.family.nunique() if len(q5) else 0,'strict_diverse':D3.to_dict('records') if len(D3) else [],'near_diverse':D5.to_dict('records') if len(D5) else []}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print('HF_SEARCH_DONE');print(json.dumps({k:v for k,v in summary.items() if k not in ('strict_diverse','near_diverse')},indent=2))
    if len(D3):print(D3[['family','variant','tp','sl','rr','n','win_rate','positive','negative','tp_count','sl_count','eod_count','net','pf','max_loss_streak','n_2023','n_2024','n_2025','n_2026','net_2023','net_2024','net_2025','net_2026']].head(20).to_string(index=False))

if __name__=='__main__':main()

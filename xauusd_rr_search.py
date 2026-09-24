import json
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_backtest as bt

OUT=Path('rr_search_results'); OUT.mkdir(exist_ok=True)
TPS=[7.5,10.,12.5,15.,20.,25.,30.,35.,40.,50.,60.]
SLS=[5.,7.5,10.,12.5,15.,20.,25.,30.,35.,40.]
TRAIN={2023,2024,2025}; OOS={2026}


def make_cand(g, idx, direction, family, variant, meta=None):
    loc=g.index.get_loc(idx)
    if loc+1>=len(g): return None
    e=g.iloc[loc+1]; fut=g.iloc[loc+1:]
    entry=float(e.open_ask if direction=='LONG' else e.open_bid)
    d={'date':str(g.iloc[0].local_date),'year':int(g.iloc[0].year),'weekday':int(pd.Timestamp(g.iloc[0].local_date).dayofweek),
       'family':family,'variant':variant,'direction':direction,'entry':entry,
       'hi':fut.high_bid.to_numpy(float) if direction=='LONG' else fut.high_ask.to_numpy(float),
       'lo':fut.low_bid.to_numpy(float) if direction=='LONG' else fut.low_ask.to_numpy(float),
       'cl':fut.close_bid.to_numpy(float) if direction=='LONG' else fut.close_ask.to_numpy(float)}
    if meta:d.update(meta)
    return d


def first_close(post,H,L,buf=0.):
    for idx,r in post.iterrows():
        if float(r.close_mid)>=H+buf:return int(idx),'LONG'
        if float(r.close_mid)<=L-buf:return int(idx),'SHORT'
    return None,None


def first_reject(post,H,L,ext=0.):
    for idx,r in post.iterrows():
        if float(r.high_mid)>=H+ext and float(r.close_mid)<H:return int(idx),'SHORT'
        if float(r.low_mid)<=L-ext and float(r.close_mid)>L:return int(idx),'LONG'
    return None,None


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


def metrics(cs,tp,sl,years=None):
    rows=[]
    for c in cs:
        if years is not None and c['year'] not in years:continue
        x=evaluate(c,tp,sl)
        if x is not None:rows.append((c['date'],c['year'],x[0],x[1]))
    if not rows:return None
    r=pd.DataFrame(rows,columns=['date','year','pnl','outcome']).sort_values('date').reset_index(drop=True)
    p=r.pnl.to_numpy(float); pos=int((p>0).sum()); neg=int((p<0).sum()); zero=int((p==0).sum())
    best=cur=0; bs=be=start=None
    for i,v in enumerate(p):
        if v<0:
            if cur==0:start=i
            cur+=1
            if cur>best:best=cur;bs=start;be=i
        else:cur=0
    gains=p[p>0].sum(); losses=-p[p<0].sum(); eq=np.cumsum(p); peak=np.maximum.accumulate(np.r_[0.,eq]); dd=peak[1:]-eq
    years_d={}
    for y,gg in r.groupby('year'):
        py=gg.pnl.to_numpy(float);years_d[int(y)]={'n':int(len(gg)),'net':float(py.sum()),'pos':int((py>0).sum()),'neg':int((py<0).sum())}
    return {'n':int(len(r)),'positive':pos,'negative':neg,'zero':zero,'win_rate':float(pos/len(r)),
            'tp_count':int((r.outcome=='TP').sum()),'sl_count':int((r.outcome=='SL').sum()),'eod_count':int((r.outcome=='EOD').sum()),
            'net':float(p.sum()),'ev':float(p.mean()),'pf':float(gains/losses) if losses>0 else 99.,'max_dd':float(dd.max()) if len(dd) else 0.,
            'max_loss_streak':int(best),'loss_streak_start':str(r.iloc[bs].date) if bs is not None else None,
            'loss_streak_end':str(r.iloc[be].date) if be is not None else None,'years':years_d}


def main():
    df=bt.load_data();df['mod']=(df.hour*60+df.minute).astype(int)
    df['ema30']=df.close_mid.ewm(span=30,adjust=False).mean();df['ema60']=df.close_mid.ewm(span=60,adjust=False).mean();df['ema240']=df.close_mid.ewm(span=240,adjust=False).mean()
    days=[]
    for day,g in df.groupby('local_date',sort=True):
        if pd.Timestamp(day).dayofweek<5:days.append((day,g.sort_values('timestamp').reset_index(drop=True)))
    C={}
    def add(fam,var,c):
        if c is not None:C.setdefault((fam,var),[]).append(c)
    prev=None
    for day,g in days:
        asia=g[(g['mod']>=60)&(g['mod']<540)];s10=g[g['mod']==600]
        # Asia close-break families, range filters and direction subsets.
        if len(asia)>=420:
            H=float(asia.high_mid.max());L=float(asia.low_mid.min());R=H-L
            for start,end,name in [(540,720,'asia_close_09_12'),(600,720,'asia_close_10_12'),(600,840,'asia_close_10_14'),(600,960,'asia_close_10_16')]:
                post=g[(g['mod']>=start)&(g['mod']<end)]
                for buf in [0.,1.,2.,3.,5.]:
                    idx,d=first_close(post,H,L,buf)
                    if d:
                        base=make_cand(g,idx,d,name,f'b={buf}',{'asia_range':R})
                        add(name,f'b={buf}',base)
                        add(name+'_'+d.lower(),f'b={buf}',base)
                        for mn in [15.,20.,25.,30.,40.,50.,60.]:
                            if R>=mn:
                                add(name+'_largerange',f'b={buf},R>={mn}',base)
                                add(name+'_largerange_'+d.lower(),f'b={buf},R>={mn}',base)
                        for mx in [10.,15.,20.,25.,30.,40.]:
                            if R<=mx:add(name+'_smallrange',f'b={buf},R<={mx}',base)
            # Asia rejection with extension.
            post=g[(g['mod']>=540)&(g['mod']<960)]
            for ext in [0.,1.,2.,3.,5.,8.]:
                idx,d=first_reject(post,H,L,ext)
                if d:
                    c=make_cand(g,idx,d,'asia_rejection',f'ext={ext}',{'asia_range':R})
                    add('asia_rejection',f'ext={ext}',c);add('asia_rejection_'+d.lower(),f'ext={ext}',c)
            # Position at 10 plus 09-10 momentum alignment/fade.
            if not s10.empty and R>0:
                r=s10.iloc[0];rel=(float(r.close_mid)-L)/R;idx=int(s10.index[0]);m09=g[(g['mod']>=540)&(g['mod']<600)]
                mv09=float(m09.iloc[-1].close_mid-m09.iloc[0].open_mid) if len(m09)>=40 else 0.
                for frac in [.75,.8,.85,.9,.95]:
                    d='LONG' if rel>=frac else ('SHORT' if rel<=1-frac else None)
                    if d:
                        c=make_cand(g,idx,d,'asia_extreme_cont',f'frac={frac}',{'asia_range':R})
                        add('asia_extreme_cont',f'frac={frac}',c);add('asia_extreme_cont_'+d.lower(),f'frac={frac}',c)
                        for th in [2.,4.,6.,8.,10.]:
                            same=(d=='LONG' and mv09>=th) or (d=='SHORT' and mv09<=-th)
                            opp=(d=='LONG' and mv09<=-th) or (d=='SHORT' and mv09>=th)
                            if same:add('asia_extreme_plus_mom',f'frac={frac},mom>={th}',c)
                            if opp:add('asia_extreme_after_pullback',f'frac={frac},pullback>={th}',c)
        # Momentum continuation/fade multiple windows and directional subsets.
        for a,b,label in [(480,540,'08_09'),(540,600,'09_10'),(600,660,'10_11'),(660,720,'11_12'),(720,780,'12_13'),(780,840,'13_14'),(840,900,'14_15'),(900,960,'15_16')]:
            w=g[(g['mod']>=a)&(g['mod']<b)]
            if len(w)>=40:
                mv=float(w.iloc[-1].close_mid-w.iloc[0].open_mid);idx=int(w.index[-1])
                for th in [3.,5.,8.,10.,12.,15.,20.,25.]:
                    if abs(mv)>=th:
                        dc='LONG' if mv>0 else 'SHORT';dfade='SHORT' if mv>0 else 'LONG'
                        cc=make_cand(g,idx,dc,label+'_mom_cont',f'th={th}')
                        cf=make_cand(g,idx,dfade,label+'_mom_fade',f'th={th}')
                        add(label+'_mom_cont',f'th={th}',cc);add(label+'_mom_cont_'+dc.lower(),f'th={th}',cc)
                        add(label+'_mom_fade',f'th={th}',cf);add(label+'_mom_fade_'+dfade.lower(),f'th={th}',cf)
        # Opening ranges.
        for a,b,end,label in [(540,570,720,'orb_09_30'),(600,615,840,'orb_10_15'),(600,630,900,'orb_10_30'),(600,660,960,'orb_10_60'),(780,810,960,'orb_13_30'),(840,870,1080,'orb_14_30'),(900,930,1140,'orb_15_30')]:
            w=g[(g['mod']>=a)&(g['mod']<b)];post=g[(g['mod']>=b)&(g['mod']<end)]
            if len(w)>=10:
                HH=float(w.high_mid.max());LL=float(w.low_mid.min())
                for buf in [0.,1.,2.,3.,5.]:
                    idx,d=first_close(post,HH,LL,buf)
                    if d:
                        c=make_cand(g,idx,d,label,f'b={buf}')
                        add(label,f'b={buf}',c);add(label+'_'+d.lower(),f'b={buf}',c)
        # EMA trend, pullback and stretch at fixed times.
        for tm in [600,660,720,840,900]:
            ss=g[g['mod']==tm]
            if ss.empty:continue
            r=ss.iloc[0];idx=int(ss.index[0]);px=float(r.close_mid);e30=float(r.ema30);e60=float(r.ema60);e240=float(r.ema240)
            for sep in [0.,2.,5.,8.,10.,15.]:
                if e60>=e240+sep and px>e30:
                    add(f'ema_trend_{tm}',f'sep={sep}',make_cand(g,idx,'LONG',f'ema_trend_{tm}',f'sep={sep}'))
                elif e60<=e240-sep and px<e30:
                    add(f'ema_trend_{tm}',f'sep={sep}',make_cand(g,idx,'SHORT',f'ema_trend_{tm}',f'sep={sep}'))
            for dist in [10.,15.,20.,25.,30.,40.,50.]:
                if px>=e240+dist:add(f'ema_stretch_fade_{tm}',f'd={dist}',make_cand(g,idx,'SHORT',f'ema_stretch_fade_{tm}',f'd={dist}'))
                elif px<=e240-dist:add(f'ema_stretch_fade_{tm}',f'd={dist}',make_cand(g,idx,'LONG',f'ema_stretch_fade_{tm}',f'd={dist}'))
        # Previous-day continuation/fade and H/L breakouts.
        if prev is not None and not s10.empty:
            ph,pl,po,pc=prev;idx=int(s10.index[0]);px=float(s10.iloc[0].close_mid);gap=px-pc;pmove=pc-po
            for th in [5.,10.,15.,20.,25.,30.,40.,50.]:
                if abs(pmove)>=th:
                    dc='LONG' if pmove>0 else 'SHORT';df='SHORT' if pmove>0 else 'LONG'
                    cc=make_cand(g,idx,dc,'prevday_direction_cont',f'th={th}');cf=make_cand(g,idx,df,'prevday_direction_fade',f'th={th}')
                    add('prevday_direction_cont',f'th={th}',cc);add('prevday_direction_fade',f'th={th}',cf)
                if abs(gap)>=th:
                    dc='LONG' if gap>0 else 'SHORT';df='SHORT' if gap>0 else 'LONG'
                    cc=make_cand(g,idx,dc,'prevclose_gap_cont',f'th={th}');cf=make_cand(g,idx,df,'prevclose_gap_fade',f'th={th}')
                    add('prevclose_gap_cont',f'th={th}',cc);add('prevclose_gap_fade',f'th={th}',cf)
            post=g[(g['mod']>=600)&(g['mod']<960)]
            for buf in [0.,1.,2.,3.,5.,8.]:
                idx2,d=first_close(post,ph,pl,buf)
                if d:add('prevday_hilo_break',f'b={buf}',make_cand(g,idx2,d,'prevday_hilo_break',f'b={buf}'))
        prev=(float(g.high_mid.max()),float(g.low_mid.min()),float(g.iloc[0].open_mid),float(g.iloc[-1].close_mid))

    rows=[]
    for (fam,var),cs in C.items():
        for tp in TPS:
            for sl in SLS:
                if tp < sl:continue
                tr=metrics(cs,tp,sl,TRAIN);oo=metrics(cs,tp,sl,OOS);aa=metrics(cs,tp,sl,None)
                if not tr or not oo or not aa:continue
                yn=[aa['years'].get(y,{}).get('net',-1e9) for y in [2023,2024,2025,2026]]
                row={'family':fam,'variant':var,'tp':tp,'sl':sl,'rr':tp/sl,
                     'strict_tp_gt_sl':tp>sl,'all_years_positive':all(x>0 for x in yn),'positive_years':sum(x>0 for x in yn)}
                for pre,m in [('train',tr),('oos',oo),('all',aa)]:
                    for k,v in m.items():
                        if k!='years':row[f'{pre}_{k}']=v
                for y,x in zip([2023,2024,2025,2026],yn):row[f'net_{y}']=x
                rows.append(row)
    G=pd.DataFrame(rows);G.to_csv(OUT/'all_grid.csv',index=False)
    base=(G.all_n>=50)&(G.oos_n>=10)&(G.all_net>0)&(G.train_net>0)&(G.oos_net>0)&(G.all_win_rate>0.80)&(G.rr>=1.0)
    Q3=G[base&(G.all_max_loss_streak<=3)].copy()
    Q45=G[base&(G.all_max_loss_streak>=4)&(G.all_max_loss_streak<=5)].copy()
    for q in (Q3,Q45):
        q['score']=q['positive_years']*5 + q['all_win_rate']*10 + np.log1p(q['all_n']) + q['all_ev'] - .002*q['all_max_dd'] + .2*q['rr']
    Q3=Q3.sort_values(['strict_tp_gt_sl','all_years_positive','positive_years','score'],ascending=False)
    Q45=Q45.sort_values(['strict_tp_gt_sl','all_years_positive','positive_years','score'],ascending=False)
    Q3.to_csv(OUT/'strict_streak_le3.csv',index=False);Q45.to_csv(OUT/'near_miss_streak_4_5.csv',index=False)
    def diverse(q,limit=20):
        picks=[];seen=set()
        for _,r in q.iterrows():
            # Normalize long/short suffixes so directional variants don't crowd list.
            root=str(r.family).replace('_long','').replace('_short','')
            if root in seen:continue
            picks.append(r);seen.add(root)
            if len(picks)>=limit:break
        return pd.DataFrame(picks)
    D3=diverse(Q3);D45=diverse(Q45)
    D3.to_csv(OUT/'diverse_streak_le3.csv',index=False);D45.to_csv(OUT/'diverse_near_miss_4_5.csv',index=False)
    with open(OUT/'summary.json','w') as f:
        json.dump({'candidate_families':len(C),'grid_rows':len(G),'strict_count':len(Q3),'near_miss_count':len(Q45),
                   'diverse_strict':D3.to_dict('records'),'diverse_near_miss':D45.to_dict('records')},f,indent=2,allow_nan=False)
    print('RR_SEARCH_DONE')
    cols=['family','variant','tp','sl','rr','all_n','all_positive','all_negative','all_win_rate','all_tp_count','all_sl_count','all_eod_count','all_net','all_pf','all_max_loss_streak','all_max_dd','net_2023','net_2024','net_2025','net_2026']
    print('STRICT')
    print(D3[cols].head(20).to_string(index=False) if len(D3) else 'NONE')
    print('NEAR')
    print(D45[cols].head(20).to_string(index=False) if len(D45) else 'NONE')

if __name__=='__main__':main()

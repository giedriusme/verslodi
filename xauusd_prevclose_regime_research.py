import json, math, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p

OUT=Path('prevclose_regime_results'); OUT.mkdir(exist_ok=True)
YEARS=list(range(2018,2027)); TRAIN=set(range(2018,2023)); OOS=set(range(2023,2027))
TPS=[6.,8.,10.,12.,15.,20.,25.,30.]
SLS=[8.,10.,12.,15.,20.,25.,30.]
HORIZONS=[240,480,1440]
VALUE=100.; START=300.; RISK=.10


def mid_o(r): return (float(r.open_bid)+float(r.open_ask))/2

def mid_h(r): return (float(r.high_bid)+float(r.high_ask))/2

def mid_l(r): return (float(r.low_bid)+float(r.low_ask))/2

def mid_c(r): return (float(r.close_bid)+float(r.close_ask))/2

def lot_size(eq,sl):
    return round(max(0,math.floor((((eq*RISK)/(sl*VALUE))+1e-12)/.01))*.01,2)

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

def daily_meta(df):
    dt=pd.to_datetime(df.dt_local)
    dates=dt.dt.date.astype(str).to_numpy(); hm=(dt.dt.hour*60+dt.dt.minute).to_numpy()
    by={}
    for i,(d,m) in enumerate(zip(dates,hm)):
        rec=by.setdefault(d,{'idx':[],'mins':{}}); rec['idx'].append(i); rec['mins'][int(m)]=i
    days=sorted(by)
    for d in days:
        ix=np.array(by[d]['idx'],dtype=int); x=df.iloc[ix]
        by[d]['open']=mid_o(x.iloc[0]); by[d]['close']=mid_c(x.iloc[-1])
        by[d]['high']=float(((x.high_bid+x.high_ask)/2).max()); by[d]['low']=float(((x.low_bid+x.low_ask)/2).min())
        by[d]['range']=by[d]['high']-by[d]['low']
        # Asia 01:00-09:00, all information known before 10:00.
        a=[i for i in ix if 60 <= int((pd.Timestamp(df.iloc[i].dt_local).hour*60+pd.Timestamp(df.iloc[i].dt_local).minute)) < 540]
        if a:
            xa=df.iloc[a]; ah=float(((xa.high_bid+xa.high_ask)/2).max()); al=float(((xa.low_bid+xa.low_ask)/2).min())
            by[d]['asia_range']=ah-al
        else: by[d]['asia_range']=np.nan
    # rolling stats using strictly PRIOR trading dates
    prev_ranges=[]; prev_abs_moves=[]; prev_asia=[]
    for j,d in enumerate(days):
        hist=days[max(0,j-20):j]
        pr=[by[z]['range'] for z in hist if np.isfinite(by[z]['range'])]
        pa=[by[z]['asia_range'] for z in hist if np.isfinite(by[z]['asia_range'])]
        pm=[]
        for k in range(max(1,j-20),j): pm.append(abs(by[days[k]]['close']-by[days[k-1]]['close']))
        by[d]['roll20_range_med']=float(np.median(pr)) if pr else np.nan
        by[d]['roll20_asia_med']=float(np.median(pa)) if pa else np.nan
        by[d]['roll20_move_med']=float(np.median(pm)) if pm else np.nan
    return by,days

def build_signals(df):
    by,days=daily_meta(df); out={'0959':[],'1000':[]}
    for j,d in enumerate(days):
        y=int(d[:4]); g=by[d]['mins']
        if y not in YEARS or j<2: continue
        pd1=days[j-1]; pd2=days[j-2]
        pc=by[pd1]['close']; ppc=by[pd2]['close']; pdr=by[pd1]['range']; pdmove=pc-ppc
        rollr=by[d]['roll20_range_med']; rolla=by[d]['roll20_asia_med']; rollm=by[d]['roll20_move_med']; ar=by[d]['asia_range']
        wd=pd.Timestamp(d).weekday()
        common={'date':d,'year':y,'weekday':wd,'prev_close':pc,'prev_range':pdr,'prev_move':pdmove,
                'prev_abs_move':abs(pdmove),'roll20_range_med':rollr,'roll20_asia_med':rolla,'roll20_move_med':rollm,
                'asia_range':ar,'prev_range_ratio':pdr/rollr if rollr and np.isfinite(rollr) else np.nan,
                'prev_move_ratio':abs(pdmove)/rollm if rollm and np.isfinite(rollm) else np.nan,
                'asia_ratio':ar/rolla if rolla and np.isfinite(rolla) else np.nan}
        if 599 in g and 600 in g:
            px=mid_c(df.iloc[g[599]])
            if px!=pc:
                s=dict(common); s.update({'side':'LONG' if px>pc else 'SHORT','signal_price':px,'entry_i':g[600],
                                          'gap_abs':abs(px-pc),'gap_ratio_range':abs(px-pc)/pdr if pdr else np.nan,
                                          'gap_ratio_roll':abs(px-pc)/rollr if rollr and np.isfinite(rollr) else np.nan})
                out['0959'].append(s)
        if 600 in g and 601 in g:
            px=mid_c(df.iloc[g[600]])
            if px!=pc:
                s=dict(common); s.update({'side':'LONG' if px>pc else 'SHORT','signal_price':px,'entry_i':g[601],
                                          'gap_abs':abs(px-pc),'gap_ratio_range':abs(px-pc)/pdr if pdr else np.nan,
                                          'gap_ratio_roll':abs(px-pc)/rollr if rollr and np.isfinite(rollr) else np.nan})
                out['1000'].append(s)
    return out

def make_filters(signals):
    # All thresholds are fixed economic ratios or train-only quantiles. No calendar-year regime filters.
    tr=pd.DataFrame([s for s in signals if s['year'] in TRAIN])
    fs=[('ALL',lambda s:True),('LONG',lambda s:s['side']=='LONG'),('SHORT',lambda s:s['side']=='SHORT')]
    for wd,nm in enumerate(['Mon','Tue','Wed','Thu','Fri']): fs.append((nm,lambda s,wd=wd:s['weekday']==wd))
    # Fixed normalized thresholds.
    specs=[
        ('gap_ratio_range',[.10,.20,.35,.50,1.0]),
        ('gap_ratio_roll',[.10,.20,.35,.50,1.0]),
        ('prev_range_ratio',[.70,1.0,1.30,1.60]),
        ('prev_move_ratio',[.70,1.0,1.50,2.0]),
        ('asia_ratio',[.70,1.0,1.30,1.60]),
    ]
    for feat,ths in specs:
        for t in ths:
            fs.append((f'{feat}>={t}',lambda s,feat=feat,t=t: np.isfinite(s.get(feat,np.nan)) and s[feat]>=t))
            fs.append((f'{feat}<{t}',lambda s,feat=feat,t=t: np.isfinite(s.get(feat,np.nan)) and s[feat]<t))
    # Train-derived quartiles as regime buckets.
    for feat in ['gap_ratio_range','gap_ratio_roll','prev_range_ratio','prev_move_ratio','asia_ratio']:
        vals=tr[feat].replace([np.inf,-np.inf],np.nan).dropna()
        if len(vals):
            q25,q50,q75=[float(x) for x in vals.quantile([.25,.5,.75])]
            for label,lo,hi in [('Q1',-np.inf,q25),('Q2',q25,q50),('Q3',q50,q75),('Q4',q75,np.inf)]:
                fs.append((f'{feat}_{label}',lambda s,feat=feat,lo=lo,hi=hi: np.isfinite(s.get(feat,np.nan)) and s[feat]>=lo and s[feat]<hi))
    # A few interpretable two-way combinations, not brute-force arbitrary conjunctions.
    bases=list(fs)
    lookup={n:f for n,f in bases}
    combos=[
        ('ThuFri',lambda s:s['weekday'] in (3,4)),
        ('MonTue',lambda s:s['weekday'] in (0,1)),
        ('HIGH_VOL',lambda s:np.isfinite(s.get('asia_ratio',np.nan)) and s['asia_ratio']>=1.0),
        ('LOW_VOL',lambda s:np.isfinite(s.get('asia_ratio',np.nan)) and s['asia_ratio']<1.0),
        ('BIG_GAP',lambda s:np.isfinite(s.get('gap_ratio_range',np.nan)) and s['gap_ratio_range']>=.35),
        ('SMALL_GAP',lambda s:np.isfinite(s.get('gap_ratio_range',np.nan)) and s['gap_ratio_range']<.35),
        ('BIG_PREV_MOVE',lambda s:np.isfinite(s.get('prev_move_ratio',np.nan)) and s['prev_move_ratio']>=1.0),
        ('SMALL_PREV_MOVE',lambda s:np.isfinite(s.get('prev_move_ratio',np.nan)) and s['prev_move_ratio']<1.0),
    ]
    fs.extend(combos)
    cm={n:f for n,f in combos}
    for a,b in [('ThuFri','HIGH_VOL'),('ThuFri','LOW_VOL'),('ThuFri','BIG_GAP'),('ThuFri','SMALL_GAP'),
                ('HIGH_VOL','BIG_GAP'),('LOW_VOL','SMALL_GAP'),('BIG_GAP','BIG_PREV_MOVE'),('SMALL_GAP','SMALL_PREV_MOVE')]:
        fa,fb=cm[a],cm[b]; fs.append((a+' & '+b,lambda s,fa=fa,fb=fb:fa(s) and fb(s)))
    return fs

def eval_subset(paths,signals,mask,tp,sl,h,years,compound=False):
    vals=[]; eq=START; peak=eq; dd=0.; stalled=False; yearpts={}; yn={}
    for j,s in enumerate(signals):
        if s['year'] not in years or not mask[j]: continue
        pts,state=outcome(paths[h][j],tp,sl); vals.append(pts); yearpts[s['year']]=yearpts.get(s['year'],0)+pts; yn[s['year']]=yn.get(s['year'],0)+1
        if compound:
            lot=lot_size(eq,sl)
            if lot<.01: stalled=True; continue
            eq=max(0.,eq+pts*VALUE*lot); peak=max(peak,eq); dd=max(dd,(peak-eq)/peak if peak else 1.)
    if not vals:return None
    a=np.array(vals,float); gp=a[a>0].sum(); gl=-a[a<0].sum()
    return {'n':len(a),'net':float(a.sum()),'ev':float(a.mean()),'wr':float((a>0).mean()),'pf':float(gp/gl) if gl>0 else 999.,
            'positive_years':sum(1 for y,v in yearpts.items() if v>0),'years_present':len(yearpts),'yearpts':yearpts,'yn':yn,
            'final':eq if compound else None,'dd':dd if compound else None,'stalled':stalled if compound else None}

def yearly_base(paths,signals,tp,sl,h):
    rows=[]; mask=np.ones(len(signals),dtype=bool)
    for y in YEARS:
        z=eval_subset(paths,signals,mask,tp,sl,h,{y})
        if z: rows.append({'year':y,**{k:v for k,v in z.items() if k not in ('yearpts','yn','final','dd','stalled')}})
    return rows

def main():
    df=p.load(); variants=build_signals(df); summary={}
    for name,signals in variants.items():
        print(name,len(signals),flush=True)
        paths={h:[trade_path(df,int(s['entry_i']),s['side'],h) for s in signals] for h in HORIZONS}
        filters=make_filters(signals)
        train_candidates=[]
        for fname,fn in filters:
            mask=np.array([bool(fn(s)) for s in signals])
            ntr=sum(mask[j] and s['year'] in TRAIN for j,s in enumerate(signals))
            if ntr<120: continue
            for tp,sl,h in itertools.product(TPS,SLS,HORIZONS):
                z=eval_subset(paths,signals,mask,tp,sl,h,TRAIN)
                if not z or z['n']<120:continue
                # Must show actual pre-2023 edge: positive EV/PF and >=3 positive train years.
                if z['ev']<=0 or z['pf']<=1 or z['positive_years']<3:continue
                score=z['ev']*(1+min(z['pf'],2))+0.05*z['positive_years']
                train_candidates.append({'variant':name,'filter':fname,'tp':tp,'sl':sl,'h':h,'train_score':score,
                    'train_n':z['n'],'train_ev':z['ev'],'train_pf':z['pf'],'train_wr':z['wr'],'train_pos_years':z['positive_years']})
        tg=pd.DataFrame(train_candidates)
        if len(tg): tg=tg.sort_values('train_score',ascending=False)
        tg.to_csv(OUT/f'{name}_train_candidates.csv',index=False)
        outs=[]
        # Expose only top train candidates to OOS.
        for r in (tg.head(100).to_dict('records') if len(tg) else []):
            fname=r['filter']; fn=dict(filters)[fname]; mask=np.array([bool(fn(s)) for s in signals]); tp=float(r['tp']);sl=float(r['sl']);h=int(r['h'])
            oo=eval_subset(paths,signals,mask,tp,sl,h,OOS); comp=eval_subset(paths,signals,mask,tp,sl,h,OOS,compound=True); fu=eval_subset(paths,signals,mask,tp,sl,h,set(YEARS))
            rec=dict(r)
            rec.update({'oos_n':oo['n'],'oos_ev':oo['ev'],'oos_pf':oo['pf'],'oos_wr':oo['wr'],'oos_pos_years':oo['positive_years'],
                        'oos_final':comp['final'],'oos_dd':comp['dd'],'oos_stalled':comp['stalled'],
                        'full_ev':fu['ev'],'full_pf':fu['pf'],'full_wr':fu['wr']})
            rec['robust']=bool(oo['ev']>0 and oo['pf']>1 and oo['positive_years']>=3 and not comp['stalled'])
            rec['robust_score']=(math.log(max(comp['final']/START,1e-9)) if comp['final'] else -20)-1.25*comp['dd']+max(0,oo['ev'])*(1+min(oo['pf'],2))
            outs.append(rec)
        od=pd.DataFrame(outs)
        if len(od):od=od.sort_values('robust_score',ascending=False)
        od.to_csv(OUT/f'{name}_oos.csv',index=False)
        robust=od[od.robust].copy() if len(od) else pd.DataFrame()
        robust.to_csv(OUT/f'{name}_robust.csv',index=False)
        # Baseline yearly profiles for both known promising raw settings.
        yr=[]
        for tp,sl,h in [(30.,30.,480),(30.,15.,480),(20.,20.,480)]:
            for row in yearly_base(paths,signals,tp,sl,h): yr.append({'tp':tp,'sl':sl,'h':h,**row})
        pd.DataFrame(yr).to_csv(OUT/f'{name}_yearly_baselines.csv',index=False)
        summary[name]={'signals':len(signals),'train_candidate_count':int(len(tg)),'robust_count':int(len(robust)),
                       'best':robust.head(10).to_dict('records') if len(robust) else [],
                       'best_oos_even_if_not_robust':od.head(5).to_dict('records') if len(od) else []}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=True)
    print(json.dumps(summary,indent=2,allow_nan=True))

if __name__=='__main__':main()

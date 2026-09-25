import json, math, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_prevclose_regime_research as r

OUT=Path('prevclose_regime_vectorized_results'); OUT.mkdir(exist_ok=True)
YEARS=r.YEARS; TRAIN=r.TRAIN; OOS=r.OOS; TPS=r.TPS; SLS=r.SLS; HORIZONS=r.HORIZONS
START=r.START; VALUE=r.VALUE

def build_signals_fast(df):
    dt=pd.to_datetime(df.dt_local)
    dates=dt.dt.strftime('%Y-%m-%d')
    mins=(dt.dt.hour*60+dt.dt.minute).to_numpy()
    mid_o=((df.open_bid+df.open_ask)/2).to_numpy(float)
    mid_h=((df.high_bid+df.high_ask)/2).to_numpy(float)
    mid_l=((df.low_bid+df.low_ask)/2).to_numpy(float)
    mid_c=((df.close_bid+df.close_ask)/2).to_numpy(float)
    tmp=pd.DataFrame({'date':dates.to_numpy(),'idx':np.arange(len(df)),'mo':mid_o,'mh':mid_h,'ml':mid_l,'mc':mid_c,'min':mins})
    g=tmp.groupby('date',sort=True)
    daily=g.agg(first_i=('idx','first'),last_i=('idx','last'),open=('mo','first'),close=('mc','last'),high=('mh','max'),low=('ml','min'))
    daily['range']=daily.high-daily.low
    asia=tmp[(tmp['min']>=60)&(tmp['min']<540)].groupby('date').agg(ah=('mh','max'),al=('ml','min'))
    daily=daily.join(asia); daily['asia_range']=daily.ah-daily.al
    daily['abs_move']=daily.close.diff().abs()
    daily['roll20_range_med']=daily['range'].shift(1).rolling(20,min_periods=1).median()
    daily['roll20_asia_med']=daily['asia_range'].shift(1).rolling(20,min_periods=1).median()
    daily['roll20_move_med']=daily['abs_move'].shift(1).rolling(20,min_periods=1).median()
    # minute index maps
    minute_maps={}
    for m in [599,600,601]:
        z=tmp[tmp['min']==m].drop_duplicates('date',keep='first').set_index('date')['idx']
        minute_maps[m]=z.to_dict()
    days=list(daily.index); out={'0959':[],'1000':[]}
    for j,d in enumerate(days):
        y=int(d[:4])
        if y not in YEARS or j<2:continue
        pd1=days[j-1]; pd2=days[j-2]
        pc=float(daily.loc[pd1,'close']); ppc=float(daily.loc[pd2,'close']); pdr=float(daily.loc[pd1,'range']); pdmove=pc-ppc
        rollr=float(daily.loc[d,'roll20_range_med']); rolla=float(daily.loc[d,'roll20_asia_med']); rollm=float(daily.loc[d,'roll20_move_med']); ar=float(daily.loc[d,'asia_range'])
        common={'date':d,'year':y,'weekday':pd.Timestamp(d).weekday(),'prev_close':pc,'prev_range':pdr,'prev_move':pdmove,'prev_abs_move':abs(pdmove),
                'roll20_range_med':rollr,'roll20_asia_med':rolla,'roll20_move_med':rollm,'asia_range':ar,
                'prev_range_ratio':pdr/rollr if np.isfinite(rollr) and rollr else np.nan,
                'prev_move_ratio':abs(pdmove)/rollm if np.isfinite(rollm) and rollm else np.nan,
                'asia_ratio':ar/rolla if np.isfinite(rolla) and rolla else np.nan}
        if d in minute_maps[599] and d in minute_maps[600]:
            si=int(minute_maps[599][d]); ei=int(minute_maps[600][d]); px=float(mid_c[si])
            if px!=pc:
                s=dict(common); s.update({'side':'LONG' if px>pc else 'SHORT','signal_price':px,'entry_i':ei,'gap_abs':abs(px-pc),
                    'gap_ratio_range':abs(px-pc)/pdr if pdr else np.nan,'gap_ratio_roll':abs(px-pc)/rollr if np.isfinite(rollr) and rollr else np.nan})
                out['0959'].append(s)
        if d in minute_maps[600] and d in minute_maps[601]:
            si=int(minute_maps[600][d]); ei=int(minute_maps[601][d]); px=float(mid_c[si])
            if px!=pc:
                s=dict(common); s.update({'side':'LONG' if px>pc else 'SHORT','signal_price':px,'entry_i':ei,'gap_abs':abs(px-pc),
                    'gap_ratio_range':abs(px-pc)/pdr if pdr else np.nan,'gap_ratio_roll':abs(px-pc)/rollr if np.isfinite(rollr) and rollr else np.nan})
                out['1000'].append(s)
    return out

def arr_stats(arr,years,mask):
    a=arr[mask]
    if not len(a):return None
    gp=a[a>0].sum(); gl=-a[a<0].sum(); ys=sorted(set(years[mask])); yp={int(y):float(arr[mask&(years==y)].sum()) for y in ys}
    return {'n':int(len(a)),'net':float(a.sum()),'ev':float(a.mean()),'wr':float((a>0).mean()),'pf':float(gp/gl) if gl>0 else 999.,'positive_years':sum(v>0 for v in yp.values()),'yearpts':yp}

def compound(arr,years,mask,sl):
    eq=START;peak=eq;dd=0.;st=False
    for pts in arr[mask]:
        lot=r.lot_size(eq,sl)
        if lot<.01:st=True;continue
        eq=max(0.,eq+float(pts)*VALUE*lot);peak=max(peak,eq);dd=max(dd,(peak-eq)/peak if peak else 1.)
    return eq,dd,st

def main():
    df=p.load(); variants=build_signals_fast(df); summary={}
    for name,signals in variants.items():
        print(name,len(signals),flush=True)
        years=np.array([s['year'] for s in signals],int); n=len(signals)
        paths={h:[r.trade_path(df,int(s['entry_i']),s['side'],h) for s in signals] for h in HORIZONS}
        cache={(tp,sl,h):np.array([r.outcome(x,tp,sl)[0] for x in paths[h]],float) for tp,sl,h in itertools.product(TPS,SLS,HORIZONS)}
        filters=r.make_filters(signals); fdict=dict(filters); cand=[]
        trainyear=np.isin(years,list(TRAIN)); oosyear=np.isin(years,list(OOS))
        for fname,fn in filters:
            fm=np.array([bool(fn(s)) for s in signals]); tm=fm&trainyear
            if tm.sum()<120:continue
            for tp,sl,h in itertools.product(TPS,SLS,HORIZONS):
                a=cache[(tp,sl,h)][tm]; gp=a[a>0].sum();gl=-a[a<0].sum();ev=float(a.mean());pf=float(gp/gl) if gl>0 else 999.;wr=float((a>0).mean())
                py=sum(float(cache[(tp,sl,h)][fm&(years==y)].sum())>0 for y in TRAIN)
                if ev>0 and pf>1 and py>=3:
                    cand.append({'variant':name,'filter':fname,'tp':tp,'sl':sl,'h':h,'train_score':ev*(1+min(pf,2))+.05*py,
                                 'train_n':int(len(a)),'train_ev':ev,'train_pf':pf,'train_wr':wr,'train_pos_years':py})
        tg=pd.DataFrame(cand)
        if len(tg):tg=tg.sort_values('train_score',ascending=False)
        tg.to_csv(OUT/f'{name}_train_candidates.csv',index=False)
        outs=[]
        for z in (tg.head(150).to_dict('records') if len(tg) else []):
            fm=np.array([bool(fdict[z['filter']](s)) for s in signals]); om=fm&oosyear; allm=fm; arr=cache[(float(z['tp']),float(z['sl']),int(z['h']))]
            oo=arr_stats(arr,years,om);fu=arr_stats(arr,years,allm);eq,dd,st=compound(arr,years,om,float(z['sl']))
            q=dict(z);q.update({'oos_n':oo['n'],'oos_ev':oo['ev'],'oos_pf':oo['pf'],'oos_wr':oo['wr'],'oos_pos_years':oo['positive_years'],'oos_yearpts':json.dumps(oo['yearpts']),
                                'oos_final':eq,'oos_dd':dd,'oos_stalled':st,'full_ev':fu['ev'],'full_pf':fu['pf'],'full_wr':fu['wr']})
            q['robust']=bool(oo['ev']>0 and oo['pf']>1 and oo['positive_years']>=3 and not st)
            q['robust_score']=math.log(max(eq/START,1e-9))-1.25*dd+max(0,oo['ev'])*(1+min(oo['pf'],2));outs.append(q)
        od=pd.DataFrame(outs)
        if len(od):od=od.sort_values('robust_score',ascending=False)
        od.to_csv(OUT/f'{name}_oos.csv',index=False);rob=od[od.robust].copy() if len(od) else pd.DataFrame();rob.to_csv(OUT/f'{name}_robust.csv',index=False)
        yr=[]
        for tp,sl,h in [(30.,30.,480),(30.,15.,480),(20.,20.,480)]:
            arr=cache[(tp,sl,h)]
            for y in YEARS:
                m=years==y
                if m.any():
                    stt=arr_stats(arr,years,m);yr.append({'tp':tp,'sl':sl,'h':h,'year':y,**{k:v for k,v in stt.items() if k!='yearpts'}})
        pd.DataFrame(yr).to_csv(OUT/f'{name}_yearly_baselines.csv',index=False)
        summary[name]={'signals':n,'train_candidates':int(len(tg)),'robust_count':int(len(rob)),'best':rob.head(12).to_dict('records') if len(rob) else [],'top_train_then_oos':od.head(8).to_dict('records') if len(od) else []}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=True)
    print(json.dumps(summary,indent=2,allow_nan=True))

if __name__=='__main__':main()

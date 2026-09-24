import json
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_backtest as bt

OUT=Path('strategy_search2_results'); OUT.mkdir(exist_ok=True)
TPS=[10.,15.,20.,25.,30.,35.]; SLS=[15.,20.,25.,30.,35.,40.]

def cand(g,pos,d,fam,param):
    if pos is None or pos+1>=len(g): return None
    e=g.iloc[pos+1]; fut=g.iloc[pos+1:]
    entry=float(e.open_ask if d=='LONG' else e.open_bid)
    return {'year':int(g.iloc[0].year),'date':str(g.iloc[0].local_date),'direction':d,'entry':entry,
            'hi':fut.high_bid.to_numpy(float) if d=='LONG' else fut.high_ask.to_numpy(float),
            'lo':fut.low_bid.to_numpy(float) if d=='LONG' else fut.low_ask.to_numpy(float),
            'cl':fut.close_bid.to_numpy(float) if d=='LONG' else fut.close_ask.to_numpy(float),
            'family':fam,'param':param}

def evalx(c,tp,sl):
    e=c['entry']; hi=c['hi']; lo=c['lo']; cl=c['cl']
    if not len(cl): return None
    if c['direction']=='LONG': a=np.flatnonzero(hi>=e+tp); b=np.flatnonzero(lo<=e-sl); eod=cl[-1]-e
    else: a=np.flatnonzero(lo<=e-tp); b=np.flatnonzero(hi>=e+sl); eod=e-cl[-1]
    a=int(a[0]) if len(a) else None; b=int(b[0]) if len(b) else None
    if a is None and b is None:return float(eod),'EOD'
    if a is not None and (b is None or a<b):return tp,'TP'
    return -sl,'SL'

def metrics(cs,tp,sl,yrs=None):
    rr=[]
    for c in cs:
        if yrs is not None and c['year'] not in yrs: continue
        x=evalx(c,tp,sl)
        if x: rr.append((c['year'],x[0],x[1]))
    if not rr:return {}
    r=pd.DataFrame(rr,columns=['year','p','o']); p=r.p.to_numpy(float); gain=p[p>0].sum(); loss=-p[p<0].sum(); eq=np.cumsum(p); pk=np.maximum.accumulate(np.r_[0,eq]); dd=pk[1:]-eq
    return {'n':len(r),'net':float(p.sum()),'ev':float(p.mean()),'pf':float(gain/loss) if loss>0 else 99.,'pos':float((p>0).mean()),'dd':float(dd.max()),'tp':int((r.o=='TP').sum()),'sl':int((r.o=='SL').sum()),'eod':int((r.o=='EOD').sum())}

def orb(g,a,b,c,d,buf,fam):
    w=g[(g.mod>=a)&(g.mod<b)]; post=g[(g.mod>=c)&(g.mod<d)]
    if len(w)<max(10,int((b-a)*.7)) or post.empty:return None
    H=float(w.high_mid.max()); L=float(w.low_mid.min())
    for idx,r in post.iterrows():
        if r.close_mid>=H+buf:return cand(g,int(idx),'LONG',fam,f'buf={buf}')
        if r.close_mid<=L-buf:return cand(g,int(idx),'SHORT',fam,f'buf={buf}')
    return None

def fixed_move(g,a,b,th,fade,fam):
    w=g[(g.mod>=a)&(g.mod<b)]
    if len(w)<max(10,int((b-a)*.7)):return None
    mv=float(w.iloc[-1].close_mid-w.iloc[0].open_mid)
    if abs(mv)<th:return None
    d='LONG' if mv>0 else 'SHORT'
    if fade:d='SHORT' if d=='LONG' else 'LONG'
    return cand(g,int(w.index[-1]),d,fam,f'th={th}')

def main():
    df=bt.load_data(); df['mod']=(df.hour*60+df.minute).astype(int)
    days=[]
    for day,g in df.groupby('local_date',sort=True):
        if pd.Timestamp(day).dayofweek<5:days.append((day,g.sort_values('timestamp').reset_index(drop=True)))
    C={}
    def add(k,x):
        if x:C.setdefault(k,[]).append(x)
    prev_close=None; prev_oc=None
    for day,g in days:
        asia=g[(g.mod>=60)&(g.mod<540)]
        # More opening ranges
        for buf in [0.,1.,2.]:
            add(('orb_10_15',f'b={buf}'),orb(g,600,615,615,960,buf,'orb_10_15'))
            add(('orb_10_60',f'b={buf}'),orb(g,600,660,660,1020,buf,'orb_10_60'))
            add(('orb_14_30',f'b={buf}'),orb(g,840,870,870,1200,buf,'orb_14_30'))
            add(('orb_15_30',f'b={buf}'),orb(g,900,930,930,1260,buf,'orb_15_30'))
        # Fixed time momentum/fades in later windows
        for th in [3.,5.,8.,12.]:
            add(('10_11_momentum_cont',f'th={th}'),fixed_move(g,600,660,th,False,'10_11_momentum_cont'))
            add(('10_11_momentum_fade',f'th={th}'),fixed_move(g,600,660,th,True,'10_11_momentum_fade'))
            add(('14_15_momentum_cont',f'th={th}'),fixed_move(g,840,900,th,False,'14_15_momentum_cont'))
            add(('14_15_momentum_fade',f'th={th}'),fixed_move(g,840,900,th,True,'14_15_momentum_fade'))
        # Asia midpoint / range-position signals at 10:00
        s=g[(g.mod>=600)&(g.mod<601)]
        if len(asia)>=420 and not s.empty:
            H=float(asia.high_mid.max()); L=float(asia.low_mid.min()); mid=(H+L)/2; rng=H-L; r=s.iloc[0]; px=float(r.close_mid); pos=int(s.index[0])
            for frac in [.25,.35,.45]:
                dist=frac*rng
                if px>=mid+dist:add(('asia_mid_fade',f'frac={frac}'),cand(g,pos,'SHORT','asia_mid_fade',f'frac={frac}'))
                elif px<=mid-dist:add(('asia_mid_fade',f'frac={frac}'),cand(g,pos,'LONG','asia_mid_fade',f'frac={frac}'))
            for frac in [.6,.75,.9]:
                rel=(px-L)/rng if rng>0 else .5
                if rel>=frac:add(('asia_position_cont',f'frac={frac}'),cand(g,pos,'LONG','asia_position_cont',f'frac={frac}'))
                elif rel<=1-frac:add(('asia_position_cont',f'frac={frac}'),cand(g,pos,'SHORT','asia_position_cont',f'frac={frac}'))
        # Gap to previous day's close at 10:00: fade / continuation
        if prev_close is not None and not s.empty:
            px=float(s.iloc[0].close_mid); gap=px-prev_close; pos=int(s.index[0])
            for th in [5.,10.,15.,20.]:
                if abs(gap)>=th:
                    dcont='LONG' if gap>0 else 'SHORT'; dfade='SHORT' if gap>0 else 'LONG'
                    add(('prevclose_gap_cont',f'th={th}'),cand(g,pos,dcont,'prevclose_gap_cont',f'th={th}'))
                    add(('prevclose_gap_fade',f'th={th}'),cand(g,pos,dfade,'prevclose_gap_fade',f'th={th}'))
        # Prior-day candle direction continuation/fade at 10
        if prev_oc is not None and not s.empty:
            mv=prev_oc[1]-prev_oc[0]; pos=int(s.index[0])
            for th in [5.,10.,20.]:
                if abs(mv)>=th:
                    dc='LONG' if mv>0 else 'SHORT'; df='SHORT' if mv>0 else 'LONG'
                    add(('prevday_direction_cont',f'th={th}'),cand(g,pos,dc,'prevday_direction_cont',f'th={th}'))
                    add(('prevday_direction_fade',f'th={th}'),cand(g,pos,df,'prevday_direction_fade',f'th={th}'))
        prev_close=float(g.iloc[-1].close_mid); prev_oc=(float(g.iloc[0].open_mid),float(g.iloc[-1].close_mid))
    rows=[]
    for (fam,param),cs in C.items():
        for tp in TPS:
            for sl in SLS:
                tr=metrics(cs,tp,sl,{2023,2024,2025}); oo=metrics(cs,tp,sl,{2026}); aa=metrics(cs,tp,sl)
                if not tr:continue
                y=[]; posyrs=0
                for yr in [2023,2024,2025]:
                    m=metrics(cs,tp,sl,{yr})
                    if m:y.append(m['ev']);posyrs+=int(m['net']>0)
                score=tr['ev']-.35*np.std(y)
                row={'family':fam,'param':param,'tp':tp,'sl':sl,'score':float(score),'positive_train_years':posyrs}
                for p,m in [('train',tr),('oos',oo),('all',aa)]:
                    for k,v in m.items():row[f'{p}_{k}']=v
                rows.append(row)
    G=pd.DataFrame(rows);G.to_csv(OUT/'all_grid.csv',index=False)
    picks=[]
    for fam,g in G.groupby('family'):
        e=g[(g.train_n>=100)&(g.train_pf>=1.03)&(g.positive_train_years>=2)]
        if e.empty:e=g[g.train_n>=80]
        if len(e):picks.append(e.sort_values(['positive_train_years','score','train_ev'],ascending=False).iloc[0])
    S=pd.DataFrame(picks).sort_values('all_net',ascending=False);S.to_csv(OUT/'selected.csv',index=False)
    R=S[(S.train_net>0)&(S.oos_net>0)&(S.all_net>0)].sort_values(['positive_train_years','all_ev'],ascending=False);R.to_csv(OUT/'robust.csv',index=False)
    print('SEARCH2_DONE');print(R.head(20).to_string(index=False))
if __name__=='__main__':main()

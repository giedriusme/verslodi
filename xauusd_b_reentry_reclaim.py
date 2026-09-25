import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_b_reentry_search as r

OUT=Path('b_reentry_reclaim_results'); OUT.mkdir(exist_ok=True)
START=300.; VALUE=100.; H=480
TRAIN=set(range(2018,2023)); OOS=set(range(2023,2027))
RECLAIMS=[0.5,1.0,1.5,2.0,3.0]
TP2S=[10.,12.,15.,20.,25.,30.]
SL2S=[6.,8.,10.,12.,15.]
WAITS=[30,60,120,240,9999]

def lot_pct(eq,sl,pct):
    raw=(eq*pct)/(sl*VALUE)
    return max(0.,math.floor((raw+1e-12)/0.01)*0.01)

def make_path(df,c):
    i=int(c['entry_i']); e=float(c['entry']); f=df.iloc[i:min(len(df),i+H)].copy()
    if len(f)==0:return None
    mid_close=(f.close_bid.to_numpy(float)+f.close_ask.to_numpy(float))/2.0
    if c['side']=='LONG':
        fav=f.high_bid.to_numpy(float)-e
        adv=e-f.low_bid.to_numpy(float)
        retouch=f.low_ask.to_numpy(float)<=e
    else:
        fav=e-f.low_ask.to_numpy(float)
        adv=f.high_ask.to_numpy(float)-e
        retouch=f.high_bid.to_numpy(float)>=e
    return {'date':c['date'],'year':int(c['year']),'side':c['side'],'entry':e,'fav':fav,'adv':adv,'retouch':retouch,
            'mid_close':mid_close,'open_ask':f.open_ask.to_numpy(float),'open_bid':f.open_bid.to_numpy(float),
            'high_bid':f.high_bid.to_numpy(float),'low_bid':f.low_bid.to_numpy(float),
            'high_ask':f.high_ask.to_numpy(float),'low_ask':f.low_ask.to_numpy(float),
            'close_bid':f.close_bid.to_numpy(float),'close_ask':f.close_ask.to_numpy(float)}

def first_trade(q):
    a=np.flatnonzero(q['fav']>=4.0); b=np.flatnonzero(q['adv']>=20.0)
    ia=int(a[0]) if len(a) else None; ib=int(b[0]) if len(b) else None
    if ib is not None and (ia is None or ib<=ia): return {'p1':-20.,'state1':'SL','i_tp1':None,'i_re':None}
    if ia is None:
        mark=(q['close_bid'][-1]-q['entry']) if q['side']=='LONG' else (q['entry']-q['close_ask'][-1])
        return {'p1':float(mark),'state1':'TIME','i_tp1':None,'i_re':None}
    rr=np.flatnonzero(q['retouch'][ia+1:]) if ia+1<len(q['retouch']) else np.array([],dtype=int)
    ir=ia+1+int(rr[0]) if len(rr) else None
    return {'p1':4.,'state1':'TP1','i_tp1':ia,'i_re':ir}

def find_reclaim(q,ir,thr,max_wait):
    if ir is None:return None
    start=ir+1; end=min(len(q['mid_close'])-1,ir+max_wait)
    if start>end:return None
    if q['side']=='LONG': cond=q['mid_close'][start:end+1] >= q['entry']+thr
    else: cond=q['mid_close'][start:end+1] <= q['entry']-thr
    h=np.flatnonzero(cond)
    if not len(h):return None
    ic=start+int(h[0])
    ie=ic+1
    if ie>=len(q['mid_close']):return None
    return ie

def second_reclaim_outcome(q,ie,tp,sl):
    if ie is None:return {'p2':0.,'state2':'NO_RECLAIM'}
    if q['side']=='LONG':
        e2=float(q['open_ask'][ie]); fav=q['high_bid'][ie:]-e2; adv=e2-q['low_bid'][ie:]
        mark=float(q['close_bid'][-1]-e2)
    else:
        e2=float(q['open_bid'][ie]); fav=e2-q['low_ask'][ie:]; adv=q['high_ask'][ie:]-e2
        mark=float(e2-q['close_ask'][-1])
    a=np.flatnonzero(fav>=tp); b=np.flatnonzero(adv>=sl)
    ia=int(a[0]) if len(a) else None; ib=int(b[0]) if len(b) else None
    if ib is not None and (ia is None or ib<=ia):return {'p2':-sl,'state2':'SL','entry2':e2}
    if ia is not None:return {'p2':tp,'state2':'TP','entry2':e2}
    return {'p2':mark,'state2':'TIME','entry2':e2}

def simulate_b(events,thr,tp,sl,wait,years):
    eq=START;peak=eq;dd=0.;exe=wins=losses=0;reclaims=0; yr={}; maxday=0.
    for ev in events:
        q=ev['path']; y=q['year']
        if y not in years:continue
        bod=eq; day=0.
        l1=lot_pct(bod,20.,.10)
        if l1<.01:continue
        p1=float(ev['p1'])*VALUE*l1; day+=p1; exe+=1; wins+=p1>0; losses+=p1<0
        mid=max(0.,bod+p1)
        if ev['i_re'] is not None and mid>0:
            ie=find_reclaim(q,ev['i_re'],thr,wait)
            if ie is not None:
                l2=lot_pct(mid,sl,.05)
                if l2>=.01:
                    z=second_reclaim_outcome(q,ie,tp,sl); p2=float(z['p2'])*VALUE*l2
                    day+=p2;reclaims+=1;exe+=1;wins+=p2>0;losses+=p2<0
        eq=max(0.,bod+day);peak=max(peak,eq);dd=max(dd,(peak-eq)/peak if peak else 1.);maxday=max(maxday,max(0.,-day/bod if bod else 0.))
        yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0:break
    for y,v in yr.items():v['return']=v['end']/v['start']-1 if v['start'] else -1
    return {'final':eq,'dd':dd,'trades':exe,'wr':wins/exe if exe else 0.,'reclaims':reclaims,'pos_years':sum(v['return']>0 for v in yr.values()),'maxdayloss':maxday,'years':yr}

def simulate_ab(Arows,events,thr,tp,sl,wait,years):
    amap={x['date']:x for x in Arows if int(x['year']) in years};bmap={e['path']['date']:e for e in events if e['path']['year'] in years}
    dates=sorted(set(amap)|set(bmap));eq=START;peak=eq;dd=0.;exe=wins=losses=reclaims=0;yr={};maxday=0.;maxrisk=0.
    for d in dates:
        bod=eq;day=0.;y=int(str(d)[:4]);arisk=0.
        if d in amap:
            a=amap[d];la=lot_pct(bod,float(a['sl']),.10)
            if la>=.01:
                pa=float(a['pnl'])*VALUE*la;day+=pa;exe+=1;wins+=pa>0;losses+=pa<0;arisk=float(a['sl'])*VALUE*la
        if d in bmap:
            ev=bmap[d];l1=lot_pct(bod,20.,.10)
            if l1>=.01:
                p1=float(ev['p1'])*VALUE*l1;day+=p1;exe+=1;wins+=p1>0;losses+=p1<0
                maxrisk=max(maxrisk,(arisk+20.*VALUE*l1)/bod if bod else 0.)
                mid=max(0.,bod+p1)
                if ev['i_re'] is not None and mid>0:
                    ie=find_reclaim(ev['path'],ev['i_re'],thr,wait)
                    if ie is not None:
                        l2=lot_pct(mid,sl,.05)
                        if l2>=.01:
                            z=second_reclaim_outcome(ev['path'],ie,tp,sl);p2=float(z['p2'])*VALUE*l2
                            day+=p2;reclaims+=1;exe+=1;wins+=p2>0;losses+=p2<0
                            maxrisk=max(maxrisk,(arisk+sl*VALUE*l2)/bod if bod else 0.)
        eq=max(0.,bod+day);peak=max(peak,eq);dd=max(dd,(peak-eq)/peak if peak else 1.);maxday=max(maxday,max(0.,-day/bod if bod else 0.))
        yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0:break
    for y,v in yr.items():v['return']=v['end']/v['start']-1 if v['start'] else -1
    return {'final':eq,'dd':dd,'trades':exe,'wr':wins/exe if exe else 0.,'reclaims':reclaims,'pos_years':sum(v['return']>0 for v in yr.values()),'maxdayloss':maxday,'maxrisk':maxrisk,'years':yr}

def main():
    df=p.load();C=p.candidates(df);Bcs=r.build_b(df,C);A=r.build_a(df,C)
    paths=[make_path(df,c) for c in Bcs];events=[]
    for q in paths:
        if q is None:continue
        z=first_trade(q);z['path']=q;events.append(z)
    rows=[]
    for thr in RECLAIMS:
      for wait in WAITS:
       for tp in TP2S:
        for sl in SL2S:
            bt=simulate_b(events,thr,tp,sl,wait,TRAIN);bo=simulate_b(events,thr,tp,sl,wait,OOS)
            at=simulate_ab(A,events,thr,tp,sl,wait,TRAIN);ao=simulate_ab(A,events,thr,tp,sl,wait,OOS)
            rec={'reclaim':thr,'wait':wait,'tp2':tp,'sl2':sl,
                 'b_train_final':bt['final'],'b_train_dd':bt['dd'],'b_train_wr':bt['wr'],'b_train_pos_years':bt['pos_years'],
                 'b_oos_final':bo['final'],'b_oos_dd':bo['dd'],'b_oos_wr':bo['wr'],'b_oos_pos_years':bo['pos_years'],'b_oos_reclaims':bo['reclaims'],
                 'ab_train_final':at['final'],'ab_train_dd':at['dd'],'ab_train_wr':at['wr'],'ab_train_pos_years':at['pos_years'],
                 'ab_oos_final':ao['final'],'ab_oos_dd':ao['dd'],'ab_oos_wr':ao['wr'],'ab_oos_pos_years':ao['pos_years'],'ab_oos_reclaims':ao['reclaims'],'ab_oos_maxdayloss':ao['maxdayloss'],'ab_oos_maxrisk':ao['maxrisk']}
            for y in sorted(OOS):rec[f'ab_ret_{y}']=ao['years'].get(y,{}).get('return')
            rows.append(rec)
    q=pd.DataFrame(rows);q.to_csv(OUT/'grid.csv',index=False)
    # Selection is based only on 2018-22; 2023-26 shown as OOS exam.
    q['train_score']=np.log(np.maximum(q.ab_train_final,1e-9)/START)-1.25*q.ab_train_dd+.08*q.ab_train_pos_years
    selected=q[(q.ab_train_pos_years>=4)].sort_values('train_score',ascending=False).head(30)
    selected.to_csv(OUT/'selected_on_train.csv',index=False)
    robust=selected[(selected.ab_oos_pos_years==4)].copy();robust.to_csv(OUT/'selected_train_and_oos4.csv',index=False)
    summary={'grid_rows':len(q),'top10_selected_on_train':selected.head(10).to_dict('records'),'oos4_among_train_top30':robust.head(10).to_dict('records')}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))
if __name__=='__main__':main()

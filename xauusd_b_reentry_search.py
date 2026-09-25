import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_compound_filter_search as s

OUT=Path('b_reentry_results');OUT.mkdir(exist_ok=True)
START=300.; VALUE=100.; H=480
TRAIN=set(range(2018,2023)); OOS=set(range(2023,2027)); FULL=set(range(2018,2027))
TP1=4.; SL1=20.; SL2=20.; TP2S=[4.,6.,8.,10.,12.,15.,20.,25.,30.]

def lot(eq,sl=20.):
    return s.lot_size(eq,sl)

def build_b(df,C):
    cs0=C[('asia_break_10_16','thu_fri')]
    F=s.feature_frame(df,cs0);masks={n:m for n,m in s.filter_defs(F,20.)}
    keep=masks['mom>=0.15'];allowed=set(np.flatnonzero(keep).tolist())
    return sorted([c for j,c in enumerate(cs0) if j in allowed],key=lambda x:x['date'])

def build_a(df,C):
    cs=C[('asia_compression_break','r070')];F=s.feature_frame(df,cs);masks={n:m for n,m in s.filter_defs(F,15.)}
    rows=s.build_engine_rows(df,cs,F,'A',15.,20.,1440)
    return s.apply_mask(rows,masks['close>=0.85'])

def make_path(df,c):
    i=int(c['entry_i']);e=float(c['entry']);f=df.iloc[i:min(len(df),i+H)]
    if len(f)==0:return None
    if c['side']=='LONG':
        fav=f.high_bid.to_numpy(float)-e
        adv=e-f.low_bid.to_numpy(float)
        retouch=(f.low_ask.to_numpy(float)<=e)
        mark=float(f.iloc[-1].close_bid-e)
    else:
        fav=e-f.low_ask.to_numpy(float)
        adv=f.high_ask.to_numpy(float)-e
        retouch=(f.high_bid.to_numpy(float)>=e)
        mark=float(e-f.iloc[-1].close_ask)
    return {'fav':fav,'adv':adv,'retouch':retouch,'mark':mark,'entry':e,'date':c['date'],'year':int(c['year']),'side':c['side']}

def first_trade(path):
    fav,adv=path['fav'],path['adv']
    a=np.flatnonzero(fav>=TP1);b=np.flatnonzero(adv>=SL1)
    ia=int(a[0]) if len(a) else None;ib=int(b[0]) if len(b) else None
    # Conservative same-M1 ordering: SL wins ties.
    if ib is not None and (ia is None or ib<=ia):
        return {'p1':-SL1,'state1':'SL','i_tp1':None,'i_re':None}
    if ia is None:
        return {'p1':path['mark'],'state1':'TIME','i_tp1':None,'i_re':None}
    # Re-entry pending order becomes eligible only from the NEXT M1 bar.
    r=np.flatnonzero(path['retouch'][ia+1:]) if ia+1<len(path['retouch']) else np.array([],dtype=int)
    ir=(ia+1+int(r[0])) if len(r) else None
    return {'p1':TP1,'state1':'TP1','i_tp1':ia,'i_re':ir}

def second_trade(path,ir,tp2):
    if ir is None:return {'p2':0.,'state2':'NO_REENTRY','hit':False,'i_exit':None}
    fav,adv=path['fav'],path['adv']
    # A pending order at the original entry is filled intrabar on the re-touch bar.
    # We never credit a TP on that same bar (its high/low may precede the fill).
    # If the stop is reached on the fill bar, count it conservatively as SL.
    if adv[ir]>=SL2:
        return {'p2':-SL2,'state2':'SL_FILLBAR','hit':True,'i_exit':ir}
    start=ir+1
    if start>=len(fav):return {'p2':path['mark'],'state2':'TIME','hit':True,'i_exit':len(fav)-1}
    a=np.flatnonzero(fav[start:]>=tp2);b=np.flatnonzero(adv[start:]>=SL2)
    ia=(start+int(a[0])) if len(a) else None;ib=(start+int(b[0])) if len(b) else None
    if ib is not None and (ia is None or ib<=ia):return {'p2':-SL2,'state2':'SL','hit':True,'i_exit':ib}
    if ia is not None:return {'p2':float(tp2),'state2':'TP2','hit':True,'i_exit':ia}
    return {'p2':path['mark'],'state2':'TIME','hit':True,'i_exit':len(fav)-1}

def prep(paths):
    out=[]
    for q in paths:
        z=first_trade(q);z.update({'path':q})
        out.append(z)
    return out

def sim_b(events,tp2,years):
    eq=START;peak=eq;dd=0.;exe=wins=losses=0;reentries=0;yr={}
    for ev in events:
        q=ev['path'];y=q['year']
        if y not in years:continue
        bod=eq;l=lot(bod,SL1)
        if l<.01:continue
        p1=float(ev['p1']);pnl1=p1*VALUE*l;exe+=1;wins+=pnl1>0;losses+=pnl1<0
        pnl2=0.
        if ev['i_re'] is not None:
            z=second_trade(q,ev['i_re'],tp2);pnl2=float(z['p2'])*VALUE*l;reentries+=1;exe+=1;wins+=pnl2>0;losses+=pnl2<0
        eq=max(0.,eq+pnl1+pnl2);peak=max(peak,eq);dd=max(dd,(peak-eq)/peak if peak else 1.)
        yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0:break
    for y,v in yr.items():v['return']=v['end']/v['start']-1 if v['start'] else -1
    return {'final':eq,'multiple':eq/START,'dd':dd,'trades':exe,'wr':wins/exe if exe else 0.,'reentries':reentries,
            'pos_years':sum(v['return']>0 for v in yr.values()),'years':yr}

def sim_portfolio(Arows,events,tp2,years):
    amap={r['date']:r for r in Arows if int(r['year']) in years}
    bmap={ev['path']['date']:ev for ev in events if ev['path']['year'] in years}
    dates=sorted(set(amap)|set(bmap));eq=START;peak=eq;dd=0.;exe=wins=losses=0;yr={};maxdayloss=0.;maxrisk=0.;reentries=0
    for d in dates:
        bod=eq;day=0.;risk=0.;y=int(str(d)[:4])
        if d in amap:
            r=amap[d];la=lot(bod,float(r['sl']))
            if la>=.01:
                pa=float(r['pnl'])*VALUE*la;day+=pa;risk+=float(r['sl'])*VALUE*la;exe+=1;wins+=pa>0;losses+=pa<0
        if d in bmap:
            ev=bmap[d];lb=lot(bod,SL1)
            if lb>=.01:
                p1=float(ev['p1'])*VALUE*lb;day+=p1;risk+=SL1*VALUE*lb;exe+=1;wins+=p1>0;losses+=p1<0
                if ev['i_re'] is not None:
                    z=second_trade(ev['path'],ev['i_re'],tp2);p2=float(z['p2'])*VALUE*lb;day+=p2;reentries+=1;exe+=1;wins+=p2>0;losses+=p2<0
                    # sequential B risk, so do not add its second SL to concurrent risk
        eq=max(0.,eq+day);peak=max(peak,eq);dd=max(dd,(peak-eq)/peak if peak else 1.)
        maxdayloss=max(maxdayloss,max(0.,-day/bod if bod else 0.));maxrisk=max(maxrisk,risk/bod if bod else 0.)
        yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0:break
    for y,v in yr.items():v['return']=v['end']/v['start']-1 if v['start'] else -1
    return {'final':eq,'multiple':eq/START,'dd':dd,'trades':exe,'wr':wins/exe if exe else 0.,'reentries':reentries,
            'pos_years':sum(v['return']>0 for v in yr.values()),'maxdayloss':maxdayloss,'maxrisk':maxrisk,'years':yr}

def excursion_stats(events,years):
    rows=[]
    for ev in events:
        q=ev['path']
        if q['year'] not in years or ev['i_tp1'] is None or ev['i_re'] is None:continue
        ir=ev['i_re'];start=ir+1
        after=q['fav'][start:] if start<len(q['fav']) else np.array([],dtype=float)
        mfe=float(np.max(after)) if len(after) else float(q['mark'])
        rec={'date':q['date'],'year':q['year'],'tp1_to_re_min':int(ir-ev['i_tp1']),'mfe_after_re':mfe}
        for t in [4.,6.,8.,10.,15.,20.,25.,30.]:
            h=np.flatnonzero(after>=t) if len(after) else np.array([],dtype=int)
            rec[f'hit_{int(t)}']=bool(len(h));rec[f'min_to_{int(t)}']=(int(h[0])+1 if len(h) else np.nan)
        rows.append(rec)
    d=pd.DataFrame(rows)
    if d.empty:return {'n':0}
    out={'n':len(d),'mean_tp1_to_re_min':float(d.tp1_to_re_min.mean()),'median_tp1_to_re_min':float(d.tp1_to_re_min.median()),
         'mean_mfe_after_re':float(d.mfe_after_re.mean()),'median_mfe_after_re':float(d.mfe_after_re.median()),
         'p75_mfe_after_re':float(d.mfe_after_re.quantile(.75)),'p90_mfe_after_re':float(d.mfe_after_re.quantile(.90))}
    for t in [4,6,8,10,15,20,25,30]:
        hit=d[f'hit_{t}'];out[f'hit_rate_{t}']=float(hit.mean())
        z=d.loc[hit,f'min_to_{t}'];out[f'mean_min_to_{t}']=float(z.mean()) if len(z) else None;out[f'median_min_to_{t}']=float(z.median()) if len(z) else None
    return out

def main():
    df=p.load();C=p.candidates(df);Bcs=build_b(df,C);A=build_a(df,C)
    paths=[make_path(df,c) for c in Bcs];paths=[q for q in paths if q is not None];events=prep(paths)
    print('B signals',len(events),'A signals',len(A),flush=True)
    stats={'train_2018_2022':excursion_stats(events,TRAIN),'oos_2023_2026':excursion_stats(events,OOS),'full_2018_2026':excursion_stats(events,FULL)}
    with open(OUT/'reentry_excursion_stats.json','w') as f:json.dump(stats,f,indent=2,allow_nan=False)

    rows=[]
    for tp2 in TP2S:
        bt=sim_b(events,tp2,TRAIN);bo=sim_b(events,tp2,OOS);bf=sim_b(events,tp2,FULL)
        pt=sim_portfolio(A,events,tp2,TRAIN);po=sim_portfolio(A,events,tp2,OOS);pf=sim_portfolio(A,events,tp2,FULL)
        rec={'tp2':tp2,
             'b_train_final':bt['final'],'b_train_dd':bt['dd'],'b_train_wr':bt['wr'],'b_train_reentries':bt['reentries'],'b_train_pos_years':bt['pos_years'],
             'b_oos_final':bo['final'],'b_oos_dd':bo['dd'],'b_oos_wr':bo['wr'],'b_oos_trades':bo['trades'],'b_oos_reentries':bo['reentries'],'b_oos_pos_years':bo['pos_years'],
             'ab_train_final':pt['final'],'ab_train_dd':pt['dd'],'ab_train_wr':pt['wr'],'ab_train_pos_years':pt['pos_years'],
             'ab_oos_final':po['final'],'ab_oos_dd':po['dd'],'ab_oos_wr':po['wr'],'ab_oos_trades':po['trades'],'ab_oos_reentries':po['reentries'],'ab_oos_pos_years':po['pos_years'],
             'ab_oos_maxrisk':po['maxrisk'],'ab_oos_maxdayloss':po['maxdayloss'],'b_full_final':bf['final'],'ab_full_final':pf['final']}
        for y,v in bo['years'].items():rec[f'b_ret_{y}']=v['return']
        for y,v in po['years'].items():rec[f'ab_ret_{y}']=v['return']
        rec['b_train_score']=math.log(max(bt['final'],1e-9)/START)-1.35*bt['dd']+.10*bt['pos_years']
        rec['ab_train_score']=math.log(max(pt['final'],1e-9)/START)-1.35*pt['dd']+.10*pt['pos_years']
        rows.append(rec)
    q=pd.DataFrame(rows)
    q.sort_values('b_train_score',ascending=False).to_csv(OUT/'results_by_b_train.csv',index=False)
    q.sort_values('ab_train_score',ascending=False).to_csv(OUT/'results_by_ab_train.csv',index=False)
    summary={'best_b_selected_on_train':q.sort_values('b_train_score',ascending=False).head(5).to_dict('records'),
             'best_ab_selected_on_train':q.sort_values('ab_train_score',ascending=False).head(5).to_dict('records'),
             'excursion_stats':stats}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':main()

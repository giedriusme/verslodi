import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_compound_filter_search as s
import xauusd_b_reentry_search as r

OUT=Path('b_reentry_fullrisk_results');OUT.mkdir(exist_ok=True)
START=300.; VALUE=100.
TRAIN=set(range(2018,2023)); OOS=set(range(2023,2027)); FULL=set(range(2018,2027))
TP2S=[4.,6.,8.,10.,12.,15.,20.,25.,30.]
SL2S=[6.,8.,10.,12.,15.,20.]


def lot(eq,sl):
    return s.lot_size(eq,sl)


def second_outcome(path,ir,tp2,sl2):
    if ir is None:return {'p2':0.,'state2':'NO_REENTRY'}
    fav,adv=path['fav'],path['adv']
    # Re-entry fills intrabar on the retouch bar. Never credit TP on fill bar.
    # If the SL is reached on fill bar, count it conservatively as SL.
    if adv[ir]>=sl2:return {'p2':-sl2,'state2':'SL_FILLBAR'}
    start=ir+1
    if start>=len(fav):return {'p2':path['mark'],'state2':'TIME'}
    a=np.flatnonzero(fav[start:]>=tp2);b=np.flatnonzero(adv[start:]>=sl2)
    ia=(start+int(a[0])) if len(a) else None;ib=(start+int(b[0])) if len(b) else None
    if ib is not None and (ia is None or ib<=ia):return {'p2':-sl2,'state2':'SL'}
    if ia is not None:return {'p2':tp2,'state2':'TP2'}
    return {'p2':path['mark'],'state2':'TIME'}


def sim_b(events,tp2,sl2,years):
    eq=START;peak=eq;dd=0.;exe=wins=losses=reentries=0;yr={};max_day_loss=0.;max_second_risk=0.
    for ev in events:
        q=ev['path']; y=q['year']
        if y not in years:continue
        bod=eq; day=0.
        l1=lot(bod,20.)
        if l1<.01:continue
        p1=float(ev['p1'])*VALUE*l1; day+=p1; exe+=1; wins+=p1>0; losses+=p1<0
        # First trade is fully closed before any re-entry. Recalculate account equity and
        # second-trade lot so SL2 risks ~10% of equity at the re-entry decision point.
        mid=max(0.,bod+p1)
        if ev['i_re'] is not None and mid>0:
            l2=lot(mid,sl2)
            if l2>=.01:
                z=second_outcome(q,ev['i_re'],tp2,sl2); p2=float(z['p2'])*VALUE*l2
                day+=p2; reentries+=1; exe+=1; wins+=p2>0; losses+=p2<0
                max_second_risk=max(max_second_risk,(sl2*VALUE*l2)/mid if mid else 0.)
        eq=max(0.,bod+day);peak=max(peak,eq);dd=max(dd,(peak-eq)/peak if peak else 1.)
        max_day_loss=max(max_day_loss,max(0.,-day/bod if bod else 0.))
        yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0:break
    for y,v in yr.items():v['return']=v['end']/v['start']-1 if v['start'] else -1
    return {'final':eq,'multiple':eq/START,'dd':dd,'trades':exe,'wr':wins/exe if exe else 0.,'reentries':reentries,
            'pos_years':sum(v['return']>0 for v in yr.values()),'max_day_loss':max_day_loss,'max_second_risk':max_second_risk,'years':yr}


def sim_portfolio(Arows,events,tp2,sl2,years):
    amap={x['date']:x for x in Arows if int(x['year']) in years}; bmap={e['path']['date']:e for e in events if e['path']['year'] in years}
    dates=sorted(set(amap)|set(bmap));eq=START;peak=eq;dd=0.;exe=wins=losses=reentries=0;yr={};maxdayloss=0.;maxrisk=0.
    for d in dates:
        bod=eq;day=0.;y=int(str(d)[:4]); concurrent_risk=0.
        # A uses day-start equity, matching prior portfolio model.
        if d in amap:
            a=amap[d];la=lot(bod,float(a['sl']))
            if la>=.01:
                pa=float(a['pnl'])*VALUE*la;day+=pa;exe+=1;wins+=pa>0;losses+=pa<0
                concurrent_risk+=float(a['sl'])*VALUE*la
        if d in bmap:
            ev=bmap[d];l1=lot(bod,20.)
            if l1>=.01:
                p1=float(ev['p1'])*VALUE*l1;day+=p1;exe+=1;wins+=p1>0;losses+=p1<0
                concurrent_risk=max(concurrent_risk,20.*VALUE*l1 + (float(amap[d]['sl'])*VALUE*lot(bod,float(amap[d]['sl'])) if d in amap else 0.))
                # At re-entry, first B trade is closed. Use equity after realized B1 only;
                # A's same-day P/L is not assumed realized intraday.
                mid=max(0.,bod+p1)
                if ev['i_re'] is not None and mid>0:
                    l2=lot(mid,sl2)
                    if l2>=.01:
                        z=second_outcome(ev['path'],ev['i_re'],tp2,sl2);p2=float(z['p2'])*VALUE*l2
                        day+=p2;reentries+=1;exe+=1;wins+=p2>0;losses+=p2<0
                        arisk=(float(amap[d]['sl'])*VALUE*lot(bod,float(amap[d]['sl']))) if d in amap else 0.
                        concurrent_risk=max(concurrent_risk,sl2*VALUE*l2+arisk)
        eq=max(0.,bod+day);peak=max(peak,eq);dd=max(dd,(peak-eq)/peak if peak else 1.)
        maxrisk=max(maxrisk,concurrent_risk/bod if bod else 0.);maxdayloss=max(maxdayloss,max(0.,-day/bod if bod else 0.))
        yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0:break
    for y,v in yr.items():v['return']=v['end']/v['start']-1 if v['start'] else -1
    return {'final':eq,'multiple':eq/START,'dd':dd,'trades':exe,'wr':wins/exe if exe else 0.,'reentries':reentries,
            'pos_years':sum(v['return']>0 for v in yr.values()),'maxdayloss':maxdayloss,'maxrisk':maxrisk,'years':yr}


def main():
    df=p.load(); C=p.candidates(df); Bcs=r.build_b(df,C); A=r.build_a(df,C)
    paths=[r.make_path(df,c) for c in Bcs];events=r.prep([q for q in paths if q is not None])
    rows=[]
    for tp2 in TP2S:
      for sl2 in SL2S:
        bt=sim_b(events,tp2,sl2,TRAIN);bo=sim_b(events,tp2,sl2,OOS);bf=sim_b(events,tp2,sl2,FULL)
        at=sim_portfolio(A,events,tp2,sl2,TRAIN);ao=sim_portfolio(A,events,tp2,sl2,OOS);af=sim_portfolio(A,events,tp2,sl2,FULL)
        z={'tp2':tp2,'sl2':sl2,
           'b_train_final':bt['final'],'b_train_dd':bt['dd'],'b_train_wr':bt['wr'],'b_train_pos_years':bt['pos_years'],
           'b_oos_final':bo['final'],'b_oos_dd':bo['dd'],'b_oos_wr':bo['wr'],'b_oos_trades':bo['trades'],'b_oos_reentries':bo['reentries'],'b_oos_pos_years':bo['pos_years'],'b_oos_maxdayloss':bo['max_day_loss'],'b_oos_max_second_risk':bo['max_second_risk'],
           'ab_train_final':at['final'],'ab_train_dd':at['dd'],'ab_train_wr':at['wr'],'ab_train_pos_years':at['pos_years'],
           'ab_oos_final':ao['final'],'ab_oos_dd':ao['dd'],'ab_oos_wr':ao['wr'],'ab_oos_trades':ao['trades'],'ab_oos_reentries':ao['reentries'],'ab_oos_pos_years':ao['pos_years'],'ab_oos_maxrisk':ao['maxrisk'],'ab_oos_maxdayloss':ao['maxdayloss'],
           'b_full_final':bf['final'],'b_full_dd':bf['dd'],'ab_full_final':af['final'],'ab_full_dd':af['dd']}
        for y,v in bo['years'].items():z[f'b_ret_{y}']=v['return']
        for y,v in ao['years'].items():z[f'ab_ret_{y}']=v['return']
        z['b_train_score']=math.log(max(bt['final'],1e-9)/START)-1.35*bt['dd']+.10*bt['pos_years']
        z['ab_train_score']=math.log(max(at['final'],1e-9)/START)-1.35*at['dd']+.10*at['pos_years']
        rows.append(z)
    q=pd.DataFrame(rows)
    q.sort_values('b_train_score',ascending=False).to_csv(OUT/'by_b_train.csv',index=False)
    q.sort_values('ab_train_score',ascending=False).to_csv(OUT/'by_ab_train.csv',index=False)
    q.sort_values('b_oos_final',ascending=False).to_csv(OUT/'by_b_oos_growth.csv',index=False)
    q.sort_values('ab_oos_final',ascending=False).to_csv(OUT/'by_ab_oos_growth.csv',index=False)
    focus=q[((q.tp2==20)&(q.sl2==6))|((q.tp2==25)&(q.sl2==8))|((q.tp2==30)&(q.sl2==6))].to_dict('records')
    summary={'focus':focus,'best_b_selected_on_train':q.sort_values('b_train_score',ascending=False).head(10).to_dict('records'),
             'best_ab_selected_on_train':q.sort_values('ab_train_score',ascending=False).head(10).to_dict('records'),
             'best_b_oos_growth':q.sort_values('b_oos_final',ascending=False).head(10).to_dict('records'),
             'best_ab_oos_growth':q.sort_values('ab_oos_final',ascending=False).head(10).to_dict('records')}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':main()

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_b_reentry_search as r

OUT=Path('b_reentry_5pct_results'); OUT.mkdir(exist_ok=True)
START=300.; VALUE=100.
TRAIN=set(range(2018,2023)); OOS=set(range(2023,2027)); FULL=set(range(2018,2027))
TP2S=[4.,6.,8.,10.,12.,15.,20.,25.,30.]
SL2S=[6.,8.,10.,12.,15.,20.]

def lot_pct(eq, sl, pct):
    raw=(eq*pct)/(sl*VALUE)
    return max(0., math.floor((raw+1e-12)/0.01)*0.01)

def second_outcome(path, ir, tp2, sl2):
    if ir is None: return {'p2':0.,'state2':'NO_REENTRY'}
    fav,adv=path['fav'],path['adv']
    if adv[ir] >= sl2: return {'p2':-sl2,'state2':'SL_FILLBAR'}
    start=ir+1
    if start>=len(fav): return {'p2':path['mark'],'state2':'TIME'}
    a=np.flatnonzero(fav[start:]>=tp2); b=np.flatnonzero(adv[start:]>=sl2)
    ia=(start+int(a[0])) if len(a) else None; ib=(start+int(b[0])) if len(b) else None
    if ib is not None and (ia is None or ib<=ia): return {'p2':-sl2,'state2':'SL'}
    if ia is not None: return {'p2':tp2,'state2':'TP'}
    return {'p2':path['mark'],'state2':'TIME'}

def simulate_b(events,tp2,sl2,years,second_risk=.05):
    eq=START; peak=eq; dd=0.; exe=wins=losses=0; reentries=0; yr={}; maxday=0.; maxrisk2=0.
    for ev in events:
        q=ev['path']; y=q['year']
        if y not in years: continue
        bod=eq; day=0.
        l1=lot_pct(bod,20.,.10)
        if l1<.01: continue
        p1=float(ev['p1'])*VALUE*l1; day += p1; exe += 1; wins += p1>0; losses += p1<0
        mid=max(0.,bod+p1)
        if ev['i_re'] is not None and mid>0:
            l2=lot_pct(mid,sl2,second_risk)
            if l2>=.01:
                z=second_outcome(q,ev['i_re'],tp2,sl2); p2=float(z['p2'])*VALUE*l2
                day += p2; reentries += 1; exe += 1; wins += p2>0; losses += p2<0
                maxrisk2=max(maxrisk2,(sl2*VALUE*l2)/mid)
        eq=max(0.,bod+day); peak=max(peak,eq); dd=max(dd,(peak-eq)/peak if peak else 1.)
        maxday=max(maxday,max(0.,-day/bod if bod else 0.)); yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0: break
    for y,v in yr.items(): v['return']=v['end']/v['start']-1 if v['start'] else -1
    return {'final':eq,'multiple':eq/START,'dd':dd,'trades':exe,'wr':wins/exe if exe else 0.,'reentries':reentries,'pos_years':sum(v['return']>0 for v in yr.values()),'maxdayloss':maxday,'max_second_risk':maxrisk2,'years':yr}

def simulate_ab(Arows,events,tp2,sl2,years,second_risk=.05):
    amap={x['date']:x for x in Arows if int(x['year']) in years}; bmap={e['path']['date']:e for e in events if e['path']['year'] in years}
    dates=sorted(set(amap)|set(bmap)); eq=START; peak=eq; dd=0.; exe=wins=losses=reentries=0; yr={}; maxday=0.; maxrisk=0.
    for d in dates:
        bod=eq; day=0.; y=int(str(d)[:4]); arisk=0.
        if d in amap:
            a=amap[d]; la=lot_pct(bod,float(a['sl']),.10)
            if la>=.01:
                pa=float(a['pnl'])*VALUE*la; day+=pa; exe+=1; wins+=pa>0; losses+=pa<0; arisk=float(a['sl'])*VALUE*la
        if d in bmap:
            ev=bmap[d]; l1=lot_pct(bod,20.,.10)
            if l1>=.01:
                p1=float(ev['p1'])*VALUE*l1; day+=p1; exe+=1; wins+=p1>0; losses+=p1<0
                maxrisk=max(maxrisk,(arisk+20.*VALUE*l1)/bod if bod else 0.)
                mid=max(0.,bod+p1)
                if ev['i_re'] is not None and mid>0:
                    l2=lot_pct(mid,sl2,second_risk)
                    if l2>=.01:
                        z=second_outcome(ev['path'],ev['i_re'],tp2,sl2); p2=float(z['p2'])*VALUE*l2
                        day+=p2; exe+=1; reentries+=1; wins+=p2>0; losses+=p2<0
                        maxrisk=max(maxrisk,(arisk+sl2*VALUE*l2)/bod if bod else 0.)
        eq=max(0.,bod+day); peak=max(peak,eq); dd=max(dd,(peak-eq)/peak if peak else 1.); maxday=max(maxday,max(0.,-day/bod if bod else 0.))
        yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0: break
    for y,v in yr.items(): v['return']=v['end']/v['start']-1 if v['start'] else -1
    return {'final':eq,'multiple':eq/START,'dd':dd,'trades':exe,'wr':wins/exe if exe else 0.,'reentries':reentries,'pos_years':sum(v['return']>0 for v in yr.values()),'maxdayloss':maxday,'maxrisk':maxrisk,'years':yr}

def trigger_only_counts(events,tp2=20.,sl2=6.,years=FULL):
    states=[]; per_year={}
    for ev in events:
        q=ev['path']; y=q['year']
        if y not in years or ev['i_re'] is None: continue
        z=second_outcome(q,ev['i_re'],tp2,sl2); st=z['state2']; states.append(st)
        py=per_year.setdefault(str(y),{'qualified':0,'TP':0,'SL':0,'TIME':0,'net_points':0.})
        py['qualified']+=1; py['net_points']+=float(z['p2'])
        if st=='TP': py['TP']+=1
        elif st.startswith('SL'): py['SL']+=1
        else: py['TIME']+=1
    tp=sum(s=='TP' for s in states); sl=sum(s.startswith('SL') for s in states); tm=sum(s=='TIME' for s in states)
    return {'qualified':len(states),'TP':tp,'SL':sl,'TIME':tm,'tp_rate_all':tp/len(states) if states else 0.,'decisive_wr':tp/(tp+sl) if tp+sl else 0.,'per_year':per_year}

def main():
    df=p.load(); C=p.candidates(df); Bcs=r.build_b(df,C); A=r.build_a(df,C)
    paths=[r.make_path(df,c) for c in Bcs]; events=r.prep([q for q in paths if q is not None])
    rows=[]
    for tp2 in TP2S:
      for sl2 in SL2S:
        bt=simulate_b(events,tp2,sl2,TRAIN); bo=simulate_b(events,tp2,sl2,OOS)
        at=simulate_ab(A,events,tp2,sl2,TRAIN); ao=simulate_ab(A,events,tp2,sl2,OOS)
        rows.append({'tp2':tp2,'sl2':sl2,'b_train_final':bt['final'],'b_train_dd':bt['dd'],'b_train_wr':bt['wr'],'b_train_pos_years':bt['pos_years'],'b_oos_final':bo['final'],'b_oos_dd':bo['dd'],'b_oos_wr':bo['wr'],'b_oos_pos_years':bo['pos_years'],'b_oos_reentries':bo['reentries'],'b_oos_maxdayloss':bo['maxdayloss'],'b_oos_max_second_risk':bo['max_second_risk'],'ab_train_final':at['final'],'ab_train_dd':at['dd'],'ab_train_wr':at['wr'],'ab_train_pos_years':at['pos_years'],'ab_oos_final':ao['final'],'ab_oos_dd':ao['dd'],'ab_oos_wr':ao['wr'],'ab_oos_pos_years':ao['pos_years'],'ab_oos_reentries':ao['reentries'],'ab_oos_maxdayloss':ao['maxdayloss'],'ab_oos_maxrisk':ao['maxrisk'],**{f'ab_ret_{y}':ao['years'].get(y,{}).get('return') for y in sorted(OOS)}})
    q=pd.DataFrame(rows)
    q.to_csv(OUT/'grid_5pct.csv',index=False)
    focus=q[((q.tp2==20)&(q.sl2==6))|((q.tp2==25)&(q.sl2==8))|((q.tp2==30)&(q.sl2==6))].to_dict('records')
    robust=q[(q.b_train_pos_years>=4)&(q.b_oos_pos_years==4)].copy(); robust['score']=np.log(robust.b_oos_final/START)-1.25*robust.b_oos_dd
    robust_ab=q[(q.ab_train_pos_years>=4)&(q.ab_oos_pos_years==4)].copy(); robust_ab['score']=np.log(robust_ab.ab_oos_final/START)-1.25*robust_ab.ab_oos_dd
    summary={'focus_5pct':focus,'best_b_robust_5pct':robust.sort_values('score',ascending=False).head(10).to_dict('records'),'best_ab_robust_5pct':robust_ab.sort_values('score',ascending=False).head(10).to_dict('records'),'trigger_only_tp20_sl6_train':trigger_only_counts(events,20.,6.,TRAIN),'trigger_only_tp20_sl6_oos':trigger_only_counts(events,20.,6.,OOS),'trigger_only_tp20_sl6_full':trigger_only_counts(events,20.,6.,FULL)}
    with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))
if __name__=='__main__': main()

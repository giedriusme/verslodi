import json, math, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_compound_filter_search as s

OUT=Path('b_partial_exit_results'); OUT.mkdir(exist_ok=True)
START=300.0; RISK=.10; LOT_STEP=.01; VALUE=100.0
TRAIN=set(range(2018,2023)); OOS=set(range(2023,2027)); FULL=set(range(2018,2027))
TP1S=[4.,5.,6.,8.,10.]; TP2S=[15.,20.,25.,30.]; SLS=[15.,20.,25.,30.]
FIRST_FRACS=[.3,.5,.7]; STOPMODES=['ORIG','BE','BE+2']; FALLBACKS=['TP1','TP2']; H=480

def lot_size(eq,sl):
    return round(max(0,math.floor((((eq*RISK)/(sl*VALUE))+1e-12)/LOT_STEP))*LOT_STEP,2)

def split_lots(total,frac):
    if total < .02-1e-12:return None
    l1=math.floor((total*frac+1e-12)/LOT_STEP)*LOT_STEP
    l1=max(.01,min(total-.01,l1)); return round(l1,2),round(total-l1,2)

def base_path(df,c):
    i=int(c['entry_i']); e=float(c['entry']); f=df.iloc[i:min(len(df),i+H)]
    if c['side']=='LONG':
        fav=f.high_bid.to_numpy(float)-e; adv=e-f.low_bid.to_numpy(float); mark=float(f.iloc[-1].close_bid-e)
    else:
        fav=e-f.low_ask.to_numpy(float); adv=f.high_ask.to_numpy(float)-e; mark=float(e-f.iloc[-1].close_ask)
    return fav,adv,mark

def partial_outcome(base,tp1,tp2,sl,mode):
    fav,adv,mark=base
    a1=np.flatnonzero(fav>=tp1); bs=np.flatnonzero(adv>=sl)
    i1=int(a1[0]) if len(a1) else None; ib0=int(bs[0]) if len(bs) else None
    if ib0 is not None and (i1 is None or ib0<=i1): return (-sl,-sl,'SL')
    if i1 is None:return (mark,mark,'TIME_PRE_TP1')
    # first part exits at tp1; moved runner stop activates next M1 bar.
    a2=np.flatnonzero(fav[i1:]>=tp2); i2=int(a2[0]) if len(a2) else None
    if mode=='ORIG':
        b=np.flatnonzero(adv[i1:]>=sl); ib=int(b[0]) if len(b) else None; stop=-sl
    else:
        nxt=i1+1; thr=0.0 if mode=='BE' else -2.0; stop=0.0 if mode=='BE' else 2.0
        if nxt<len(adv):
            b=np.flatnonzero(adv[nxt:]>=thr); ib=(int(b[0])+1) if len(b) else None
        else:ib=None
    if ib is not None and (i2 is None or ib<=i2): runner=stop; state='TP1_STOP'
    elif i2 is not None: runner=tp2; state='TP2'
    else: runner=mark; state='TP1_TIME'
    return (tp1,float(runner),state)

def full_outcome(base,tp,sl):
    fav,adv,mark=base
    a=np.flatnonzero(fav>=tp); b=np.flatnonzero(adv>=sl)
    ia=int(a[0]) if len(a) else None; ib=int(b[0]) if len(b) else None
    if ia is not None and (ib is None or ia<ib):return tp
    if ib is not None:return -sl
    return mark

def simulate(cs,cache,fullcache,scheme,years):
    tp1,tp2,sl,frac,mode,fb=scheme
    eq=START;peak=eq;maxdd=0.;wins=losses=executed=skipped=0;yr={}
    for j,c in enumerate(cs):
        if int(c['year']) not in years:continue
        y=int(c['year']);bod=eq;lot=lot_size(bod,sl)
        if lot<.01:skipped+=1;continue
        sp=split_lots(lot,frac)
        if sp is None:
            tp=tp1 if fb=='TP1' else tp2; pts=fullcache[(j,tp,sl)]; pnl=pts*VALUE*lot
        else:
            l1,l2=sp; q=cache[(j,tp1,tp2,sl,mode)]; p1,pr,state=q
            if state in ('SL','TIME_PRE_TP1'):pnl=p1*VALUE*lot
            else:pnl=p1*VALUE*l1+pr*VALUE*l2
        eq=max(0.,eq+pnl);executed+=1;wins+=pnl>0;losses+=pnl<0
        peak=max(peak,eq);maxdd=max(maxdd,(peak-eq)/peak if peak else 1.)
        yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0:break
    for y,v in yr.items():v['return']=v['end']/v['start']-1 if v['start'] else -1
    return {'final':eq,'multiple':eq/START,'maxdd':maxdd,'wr':wins/executed if executed else 0.,'executed':executed,
            'positive_years':sum(v['return']>0 for v in yr.values()),'years':yr}

def main():
    df=p.load(); C=p.candidates(df); cs0=C[('asia_break_10_16','thu_fri')]
    F=s.feature_frame(df,cs0); masks={n:m for n,m in s.filter_defs(F,20.)}; keep=masks['mom>=0.15']
    allowed=set(np.flatnonzero(keep).tolist()); cs=[c for j,c in enumerate(cs0) if j in allowed]
    cs=sorted(cs,key=lambda c:c['date']); print('filtered signals',len(cs),flush=True)
    bases=[base_path(df,c) for c in cs]
    fullcache={}
    alltps=sorted(set(TP1S+TP2S))
    for j,b in enumerate(bases):
        for tp in alltps:
            for sl in SLS: fullcache[(j,tp,sl)]=full_outcome(b,tp,sl)
    cache={}
    for j,b in enumerate(bases):
        for tp1,tp2,sl,mode in itertools.product(TP1S,TP2S,SLS,STOPMODES):
            if tp2>tp1: cache[(j,tp1,tp2,sl,mode)]=partial_outcome(b,tp1,tp2,sl,mode)
    print('cache ready',len(cache),flush=True)

    rows=[]
    for tp1,tp2,sl,frac,mode,fb in itertools.product(TP1S,TP2S,SLS,FIRST_FRACS,STOPMODES,FALLBACKS):
        if tp2<=tp1:continue
        sch=(tp1,tp2,sl,frac,mode,fb);tr=simulate(cs,cache,fullcache,sch,TRAIN)
        if tr['executed']<120 or tr['final']<=START:continue
        sc=math.log(tr['final']/START)-1.35*tr['maxdd']+.10*tr['positive_years']
        rows.append({'tp1':tp1,'tp2':tp2,'sl':sl,'first_frac':frac,'runner_stop':mode,'fallback_001':fb,
                     'train_final':tr['final'],'train_multiple':tr['multiple'],'train_dd':tr['maxdd'],'train_wr':tr['wr'],
                     'train_trades':tr['executed'],'train_pos_years':tr['positive_years'],'train_score':sc})
    grid=pd.DataFrame(rows).sort_values('train_score',ascending=False);grid.to_csv(OUT/'train_grid.csv',index=False)
    picks=grid.head(80).to_dict('records')
    out=[]
    for d in picks:
        sch=(d['tp1'],d['tp2'],d['sl'],d['first_frac'],d['runner_stop'],d['fallback_001'])
        oo=simulate(cs,cache,fullcache,sch,OOS);full=simulate(cs,cache,fullcache,sch,FULL)
        z=dict(d);z.update({'oos_final':oo['final'],'oos_multiple':oo['multiple'],'oos_dd':oo['maxdd'],'oos_wr':oo['wr'],
                           'oos_trades':oo['executed'],'oos_pos_years':oo['positive_years'],'full_final':full['final'],'full_dd':full['maxdd']})
        for y,v in oo['years'].items():z[f'ret_{y}']=v['return']
        out.append(z)
    q=pd.DataFrame(out);q['growth_dd']=q.oos_multiple/(1+3*q.oos_dd)
    q.sort_values('oos_final',ascending=False).to_csv(OUT/'selected_oos.csv',index=False)
    q[q.oos_dd<=.40].sort_values('oos_final',ascending=False).to_csv(OUT/'oos_dd40.csv',index=False)
    q[q.oos_dd<=.35].sort_values('oos_final',ascending=False).to_csv(OUT/'oos_dd35.csv',index=False)
    q.sort_values('growth_dd',ascending=False).to_csv(OUT/'growth_dd.csv',index=False)

    basesum=[]
    for name,tp,sl in [('B5_20',5.,20.),('B20_25',20.,25.),('B20_20',20.,20.)]:
        def simbase(years):
            eq=START;peak=eq;dd=0.;wins=exe=0;yr={}
            for j,c in enumerate(cs):
                if int(c['year']) not in years:continue
                bod=eq;lot=lot_size(eq,sl)
                if lot<.01:continue
                pnl=fullcache[(j,tp,sl)]*VALUE*lot;eq=max(0.,eq+pnl);exe+=1;wins+=pnl>0;peak=max(peak,eq);dd=max(dd,(peak-eq)/peak)
                y=int(c['year']);yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
            for y,v in yr.items():v['return']=v['end']/v['start']-1
            return {'final':eq,'dd':dd,'wr':wins/exe if exe else 0.,'trades':exe,'positive_years':sum(v['return']>0 for v in yr.values()),'years':yr}
        tr=simbase(TRAIN);oo=simbase(OOS);fu=simbase(FULL)
        z={'name':name,'train_final':tr['final'],'train_dd':tr['dd'],'train_wr':tr['wr'],'train_pos_years':tr['positive_years'],
           'oos_final':oo['final'],'oos_dd':oo['dd'],'oos_wr':oo['wr'],'oos_trades':oo['trades'],'oos_pos_years':oo['positive_years'],
           'full_final':fu['final'],'full_dd':fu['dd']};z.update({f'ret_{y}':v['return'] for y,v in oo['years'].items()});basesum.append(z)
    pd.DataFrame(basesum).to_csv(OUT/'baselines.csv',index=False)
    summary={'best_growth':q.sort_values('oos_final',ascending=False).head(10).to_dict('records'),
             'best_dd40':q[q.oos_dd<=.40].sort_values('oos_final',ascending=False).head(10).to_dict('records'),
             'best_dd35':q[q.oos_dd<=.35].sort_values('oos_final',ascending=False).head(10).to_dict('records'),
             'best_growth_dd':q.sort_values('growth_dd',ascending=False).head(10).to_dict('records'),'baselines':basesum}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))
if __name__=='__main__':main()

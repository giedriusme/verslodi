import json, math, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_compound_filter_search as s

OUT=Path('b_partial_exit_results'); OUT.mkdir(exist_ok=True)
START=300.0; RISK=.10; LOT_STEP=.01; VALUE=100.0
TRAIN=set(range(2018,2023)); OOS=set(range(2023,2027)); FULL=set(range(2018,2027))

# Partial-exit search around the validated B signal:
# Thu/Fri Asia close breakout 10-16 + 60m directional momentum >= 0.15.
TP1S=[4.,5.,6.,8.,10.]
TP2S=[15.,20.,25.,30.]
SLS=[15.,20.,25.,30.]
FIRST_FRACS=[.3,.5,.7]
STOPMODES=['ORIG','BE','BE+2']
FALLBACKS=['TP1','TP2']  # when total size is only 0.01 and cannot be split
H=480

def lot_size(eq,sl):
    raw=(eq*RISK)/(sl*VALUE)
    return round(math.floor((raw+1e-12)/LOT_STEP)*LOT_STEP,2)

def split_lots(total, frac):
    # Keep both legs at >= 0.01. Round first leg down to lot step.
    if total < 2*LOT_STEP-1e-12:return None
    l1=math.floor((total*frac+1e-12)/LOT_STEP)*LOT_STEP
    l1=max(LOT_STEP,min(total-LOT_STEP,l1))
    l2=round(total-l1,2)
    return round(l1,2),round(l2,2)

def path(df,c,tp1,tp2,sl,stopmode):
    i=int(c['entry_i']); e=float(c['entry']); side=c['side']; f=df.iloc[i:min(len(df),i+H)]
    if len(f)==0:return None
    if side=='LONG':
        favorable=f.high_bid.to_numpy(float)-e
        adverse=e-f.low_bid.to_numpy(float)
        mark=float(f.iloc[-1].close_bid-e)
    else:
        favorable=e-f.low_ask.to_numpy(float)
        adverse=f.high_ask.to_numpy(float)-e
        mark=float(e-f.iloc[-1].close_ask)
    a1=np.flatnonzero(favorable>=tp1); astop=np.flatnonzero(adverse>=sl)
    i1=int(a1[0]) if len(a1) else None; is0=int(astop[0]) if len(astop) else None
    # Conservative same-M1 ordering: initial SL wins ties.
    if is0 is not None and (i1 is None or is0<=i1):
        return {'whole':-sl,'leg1':-sl,'runner':-sl,'state':'SL'}
    if i1 is None:
        return {'whole':mark,'leg1':mark,'runner':mark,'state':'TIME_PRE_TP1'}
    # First target was reached before original stop.
    leg1=tp1
    # runner: evaluate from TP1 bar onward. Stop may become BE or BE+2.
    start=i1
    a2=np.flatnonzero(favorable[start:]>=tp2)
    if stopmode=='ORIG':
        stop_level=-sl
        stophit=np.flatnonzero(adverse[start:]>=sl)
    elif stopmode=='BE':
        stop_level=0.0
        # executable side returning to entry or worse
        stophit=np.flatnonzero(adverse[start:]>=0.0)
    else:  # BE+2 locks +2 after TP1
        stop_level=2.0
        # runner PnL <= +2 => adverse >= -2; use executable bid/ask path.
        stophit=np.flatnonzero(adverse[start:]>=-2.0)
    i2=int(a2[0]) if len(a2) else None; ib=int(stophit[0]) if len(stophit) else None
    # On the TP1 bar BE/BE+2 can be immediately touched by construction; to avoid impossible
    # sequence assumptions, only activate moved stop from the NEXT M1 bar. Original SL stays live on TP1 bar.
    if stopmode!='ORIG':
        # first check original SL on TP1 bar; if both TP1 and SL same bar this case would have been caught above
        # because is0<=i1. Then moved stop starts next bar.
        a2n=np.flatnonzero(favorable[start:]>=tp2)
        i2=int(a2n[0]) if len(a2n) else None
        nxt=start+1
        if nxt<len(f):
            if stopmode=='BE': stoparr=adverse[nxt:]
            else: stoparr=adverse[nxt:]
            thr=0.0 if stopmode=='BE' else -2.0
            b=np.flatnonzero(stoparr>=thr); ib=(int(b[0])+1) if len(b) else None
        else: ib=None
    # Conservative tie after activation: stop wins ties.
    if ib is not None and (i2 is None or ib<=i2): runner=stop_level; state='TP1_STOP'
    elif i2 is not None: runner=tp2; state='TP2'
    else: runner=mark; state='TP1_TIME'
    return {'whole':None,'leg1':leg1,'runner':float(runner),'state':state}

def full_exit_result(df,c,tp,sl):
    z=p.trade_result(df,c,tp,sl,H)
    return None if not z else float(z['pnl'])

def simulate(df,cs,scheme,years):
    # One B signal max per day. All sizing uses beginning-of-day equity.
    eq=START; peak=eq; maxdd=0.; wins=losses=0; executed=0; skipped=0; daily=[]; yr={}
    tp1,tp2,sl,frac,stopmode,fallback=scheme
    paths={}
    for c in cs:
        if int(c['year']) not in years:continue
        k=(c['date'],c['entry_i'])
        paths[k]=path(df,c,tp1,tp2,sl,stopmode)
    for c in sorted([c for c in cs if int(c['year']) in years],key=lambda z:z['date']):
        y=int(c['year']); bod=eq; lot=lot_size(bod,sl)
        if lot<LOT_STEP:
            skipped+=1; continue
        sp=split_lots(lot,frac); z=paths[(c['date'],c['entry_i'])]
        if sp is None:
            tp=tp1 if fallback=='TP1' else tp2
            pts=full_exit_result(df,c,tp,sl)
            pnl=pts*VALUE*lot
        else:
            l1,l2=sp
            if z['state'] in ('SL','TIME_PRE_TP1'):
                pts=z['leg1']; pnl=pts*VALUE*(l1+l2)
            else:
                pnl=z['leg1']*VALUE*l1 + z['runner']*VALUE*l2
        eq=max(0.,eq+pnl); executed+=1
        if pnl>0:wins+=1
        elif pnl<0:losses+=1
        peak=max(peak,eq); maxdd=max(maxdd,(peak-eq)/peak if peak else 1.)
        daily.append({'date':c['date'],'year':y,'bod':bod,'pnl':pnl,'eod':eq,'lot':lot})
        yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0:break
    for y,v in yr.items():v['return']=v['end']/v['start']-1 if v['start'] else -1
    return {'final':eq,'multiple':eq/START,'maxdd':maxdd,'wr':wins/executed if executed else 0.,'executed':executed,
            'wins':wins,'losses':losses,'skipped':skipped,'positive_years':sum(v['return']>0 for v in yr.values()),'years':yr}

def main():
    df=p.load(); C=p.candidates(df); cs=C[('asia_break_10_16','thu_fri')]
    F=s.feature_frame(df,cs)
    masks={n:m for n,m in s.filter_defs(F,20.)}
    keep=masks['mom>=0.15']
    allowed=set(np.flatnonzero(keep).tolist())
    cs=[c for j,c in enumerate(cs) if j in allowed]
    print('filtered signals',len(cs),flush=True)

    rows=[]
    for tp1,tp2,sl,frac,mode,fb in itertools.product(TP1S,TP2S,SLS,FIRST_FRACS,STOPMODES,FALLBACKS):
        if tp2<=tp1:continue
        scheme=(tp1,tp2,sl,frac,mode,fb)
        tr=simulate(df,cs,scheme,TRAIN)
        # pre-2023 selection score only
        if tr['executed']<120 or tr['final']<=START: continue
        score=math.log(tr['final']/START)-1.35*tr['maxdd']+0.10*tr['positive_years']
        rec={'tp1':tp1,'tp2':tp2,'sl':sl,'first_frac':frac,'runner_stop':mode,'fallback_001':fb,
             'train_final':tr['final'],'train_multiple':tr['multiple'],'train_dd':tr['maxdd'],'train_wr':tr['wr'],
             'train_trades':tr['executed'],'train_pos_years':tr['positive_years'],'train_score':score}
        rows.append(rec)
    grid=pd.DataFrame(rows).sort_values('train_score',ascending=False)
    grid.to_csv(OUT/'train_grid.csv',index=False)

    # Freeze only top diverse pre-2023 choices, then examine OOS.
    picks=[]; seen=set()
    for _,r in grid.iterrows():
        sig=(r.tp1,r.tp2,r.sl,r.first_frac,r.runner_stop,r.fallback_001)
        if sig in seen:continue
        picks.append(r.to_dict()); seen.add(sig)
        if len(picks)>=60:break

    out=[]
    for d in picks:
        sch=(d['tp1'],d['tp2'],d['sl'],d['first_frac'],d['runner_stop'],d['fallback_001'])
        oo=simulate(df,cs,sch,OOS); full=simulate(df,cs,sch,FULL)
        z=dict(d); z.update({'oos_final':oo['final'],'oos_multiple':oo['multiple'],'oos_dd':oo['maxdd'],'oos_wr':oo['wr'],
                             'oos_trades':oo['executed'],'oos_pos_years':oo['positive_years'],'full_final':full['final'],'full_dd':full['maxdd']})
        for y,v in oo['years'].items():z[f'ret_{y}']=v['return']
        out.append(z)
    q=pd.DataFrame(out)
    q['growth_dd']=q.oos_multiple/(1+3*q.oos_dd)
    q.sort_values('oos_final',ascending=False).to_csv(OUT/'selected_oos.csv',index=False)
    q[q.oos_dd<=.40].sort_values('oos_final',ascending=False).to_csv(OUT/'oos_dd40.csv',index=False)
    q[q.oos_dd<=.35].sort_values('oos_final',ascending=False).to_csv(OUT/'oos_dd35.csv',index=False)
    q.sort_values('growth_dd',ascending=False).to_csv(OUT/'growth_dd.csv',index=False)

    # Direct baselines, same B signal + mom filter.
    bases=[]
    for name,tp,sl in [('B5_20',5.,20.),('B20_25',20.,25.),('B20_20',20.,20.)]:
        # represent full exit via frac=1 concept using direct loop
        rr=[]
        for c in cs:
            z=p.trade_result(df,c,tp,sl,H)
            if z:
                w=dict(z);w['sl']=sl;w['engine']='B';rr.append(w)
        tr=s.simulate(rr,TRAIN);oo=s.simulate(rr,OOS);full=s.simulate(rr,FULL)
        bases.append({'name':name,'train_final':tr['final'],'train_dd':tr['maxdd'],'train_wr':tr['wr'],'train_pos_years':tr['positive_years'],
                      'oos_final':oo['final'],'oos_dd':oo['maxdd'],'oos_wr':oo['wr'],'oos_trades':oo['executed'],'oos_pos_years':oo['positive_years'],
                      'full_final':full['final'],'full_dd':full['maxdd'],**{f'ret_{y}':v['return'] for y,v in oo['years'].items()}})
    pd.DataFrame(bases).to_csv(OUT/'baselines.csv',index=False)

    summary={'best_oos_growth':q.sort_values('oos_final',ascending=False).head(10).to_dict('records'),
             'best_dd40':q[q.oos_dd<=.40].sort_values('oos_final',ascending=False).head(10).to_dict('records'),
             'best_dd35':q[q.oos_dd<=.35].sort_values('oos_final',ascending=False).head(10).to_dict('records'),
             'best_growth_dd':q.sort_values('growth_dd',ascending=False).head(10).to_dict('records'),
             'baselines':bases}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':main()

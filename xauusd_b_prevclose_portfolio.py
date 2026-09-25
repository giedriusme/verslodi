import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_partial_portfolio_final as pf
import xauusd_b_partial_exit_search as b
import xauusd_prevclose_regime_research as r
import xauusd_prevclose_regime_vectorized as v

OUT=Path('b_prevclose_portfolio_results');OUT.mkdir(exist_ok=True)
START=300.;VALUE=100.;YEARS=list(range(2018,2027))

def lot_custom(eq,sl,risk):
    raw=(eq*risk)/(sl*VALUE);return round(max(0,math.floor((raw+1e-12)/.01))*.01,2)

def b_pnl(base,lot):
    if lot<.01:return 0.,None
    sp=b.split_lots(lot,.5)
    if sp is None:
        pts=b.full_outcome(base,25.,20.);return pts*VALUE*lot,pts/20.
    l1,l2=sp;p1,pr,state=b.partial_outcome(base,4.,25.,20.,'ORIG')
    if state in ('SL','TIME_PRE_TP1'):
        pnl=p1*VALUE*lot; weighted=p1
    else:
        pnl=p1*VALUE*l1+pr*VALUE*l2; weighted=(p1*l1+pr*l2)/lot
    return pnl,weighted/20.

def p_pts(df,s):
    path=r.trade_path(df,int(s['entry_i']),s['side'],1440);return r.outcome(path,20.,20.)[0]

def simulate_year(df,bmap,pmap,year,prisk):
    dates=sorted(set([d for d in bmap if d.startswith(str(year))])|set([d for d in pmap if d.startswith(str(year))]))
    eq=START;peak=eq;dd=0.;maxrisk=0.;maxloss=0.;exe_b=exe_p=0;stall_b=stall_p=False
    for d in dates:
        bod=eq;day=0.;riskamt=0.
        if d in bmap:
            base=bmap[d];lb=lot_custom(bod,20.,.10)
            if lb>=.01:
                pb,_=b_pnl(base,lb);day+=pb;riskamt+=20*VALUE*lb;exe_b+=1
            else:stall_b=True
        if d in pmap:
            pts=pmap[d];lp=lot_custom(bod,20.,prisk)
            if lp>=.01:
                pp=pts*VALUE*lp;day+=pp;riskamt+=20*VALUE*lp;exe_p+=1
            else:stall_p=True
        eq=max(0.,eq+day);peak=max(peak,eq);dd=max(dd,(peak-eq)/peak if peak else 1.);maxrisk=max(maxrisk,riskamt/bod if bod else 0.);maxloss=max(maxloss,max(0.,-day/bod if bod else 0.))
    return {'year':year,'final':eq,'return_pct':(eq/START-1)*100,'dd_pct':dd*100,'max_nominal_risk_pct':maxrisk*100,'max_day_loss_pct':maxloss*100,
            'b_trades':exe_b,'p_trades':exe_p,'stall_b':stall_b,'stall_p':stall_p}

def main():
    df=p.load();C=p.candidates(df);Bcs,Bbases=pf.build_b(df,C)
    bmap={str(c['date']):base for c,base in zip(Bcs,Bbases)}
    P=[s for s in v.build_signals_fast(df)['0959'] if s['asia_ratio']>=1.0 and s['gap_ratio_range']>=.35]
    pmap={s['date']:p_pts(df,s) for s in P}
    # standardized R correlation; zero on no-signal days, plus overlapping-signal correlation.
    all_dates=sorted(set(bmap)|set(pmap));br=[];pr=[];ov_b=[];ov_p=[];same_side=overlap=0
    bside={str(c['date']):c['side'] for c in Bcs};pside={s['date']:s['side'] for s in P}
    for d in all_dates:
        rb=0.;rp=0.
        if d in bmap:
            _,rb=b_pnl(bmap[d],.02)
        if d in pmap:rp=pmap[d]/20.
        br.append(rb);pr.append(rp)
        if d in bmap and d in pmap:
            overlap+=1;ov_b.append(rb);ov_p.append(rp);same_side+=int(bside[d]==pside[d])
    daily_corr=float(np.corrcoef(br,pr)[0,1]) if len(br)>2 else np.nan
    overlap_corr=float(np.corrcoef(ov_b,ov_p)[0,1]) if len(ov_b)>2 else np.nan
    corr={'b_signal_days':len(bmap),'p_signal_days':len(pmap),'union_days':len(all_dates),'overlap_days':overlap,'overlap_pct_of_p':overlap/len(pmap) if pmap else 0.,
          'same_side_pct_overlap':same_side/overlap if overlap else None,'daily_R_corr_zero_fill':daily_corr,'overlap_R_corr':overlap_corr}
    rows=[];summary={'correlation':corr,'risk_modes':{}}
    for prisk in [.03,.05,.10]:
        rr=[simulate_year(df,bmap,pmap,y,prisk) for y in YEARS];rows+= [{'p_risk':prisk,**x} for x in rr]
        summary['risk_modes'][str(prisk)]={'profitable_years':sum(x['final']>START for x in rr),'stalled_any':sum(x['stall_b'] or x['stall_p'] for x in rr),
            'median_final':float(np.median([x['final'] for x in rr])),'worst_final':float(min(x['final'] for x in rr)),'best_final':float(max(x['final'] for x in rr)),
            'worst_dd_pct':float(max(x['dd_pct'] for x in rr)),'years':rr}
    pd.DataFrame(rows).to_csv(OUT/'annual_reset_portfolios.csv',index=False)
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=True)
    print(json.dumps(summary,indent=2,allow_nan=True))

if __name__=='__main__':main()

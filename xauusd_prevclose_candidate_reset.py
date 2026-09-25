import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_prevclose_regime_research as r
import xauusd_prevclose_regime_vectorized as v

OUT=Path('prevclose_candidate_reset_results'); OUT.mkdir(exist_ok=True)
START=300.; VALUE=100.; RISK=.10; YEARS=list(range(2018,2027))

def run_year(df,signals,year,tp=20.,sl=20.,h=1440):
    ss=[s for s in signals if s['year']==year and s['asia_ratio']>=1.0 and s['gap_ratio_range']>=.35]
    eq=START;peak=eq;dd=0.;tr=[];stalled=False
    for s in ss:
        lot=r.lot_size(eq,sl)
        if lot<.01:
            stalled=True;continue
        path=r.trade_path(df,int(s['entry_i']),s['side'],h); pts,state=r.outcome(path,tp,sl)
        pnl=pts*VALUE*lot; before=eq;eq=max(0.,eq+pnl);peak=max(peak,eq);dd=max(dd,(peak-eq)/peak if peak else 1.)
        tr.append({'date':s['date'],'side':s['side'],'lot':lot,'pts':pts,'state':state,'pnl':pnl,'before':before,'after':eq})
    a=np.array([x['pts'] for x in tr],float)
    return {'year':year,'signals':len(ss),'executed':len(tr),'final':eq,'return_pct':(eq/START-1)*100,'max_dd_pct':dd*100,'stalled':stalled,
            'wins':int((a>0).sum()) if len(a) else 0,'losses':int((a<0).sum()) if len(a) else 0,'wr':float((a>0).mean()) if len(a) else None,
            'net_points':float(a.sum()) if len(a) else 0.,'trades':tr}

def main():
    df=p.load(); variants=v.build_signals_fast(df); summary={}
    for name in ['0959','1000']:
        rows=[]; alltr=[]
        for y in YEARS:
            z=run_year(df,variants[name],y); alltr.extend([{'variant':name,**t} for t in z.pop('trades')]);rows.append(z)
        pd.DataFrame(rows).to_csv(OUT/f'{name}_yearly_reset.csv',index=False)
        pd.DataFrame(alltr).to_csv(OUT/f'{name}_trades.csv',index=False)
        summary[name]={'profitable_years':sum(x['final']>START for x in rows),'stalled_years':sum(x['stalled'] for x in rows),
                       'median_final':float(np.median([x['final'] for x in rows])),'worst_final':float(min(x['final'] for x in rows)),
                       'best_final':float(max(x['final'] for x in rows)),'worst_dd':float(max(x['max_dd_pct'] for x in rows)),
                       'years':rows}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=True)
    print(json.dumps(summary,indent=2,allow_nan=True))

if __name__=='__main__':main()

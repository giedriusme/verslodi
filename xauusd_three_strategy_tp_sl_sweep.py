import json
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p

OUT=Path('three_strategy_tp_sl_sweep_results'); OUT.mkdir(exist_ok=True)

TPS=[4.,5.,6.,7.5,8.,10.,12.,15.,20.,25.,30.]
SLS=[8.,10.,12.,15.,20.,25.,30.,35.]
SPECS=[
    ('CORE A - Asia Compression Breakout', ('asia_compression_break','r070'), 1440),
    ('CORE B - Thu/Fri Asia Close Breakout', ('asia_break_10_16','thu_fri'), 480),
    ('Satellite C - 14:00 Previous Close Continuation', ('prevclose_cont_840','all'), 1440),
]
TRAIN=list(range(2018,2023)); OOS=[2023,2024,2025,2026]

def met(rows, years=None):
    r=pd.DataFrame(rows)
    if r.empty:return None
    if years is not None:r=r[r.year.isin(years)]
    if r.empty:return None
    pnl=r.pnl.to_numpy(float); gains=pnl[pnl>0].sum(); losses=-pnl[pnl<0].sum()
    eq=np.cumsum(pnl); peak=np.maximum.accumulate(np.r_[0.,eq]); dd=peak[1:]-eq
    streak=cur=0
    for v in pnl:
        cur=cur+1 if v<0 else 0; streak=max(streak,cur)
    yd={}
    for y,g in r.groupby('year'):
        x=g.pnl.to_numpy(float)
        yd[int(y)]={'n':int(len(g)),'net':float(x.sum()),'wr':float((x>0).mean()),'tp':int((g.outcome=='TP').sum()),'sl':int((g.outcome=='SL').sum()),'time':int((g.outcome=='TIME').sum())}
    return {'n':int(len(r)),'net':float(pnl.sum()),'ev':float(pnl.mean()),'wr':float((pnl>0).mean()),'pf':float(gains/losses) if losses>0 else 99.,'dd':float(dd.max()),'streak':int(streak),'tp':int((r.outcome=='TP').sum()),'sl':int((r.outcome=='SL').sum()),'time':int((r.outcome=='TIME').sum()),'years':yd}

def signal_grid_results(df, cs, horizon):
    store={(tp,sl):[] for tp in TPS for sl in SLS}
    for c in cs:
        i=int(c['entry_i']); e=float(c['entry']); side=c['side']
        f=df.iloc[i:min(len(df),i+horizon)]
        if f.empty: continue
        if side=='LONG':
            fav=f.high_bid.to_numpy(float)-e
            adv=e-f.low_bid.to_numpy(float)
            mark=float(f.iloc[-1].close_bid-e)
        else:
            fav=e-f.low_ask.to_numpy(float)
            adv=f.high_ask.to_numpy(float)-e
            mark=float(e-f.iloc[-1].close_ask)
        tp_first={}
        sl_first={}
        for tp in TPS:
            a=np.flatnonzero(fav>=tp); tp_first[tp]=int(a[0]) if len(a) else None
        for sl in SLS:
            b=np.flatnonzero(adv>=sl); sl_first[sl]=int(b[0]) if len(b) else None
        for tp in TPS:
            ia=tp_first[tp]
            for sl in SLS:
                ib=sl_first[sl]
                if ia is not None and (ib is None or ia<ib): pnl=tp; outcome='TP'
                elif ib is not None: pnl=-sl; outcome='SL'
                else: pnl=mark; outcome='TIME'
                store[(tp,sl)].append({'year':int(c['year']),'pnl':float(pnl),'outcome':outcome})
    return store

def main():
    df=p.load(); C=p.candidates(df)
    out=[]
    for name,key,horizon in SPECS:
        cs=C.get(key,[])
        print(name, 'signals', len(cs), flush=True)
        store=signal_grid_results(df,cs,horizon)
        for tp in TPS:
            for sl in SLS:
                rows=store[(tp,sl)]
                tr=met(rows,TRAIN); oo=met(rows,OOS); al=met(rows)
                if not tr or not oo: continue
                rec={'strategy':name,'tp':tp,'sl':sl,'rr':tp/sl,'horizon_min':horizon,
                     'train_n':tr['n'],'train_net':tr['net'],'train_ev':tr['ev'],'train_wr':tr['wr'],'train_pf':tr['pf'],'train_dd':tr['dd'],'train_streak':tr['streak'],'train_tp':tr['tp'],'train_sl':tr['sl'],'train_time':tr['time'],
                     'oos_n':oo['n'],'oos_net':oo['net'],'oos_ev':oo['ev'],'oos_wr':oo['wr'],'oos_pf':oo['pf'],'oos_dd':oo['dd'],'oos_streak':oo['streak'],'oos_tp':oo['tp'],'oos_sl':oo['sl'],'oos_time':oo['time'],
                     'all_n':al['n'],'all_net':al['net'],'all_wr':al['wr'],'all_pf':al['pf']}
                for y in range(2018,2027):
                    q=al['years'].get(y,{})
                    rec[f'n_{y}']=q.get('n',0); rec[f'net_{y}']=q.get('net',0.); rec[f'wr_{y}']=q.get('wr',np.nan); rec[f'tp_{y}']=q.get('tp',0); rec[f'sl_{y}']=q.get('sl',0); rec[f'time_{y}']=q.get('time',0)
                rec['oos_all_years_positive']=all(rec[f'net_{y}']>0 for y in OOS)
                rec['train_pos_years']=sum(rec[f'net_{y}']>0 for y in TRAIN)
                out.append(rec)
    grid=pd.DataFrame(out)
    grid.to_csv(OUT/'grid.csv',index=False)

    robust=grid[(grid.oos_all_years_positive)&(grid.train_net>0)&(grid.train_pos_years>=3)&(grid.oos_pf>1)].copy()
    robust['smooth_score']=robust.oos_wr + 0.08*np.log(np.maximum(robust.oos_pf,1e-9)) + 0.001*np.maximum(robust.oos_net,0) - 0.002*robust.oos_streak
    robust['profit_score']=robust.oos_ev*np.sqrt(robust.oos_n) + 0.15*(robust.oos_pf-1) - 0.01*robust.oos_dd
    robust.to_csv(OUT/'robust.csv',index=False)

    summary={}
    for name,_,_ in SPECS:
        g=grid[grid.strategy==name]
        r=robust[robust.strategy==name]
        def rows(df,sortcol,n=8):
            if df.empty:return []
            cols=['tp','sl','rr','oos_n','oos_net','oos_ev','oos_wr','oos_pf','oos_dd','oos_streak','oos_tp','oos_sl','oos_time','train_net','train_wr','train_pf','train_pos_years']+[f'net_{y}' for y in range(2018,2027)]
            return df.sort_values(sortcol,ascending=False).head(n)[cols].to_dict('records')
        examples=g[((g.tp==6)&(g.sl==12))|((g.tp==5)&(g.sl==15))].copy()
        summary[name]={
            'signals':int(g.oos_n.max()) if len(g) else 0,
            'top_robust_by_oos_net':rows(r,'oos_net'),
            'top_robust_by_oos_wr':rows(r,'oos_wr'),
            'top_robust_by_pf':rows(r,'oos_pf'),
            'requested_examples':examples[['tp','sl','oos_n','oos_net','oos_wr','oos_pf','oos_streak','oos_tp','oos_sl','oos_time','train_net','train_wr','train_pf']+[f'net_{y}' for y in range(2018,2027)]].to_dict('records')
        }
    with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__': main()

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_compound_filter_search as s
import xauusd_b_reentry_search as r
import xauusd_b_reentry_reclaim as rr

OUT=Path('reclaim_2026_ledger'); OUT.mkdir(exist_ok=True)
START=300.0; VALUE=100.0; YEAR=2026

# Frozen system: A Smooth + B reclaim
A_TP=15.0; A_SL=20.0; A_H=1440
B1_TP=4.0; B1_SL=20.0; B_H=480
RECLAIM=2.0; RECLAIM_WAIT=120; B2_TP=15.0; B2_SL=12.0
RISK_A=0.10; RISK_B1=0.10; RISK_B2=0.05

def lot_pct(eq,sl,pct):
    if eq<=0:return 0.0
    raw=(eq*pct)/(sl*VALUE)
    return round(max(0,math.floor((raw+1e-12)/0.01))*0.01,2)

def ts(df,i):
    if i is None or i<0 or i>=len(df):return ''
    return pd.Timestamp(df.iloc[int(i)].dt_local).isoformat()

def build_a_detailed(df,C):
    cs=C[('asia_compression_break','r070')]
    F=s.feature_frame(df,cs); masks={n:m for n,m in s.filter_defs(F,A_TP)}
    keep=masks['close>=0.85']; allowed=set(np.flatnonzero(keep).tolist())
    out=[]
    for j,c in enumerate(cs):
        if j not in allowed or int(c['year'])!=YEAR:continue
        z=p.trade_result(df,c,A_TP,A_SL,A_H)
        if not z:continue
        ei=int(c['entry_i']); hold=int(z['hold']); xi=min(len(df)-1,ei+hold-1)
        out.append({
            'date':c['date'],'engine':'A','side':c['side'],'entry_i':ei,'exit_i':xi,
            'entry_time':ts(df,ei),'exit_time':ts(df,xi),'entry_price':float(c['entry']),
            'points':float(z['pnl']),'outcome':z['outcome'],'tp':A_TP,'sl':A_SL,'risk_pct':RISK_A
        })
    return out

def b_detail(df,c):
    q=rr.make_path(df,c)
    if q is None:return None
    ei=int(c['entry_i']); fav=q['fav']; adv=q['adv']
    a=np.flatnonzero(fav>=B1_TP); b=np.flatnonzero(adv>=B1_SL)
    ia=int(a[0]) if len(a) else None; ib=int(b[0]) if len(b) else None
    rec={'date':c['date'],'side':c['side'],'entry_i':ei,'entry_time':ts(df,ei),'entry_price':float(c['entry'])}
    if ib is not None and (ia is None or ib<=ia):
        rec.update({'b1_points':-B1_SL,'b1_outcome':'SL','b1_exit_off':ib,'i_tp1':None,'i_re':None})
        return rec
    if ia is None:
        mark=(q['close_bid'][-1]-q['entry']) if q['side']=='LONG' else (q['entry']-q['close_ask'][-1])
        rec.update({'b1_points':float(mark),'b1_outcome':'TIME','b1_exit_off':len(fav)-1,'i_tp1':None,'i_re':None})
        return rec
    touch=np.flatnonzero(q['retouch'][ia+1:]) if ia+1<len(q['retouch']) else np.array([],dtype=int)
    ir=ia+1+int(touch[0]) if len(touch) else None
    rec.update({'b1_points':B1_TP,'b1_outcome':'TP','b1_exit_off':ia,'i_tp1':ia,'i_re':ir})
    if ir is None:return rec
    ie=rr.find_reclaim(q,ir,RECLAIM,RECLAIM_WAIT)
    if ie is None:return rec
    z=rr.second_reclaim_outcome(q,ie,B2_TP,B2_SL)
    # recover exit offset using identical conservative ordering
    if c['side']=='LONG':
        e2=float(q['open_ask'][ie]); f2=q['high_bid'][ie:]-e2; a2=e2-q['low_bid'][ie:]
    else:
        e2=float(q['open_bid'][ie]); f2=e2-q['low_ask'][ie:]; a2=q['high_ask'][ie:]-e2
    ha=np.flatnonzero(f2>=B2_TP); hb=np.flatnonzero(a2>=B2_SL)
    xa=int(ha[0]) if len(ha) else None; xb=int(hb[0]) if len(hb) else None
    if xb is not None and (xa is None or xb<=xa): xo=ie+xb
    elif xa is not None: xo=ie+xa
    else: xo=len(q['mid_close'])-1
    rec.update({'b2_entry_off':ie,'b2_entry_price':float(z.get('entry2',e2)),'b2_points':float(z['p2']),'b2_outcome':z['state2'],'b2_exit_off':xo,
                'retest_off':ir,'reclaim_signal_off':ie-1})
    return rec

def main():
    df=p.load(); C=p.candidates(df)
    A=build_a_detailed(df,C)
    Bcs=[c for c in r.build_b(df,C) if int(c['year'])==YEAR]
    B=[b_detail(df,c) for c in Bcs]; B=[x for x in B if x is not None]
    amap={x['date']:x for x in A}; bmap={x['date']:x for x in B}
    dates=sorted(set(amap)|set(bmap))
    eq=START; daily=[]; trades=[]; wins=losses=0; peak=eq; maxdd=0.0
    for d in dates:
        bod=eq; day_pnl=0.0; Arow=amap.get(d); Brow=bmap.get(d)
        day={'date':d,'start_balance':round(bod,2)}
        # Preserve the exact sizing convention used in the validated portfolio:
        # A and B1 size from beginning-of-day equity; B2 size from BOD + realized B1 only.
        if Arow:
            la=lot_pct(bod,A_SL,RISK_A)
            pa=Arow['points']*VALUE*la if la>=.01 else 0.0
            day.update({'A_lot':la,'A_result':Arow['points'] if la>=.01 else np.nan,'A_outcome':Arow['outcome'] if la>=.01 else 'SKIP_MINLOT','A_pnl_eur':round(pa,2)})
            if la>=.01:
                trades.append({**Arow,'lot':la,'sizing_equity':bod,'pnl_eur':pa}); wins+=pa>0; losses+=pa<0
            day_pnl+=pa
        else: day.update({'A_lot':np.nan,'A_result':np.nan,'A_outcome':'','A_pnl_eur':np.nan})
        if Brow:
            l1=lot_pct(bod,B1_SL,RISK_B1)
            p1=Brow['b1_points']*VALUE*l1 if l1>=.01 else 0.0
            day.update({'B1_lot':l1,'B1_result':Brow['b1_points'] if l1>=.01 else np.nan,'B1_outcome':Brow['b1_outcome'] if l1>=.01 else 'SKIP_MINLOT','B1_pnl_eur':round(p1,2)})
            if l1>=.01:
                xi=ei=None
                b1={
                    'date':d,'engine':'B1','side':Brow['side'],'entry_i':Brow['entry_i'],'entry_time':Brow['entry_time'],
                    'exit_time':ts(df,Brow['entry_i']+Brow['b1_exit_off']),'entry_price':Brow['entry_price'],
                    'points':Brow['b1_points'],'outcome':Brow['b1_outcome'],'tp':B1_TP,'sl':B1_SL,'risk_pct':RISK_B1,
                    'lot':l1,'sizing_equity':bod,'pnl_eur':p1
                }
                trades.append(b1); wins+=p1>0; losses+=p1<0
            day_pnl+=p1
            day.update({'B2_lot':np.nan,'B2_result':np.nan,'B2_outcome':'','B2_pnl_eur':np.nan})
            if l1>=.01 and 'b2_points' in Brow:
                eq2=max(0.0,bod+p1)
                l2=lot_pct(eq2,B2_SL,RISK_B2)
                p2=Brow['b2_points']*VALUE*l2 if l2>=.01 else 0.0
                day.update({'B2_lot':l2,'B2_result':Brow['b2_points'] if l2>=.01 else np.nan,'B2_outcome':Brow['b2_outcome'] if l2>=.01 else 'SKIP_MINLOT','B2_pnl_eur':round(p2,2)})
                if l2>=.01:
                    ie=Brow['entry_i']+Brow['b2_entry_off']; xo=Brow['entry_i']+Brow['b2_exit_off']
                    trades.append({
                        'date':d,'engine':'B2','side':Brow['side'],'entry_i':ie,'entry_time':ts(df,ie),'exit_time':ts(df,xo),
                        'entry_price':Brow['b2_entry_price'],'points':Brow['b2_points'],'outcome':Brow['b2_outcome'],
                        'tp':B2_TP,'sl':B2_SL,'risk_pct':RISK_B2,'lot':l2,'sizing_equity':eq2,'pnl_eur':p2,
                        'retest_time':ts(df,Brow['entry_i']+Brow['retest_off']),
                        'reclaim_signal_time':ts(df,Brow['entry_i']+Brow['reclaim_signal_off'])
                    }); wins+=p2>0; losses+=p2<0
                day_pnl+=p2
        else:
            day.update({'B1_lot':np.nan,'B1_result':np.nan,'B1_outcome':'','B1_pnl_eur':np.nan,'B2_lot':np.nan,'B2_result':np.nan,'B2_outcome':'','B2_pnl_eur':np.nan})
        eq=max(0.0,bod+day_pnl); peak=max(peak,eq); maxdd=max(maxdd,(peak-eq)/peak if peak else 1.0)
        day['day_pnl_eur']=round(day_pnl,2); day['end_balance']=round(eq,2); daily.append(day)
    td=pd.DataFrame(trades).sort_values(['date','entry_time','engine']) if trades else pd.DataFrame()
    dd=pd.DataFrame(daily)
    # Human compact calendar: points only, as requested in earlier manual checks.
    compact=dd[['date','A_result','B1_result','B2_result','start_balance','day_pnl_eur','end_balance']].copy()
    td.to_csv(OUT/'trade_ledger_2026.csv',index=False)
    dd.to_csv(OUT/'daily_ledger_2026.csv',index=False)
    compact.to_csv(OUT/'compact_calendar_2026.csv',index=False)
    summary={
        'system':'A Smooth TP15/SL20 10% + B1 TP4/SL20 10% + retest/reclaim +2 within 120m + B2 TP15/SL12 5%',
        'start':START,'final':round(eq,2),'return_pct':round((eq/START-1)*100,2),'max_dd_pct':round(maxdd*100,2),
        'active_days':len(dd),'executed_trades':len(td),'wins':int(wins),'losses':int(losses),'win_rate_pct':round(100*wins/len(td),2) if len(td) else 0,
        'A_trades':int((td.engine=='A').sum()) if len(td) else 0,'B1_trades':int((td.engine=='B1').sum()) if len(td) else 0,'B2_trades':int((td.engine=='B2').sum()) if len(td) else 0,
        'sizing_note':'A and B1 lot use beginning-of-day equity to reproduce validated portfolio; B2 uses beginning-of-day equity plus realized B1 P/L. 0.01 lot step, rounded down.'
    }
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

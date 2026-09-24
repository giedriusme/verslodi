import json
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_backtest as bt

OUT=Path('exact_compare_results'); OUT.mkdir(exist_ok=True)

def first_close_break(frame, ah, al):
    for idx,r in frame.iterrows():
        if float(r.close_mid)>ah:return int(idx),'LONG'
        if float(r.close_mid)<al:return int(idx),'SHORT'
    return None,None

def mk(g, signal_idx, direction, family, tp, sl):
    loc=g.index.get_loc(signal_idx)
    if loc+1>=len(g): return None
    e=g.iloc[loc+1]; fut=g.iloc[loc+1:]
    entry=float(e.open_ask if direction=='LONG' else e.open_bid)
    return {'date':str(g.iloc[0].local_date),'year':int(g.iloc[0].year),'family':family,'direction':direction,
            'entry':entry,'tpv':tp,'slv':sl,
            'hi':fut.high_bid.to_numpy(float) if direction=='LONG' else fut.high_ask.to_numpy(float),
            'lo':fut.low_bid.to_numpy(float) if direction=='LONG' else fut.low_ask.to_numpy(float),
            'cl':fut.close_bid.to_numpy(float) if direction=='LONG' else fut.close_ask.to_numpy(float)}

def eval_trade(c):
    e,tp,sl=c['entry'],c['tpv'],c['slv']; hi,lo,cl=c['hi'],c['lo'],c['cl']
    if len(cl)==0:return None
    if c['direction']=='LONG':
        itp=np.flatnonzero(hi>=e+tp); isl=np.flatnonzero(lo<=e-sl); eod=float(cl[-1]-e)
    else:
        itp=np.flatnonzero(lo<=e-tp); isl=np.flatnonzero(hi>=e+sl); eod=float(e-cl[-1])
    a=int(itp[0]) if len(itp) else None; b=int(isl[0]) if len(isl) else None
    if a is None and b is None:return eod,'EOD'
    if a is not None and (b is None or a<b):return float(tp),'TP'
    return -float(sl),'SL'

def stats(cs):
    rows=[]
    for c in sorted(cs,key=lambda x:x['date']):
        x=eval_trade(c)
        if x is not None:rows.append({'date':c['date'],'year':c['year'],'direction':c['direction'],'pnl':x[0],'outcome':x[1]})
    r=pd.DataFrame(rows); p=r.pnl.to_numpy(float)
    pos=int((p>0).sum()); neg=int((p<0).sum()); zero=int((p==0).sum())
    def streak(loss=True):
        best=cur=0; bstart=bend=start=None
        for i,v in enumerate(p):
            ok=(v<0) if loss else (v>0)
            if ok:
                if cur==0:start=i
                cur+=1
                if cur>best:best=cur;bstart=start;bend=i
            else:cur=0
        return best,(r.iloc[bstart].date if bstart is not None else None),(r.iloc[bend].date if bend is not None else None)
    ls,lsd,led=streak(True); ws,wsd,wed=streak(False)
    gains=p[p>0].sum(); losses=-p[p<0].sum(); eq=np.cumsum(p); peak=np.maximum.accumulate(np.r_[0.,eq]); dd=peak[1:]-eq
    years={}
    for y,g in r.groupby('year'):
        py=g.pnl.to_numpy(float); years[str(y)]={'n':len(g),'positive':int((py>0).sum()),'negative':int((py<0).sum()),'net':float(py.sum())}
    return {'n':len(r),'positive':pos,'negative':neg,'zero':zero,'positive_rate':float(pos/len(r)),'negative_rate':float(neg/len(r)),
            'tp_count':int((r.outcome=='TP').sum()),'sl_count':int((r.outcome=='SL').sum()),'eod_count':int((r.outcome=='EOD').sum()),
            'net':float(p.sum()),'expectancy':float(p.mean()),'profit_factor':float(gains/losses) if losses>0 else None,
            'max_drawdown':float(dd.max()) if len(dd) else 0.,'max_loss_streak':ls,'loss_streak_start':lsd,'loss_streak_end':led,
            'max_win_streak':ws,'win_streak_start':wsd,'win_streak_end':wed,'years':years},r

def main():
    df=bt.load_data(); df['mod']=(df.hour*60+df.minute).astype(int)
    close_carry=[]; close_post10=[]; extreme=[]; mom=[]
    for day,g0 in df.groupby('local_date',sort=True):
        if pd.Timestamp(day).dayofweek>=5:continue
        g=g0.sort_values('timestamp').reset_index(drop=True)
        asia=g[(g['mod']>=60)&(g['mod']<540)]
        if len(asia)>=420:
            ah=float(asia.high_mid.max()); al=float(asia.low_mid.min())
            # A: confirmed close from 09:00-10:00 can carry direction into 10:00.
            pre=g[(g['mod']>=540)&(g['mod']<600)]; idx,d=first_close_break(pre,ah,al)
            if d is not None:
                startbar=g[g['mod']==600]
                if not startbar.empty:
                    c=mk(g,int(startbar.index[0]),d,'asia_close_carry_pre10',20.,20.)
                    if c:close_carry.append(c)
            else:
                post=g[(g['mod']>=600)&(g['mod']<1440)]; idx,d=first_close_break(post,ah,al)
                if d is not None:
                    c=mk(g,idx,d,'asia_close_carry_pre10',20.,20.)
                    if c:close_carry.append(c)
            # B: ignore all pre-10 action; only first confirmed 1m close after 10:00 counts.
            post=g[(g['mod']>=600)&(g['mod']<1440)]; idx,d=first_close_break(post,ah,al)
            if d is not None:
                c=mk(g,idx,d,'asia_close_post10_only',20.,20.)
                if c:close_post10.append(c)
            # Asia Extreme Continuation: at 10:00 close, top/bottom 10% of Asia range.
            s=g[g['mod']==600]
            if not s.empty and ah>al:
                r=s.iloc[0]; rel=(float(r.close_mid)-al)/(ah-al); idx=int(s.index[0])
                d='LONG' if rel>=.9 else ('SHORT' if rel<=.1 else None)
                if d:
                    c=mk(g,idx,d,'asia_extreme_continuation',20.,40.)
                    if c:extreme.append(c)
        # 10:00-11:00 move >=8, continuation, enter next bar, TP35/SL30.
        w=g[(g['mod']>=600)&(g['mod']<660)]
        if len(w)>=42:
            mv=float(w.iloc[-1].close_mid-w.iloc[0].open_mid)
            if abs(mv)>=8.:
                d='LONG' if mv>0 else 'SHORT'; idx=int(w.index[-1])
                c=mk(g,idx,d,'10_11_momentum_continuation',35.,30.)
                if c:mom.append(c)
    result={}
    for name,cs in [('asia_close_carry_pre10',close_carry),('asia_close_post10_only',close_post10),('asia_extreme_continuation',extreme),('10_11_momentum_continuation',mom)]:
        s,r=stats(cs); result[name]=s; r.to_csv(OUT/f'{name}_trades.csv',index=False)
    with open(OUT/'summary.json','w') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_backtest as bt

OUT = Path('strategy_search_results')
OUT.mkdir(exist_ok=True)
TPS = [10.0, 15.0, 20.0, 25.0, 30.0]
SLS = [15.0, 20.0, 25.0, 30.0, 35.0]


def make_candidate(g, signal_pos, direction, strategy, param):
    if signal_pos is None or signal_pos + 1 >= len(g):
        return None
    e = g.iloc[signal_pos + 1]
    fut = g.iloc[signal_pos + 1:]
    entry = float(e['open_ask'] if direction == 'LONG' else e['open_bid'])
    return {
        'date': str(g.iloc[0]['local_date']), 'year': int(g.iloc[0]['year']),
        'strategy': strategy, 'param': str(param), 'direction': direction,
        'entry': entry,
        'hi': fut['high_bid'].to_numpy(float) if direction == 'LONG' else fut['high_ask'].to_numpy(float),
        'lo': fut['low_bid'].to_numpy(float) if direction == 'LONG' else fut['low_ask'].to_numpy(float),
        'close': fut['close_bid'].to_numpy(float) if direction == 'LONG' else fut['close_ask'].to_numpy(float),
    }


def first_signal(post, hi_level, lo_level, mode='close_break', buffer=0.0, fade=False):
    for p, (_, r) in enumerate(post.iterrows()):
        if fade:
            up = r['high_mid'] >= hi_level + buffer and r['close_mid'] < hi_level
            dn = r['low_mid'] <= lo_level - buffer and r['close_mid'] > lo_level
            if up and dn: continue
            if up: return p, 'SHORT'
            if dn: return p, 'LONG'
        else:
            up = r['close_mid'] >= hi_level + buffer
            dn = r['close_mid'] <= lo_level - buffer
            if up and dn: continue
            if up: return p, 'LONG'
            if dn: return p, 'SHORT'
    return None, None


def gen_breakout(g, range_start, range_end, search_start, search_end, name, buffer):
    w = g[(g['mod'] >= range_start) & (g['mod'] < range_end)]
    post = g[(g['mod'] >= search_start) & (g['mod'] < search_end)]
    if len(w) < max(20, int((range_end-range_start)*0.75)) or post.empty: return None
    hi, lo = float(w['high_mid'].max()), float(w['low_mid'].min())
    rel, d = first_signal(post, hi, lo, buffer=buffer)
    if d is None: return None
    pos = int(post.index[rel])
    return make_candidate(g, pos, d, name, {'buffer': buffer})


def gen_rejection(g, hi, lo, search_start, search_end, name, ext):
    post = g[(g['mod'] >= search_start) & (g['mod'] < search_end)]
    if post.empty: return None
    rel, d = first_signal(post, hi, lo, buffer=ext, fade=True)
    if d is None: return None
    pos = int(post.index[rel])
    return make_candidate(g, pos, d, name, {'extension': ext})


def gen_momentum(g, start, end, threshold, name, reverse=False):
    w = g[(g['mod'] >= start) & (g['mod'] < end)]
    if len(w) < max(20, int((end-start)*0.75)): return None
    move = float(w.iloc[-1]['close_mid'] - w.iloc[0]['open_mid'])
    if abs(move) < threshold: return None
    d = 'LONG' if move > 0 else 'SHORT'
    if reverse: d = 'SHORT' if d == 'LONG' else 'LONG'
    pos = int(w.index[-1])
    return make_candidate(g, pos, d, name, {'threshold': threshold})


def eval_trade(c, tp, sl):
    e, hi, lo, cl = c['entry'], c['hi'], c['lo'], c['close']
    if len(cl) == 0: return None
    if c['direction'] == 'LONG':
        itp = np.flatnonzero(hi >= e + tp); isl = np.flatnonzero(lo <= e - sl); eod = float(cl[-1] - e)
    else:
        itp = np.flatnonzero(lo <= e - tp); isl = np.flatnonzero(hi >= e + sl); eod = float(e - cl[-1])
    a = int(itp[0]) if len(itp) else None; b = int(isl[0]) if len(isl) else None
    if a is None and b is None: return eod, 'EOD'
    if a is not None and (b is None or a < b): return float(tp), 'TP'
    return -float(sl), 'SL'


def calc_metrics(cands, tp, sl, years=None):
    rows=[]
    for c in cands:
        if years is not None and c['year'] not in years: continue
        x = eval_trade(c,tp,sl)
        if x is not None: rows.append((c['year'], x[0], x[1]))
    if not rows: return {}
    r=pd.DataFrame(rows,columns=['year','pnl','outcome']); p=r.pnl.to_numpy(float)
    gains=p[p>0].sum(); losses=-p[p<0].sum(); eq=np.cumsum(p); peak=np.maximum.accumulate(np.r_[0.,eq]); dd=peak[1:]-eq
    return {'n':len(r),'net':float(p.sum()),'expectancy':float(p.mean()),'pf':float(gains/losses) if losses>0 else 99.0,
            'positive_rate':float((p>0).mean()),'max_dd':float(dd.max()) if len(dd) else 0.0,
            'tp':int((r.outcome=='TP').sum()),'sl':int((r.outcome=='SL').sum()),'eod':int((r.outcome=='EOD').sum())}


def main():
    df=bt.load_data(); df['mod']=(df['hour']*60+df['minute']).astype(int)
    day_frames=[]
    for day,g in df.groupby('local_date',sort=True):
        if pd.Timestamp(day).dayofweek>=5: continue
        day_frames.append((day,g.sort_values('timestamp').reset_index(drop=True)))

    candidates={}
    def add(key,c):
        if c is not None: candidates.setdefault(key,[]).append(c)

    prev=None
    for day,g in day_frames:
        asia=g[(g['mod']>=60)&(g['mod']<540)]
        # Europe 09:00-10:00 range breakout
        for b in [0.,1.,2.]: add(('europe_09_10_orb',f'b={b}'),gen_breakout(g,540,600,600,960,'europe_09_10_orb',b))
        # 10:00-10:30 opening range breakout
        for b in [0.,1.,2.]: add(('open_10_30_orb',f'b={b}'),gen_breakout(g,600,630,630,960,'open_10_30_orb',b))
        # 14:00-15:00 range breakout
        for b in [0.,1.,2.]: add(('afternoon_14_15_orb',f'b={b}'),gen_breakout(g,840,900,900,1200,'afternoon_14_15_orb',b))
        # Asia close-confirmed breakout and rejection
        if len(asia)>=420:
            ah,al=float(asia.high_mid.max()),float(asia.low_mid.min())
            for b in [0.,1.,2.]:
                post=g[(g['mod']>=600)&(g['mod']<960)]
                rel,d=first_signal(post,ah,al,buffer=b)
                if d is not None: add(('asia_close_breakout',f'b={b}'),make_candidate(g,int(post.index[rel]),d,'asia_close_breakout',{'buffer':b}))
            for ext in [0.,2.,5.]: add(('asia_rejection',f'ext={ext}'),gen_rejection(g,ah,al,540,960,'asia_rejection',ext))
        # Previous-day breakout and rejection
        if prev is not None:
            ph,pl=prev
            for b in [0.,2.,5.]:
                post=g[(g['mod']>=600)&(g['mod']<1080)]
                rel,d=first_signal(post,ph,pl,buffer=b)
                if d is not None: add(('prevday_breakout',f'b={b}'),make_candidate(g,int(post.index[rel]),d,'prevday_breakout',{'buffer':b}))
            for ext in [0.,2.,5.]: add(('prevday_rejection',f'ext={ext}'),gen_rejection(g,ph,pl,600,1080,'prevday_rejection',ext))
        # Time-window momentum continuation/reversal
        for th in [3.,5.,8.,12.]:
            add(('morning_momentum_cont',f'th={th}'),gen_momentum(g,540,600,th,'morning_momentum_cont',False))
            add(('morning_momentum_fade',f'th={th}'),gen_momentum(g,540,600,th,'morning_momentum_fade',True))
            add(('midday_momentum_cont',f'th={th}'),gen_momentum(g,780,840,th,'midday_momentum_cont',False))
        prev=(float(g.high_mid.max()),float(g.low_mid.min()))

    rows=[]
    for (fam,param),cands in candidates.items():
        for tp in TPS:
            for sl in SLS:
                tr=calc_metrics(cands,tp,sl,{2023,2024,2025}); oo=calc_metrics(cands,tp,sl,{2026}); aa=calc_metrics(cands,tp,sl,None)
                if not tr: continue
                year_exp=[]; posyrs=0
                for y in [2023,2024,2025]:
                    ym=calc_metrics(cands,tp,sl,{y})
                    if ym:
                        year_exp.append(ym['expectancy']); posyrs += int(ym['net']>0)
                robust=float(tr['expectancy']-0.35*np.std(year_exp)) if year_exp else -999
                row={'family':fam,'param':param,'tp':tp,'sl':sl,'robust_score':robust,'positive_train_years':posyrs}
                for pre,m in [('train',tr),('oos',oo),('all',aa)]:
                    for k,v in m.items(): row[f'{pre}_{k}']=v
                rows.append(row)
    grid=pd.DataFrame(rows)
    grid.to_csv(OUT/'all_grid.csv',index=False)
    selected=[]
    for fam,g in grid.groupby('family'):
        elig=g[(g.train_n>=100)&(g.train_pf>=1.03)&(g.positive_train_years>=2)]
        if elig.empty: elig=g[g.train_n>=80]
        if elig.empty: continue
        best=elig.sort_values(['positive_train_years','robust_score','train_expectancy'],ascending=False).iloc[0]
        selected.append(best)
    sel=pd.DataFrame(selected).sort_values(['all_net','oos_net'],ascending=False)
    sel.to_csv(OUT/'selected_by_family.csv',index=False)
    robust=sel[(sel.all_net>0)&(sel.oos_net>0)&(sel.train_net>0)].copy()
    robust=robust.sort_values(['positive_train_years','all_expectancy','oos_expectancy'],ascending=False)
    robust.to_csv(OUT/'profitable_train_oos.csv',index=False)
    with open(OUT/'summary.json','w') as f:
        json.dump({'families':int(grid.family.nunique()),'parameter_rows':int(len(grid)),
                   'selected':sel.to_dict('records'),'profitable_train_oos':robust.to_dict('records')},f,indent=2,allow_nan=False)
    print('STRATEGY_SEARCH_COMPLETE')
    print(robust.head(12).to_string(index=False))

if __name__=='__main__': main()

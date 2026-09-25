import json, math, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p

OUT=Path('compound_filter_results'); OUT.mkdir(exist_ok=True)
START=300.0
RISK=0.10
LOT_STEP=0.01
USD_PER_DOLLAR_PER_LOT=100.0
TRAIN_YEARS=set(range(2018,2023))
MODERN_TRAIN_YEARS=set(range(2020,2023))
OOS_YEARS=set(range(2023,2027))

ENGINES={
 'A': {'key':('asia_compression_break','r070'),'h':1440,'tps':[4,5,6,8,10,12,15,20],'sls':[8,10,12,15,20]},
 'B': {'key':('asia_break_10_16','thu_fri'),'h':480,'tps':[4,5,6,8,10,12,15,20],'sls':[8,10,12,15,20,25,30]},
 'C': {'key':('prevclose_cont_840','all'),'h':1440,'tps':[10,15,20,25,30],'sls':[10,12,15,20,25,30]},
}

def lot_size(equity, sl):
    if equity<=0:return 0.0
    raw=(equity*RISK)/(sl*USD_PER_DOLLAR_PER_LOT)
    steps=math.floor((raw+1e-12)/LOT_STEP)
    return round(max(0,steps)*LOT_STEP,2)

def simulate(rows, years, start=START):
    x=[r for r in rows if int(r['year']) in years]
    x.sort(key=lambda r:(r['date'],r.get('engine','')))
    eq=float(start); peak=eq; maxdd=0.0; maxdd_amt=0.0; max_day_loss=0.0
    skipped=0; executed=0; wins=0; losses=0; daily=[]; yr={}
    for date, grp_it in itertools.groupby(x,key=lambda r:r['date']):
        grp=list(grp_it); y=int(grp[0]['year']); bod=eq; day_pnl=0.0; day_risk=0.0
        for r in grp:
            lot=lot_size(bod,float(r['sl']))
            if lot<LOT_STEP:
                skipped+=1; continue
            risk_amt=float(r['sl'])*USD_PER_DOLLAR_PER_LOT*lot
            pnl=float(r['pnl'])*USD_PER_DOLLAR_PER_LOT*lot
            day_pnl+=pnl; day_risk+=risk_amt; executed+=1
            if pnl>0:wins+=1
            elif pnl<0:losses+=1
        eq+=day_pnl
        if eq<0:eq=0.0
        peak=max(peak,eq)
        dd_amt=peak-eq; dd=dd_amt/peak if peak>0 else 1.0
        maxdd=max(maxdd,dd); maxdd_amt=max(maxdd_amt,dd_amt)
        if bod>0:max_day_loss=max(max_day_loss,max(0.0,-day_pnl/bod))
        daily.append({'date':date,'year':y,'bod':bod,'pnl':day_pnl,'eod':eq,'risk_frac':day_risk/bod if bod>0 else 0.0})
        yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0:break
    for y,v in yr.items():v['return']=v['end']/v['start']-1 if v['start']>0 else -1
    return {'final':eq,'multiple':eq/start if start>0 else 0,'maxdd':maxdd,'maxdd_amt':maxdd_amt,'max_day_loss':max_day_loss,
            'executed':executed,'skipped_minlot':skipped,'wins':wins,'losses':losses,'wr':wins/executed if executed else 0.0,
            'positive_years':sum(v['return']>0 for v in yr.values()),'years':yr,'daily':daily}

def feature_frame(df, cs):
    daystats={}; weekdays=[]
    for day,g in df.groupby('date',sort=True):
        if pd.Timestamp(day).dayofweek>=5:continue
        asia=g[(g['mod']>=60)&(g['mod']<540)]
        daystats[str(day)]={'H':float(g.high_mid.max()),'L':float(g.low_mid.min()),'C':float(g.iloc[-1].close_mid),
                            'AH':float(asia.high_mid.max()) if len(asia) else np.nan,
                            'AL':float(asia.low_mid.min()) if len(asia) else np.nan}
        weekdays.append(str(day))
    prevmap={}
    for i,d in enumerate(weekdays):
        if i>0:prevmap[d]=daystats[weekdays[i-1]]
    rec=[]
    for j,c in enumerate(cs):
        i=int(c['entry_i']); s=df.iloc[max(0,i-1)]; e=df.iloc[i]
        rng=float(s.high_mid-s.low_mid); r60=float(s.r60) if pd.notna(s.r60) and float(s.r60)>0 else np.nan
        sign=1.0 if c['side']=='LONG' else -1.0
        body=abs(float(s.close_mid-s.open_mid))/rng if rng>0 else 0.0
        close_strength=((float(s.close_mid-s.low_mid))/rng if c['side']=='LONG' else (float(s.high_mid-s.close_mid))/rng) if rng>0 else 0.5
        trend=(1 if sign*(float(s.close_mid-s.ema20))>0 else 0)+(1 if sign*(float(s.ema20-s.ema60))>0 else 0)+(1 if sign*(float(s.ema60-s.ema240))>0 else 0)
        mom=np.nan
        if i>=61 and np.isfinite(r60):mom=sign*(float(s.close_mid)-float(df.iloc[i-61].close_mid))/r60
        ema_dist=sign*(float(s.close_mid-s.ema240))/r60 if np.isfinite(r60) else np.nan
        st=daystats.get(c['date'],{}); prev=prevmap.get(c['date'])
        boundary=st.get('AH',np.nan) if c['side']=='LONG' else st.get('AL',np.nan)
        break_depth=sign*(float(s.close_mid)-boundary)/r60 if np.isfinite(boundary) and np.isfinite(r60) else np.nan
        prev_room=np.nan; prev_dist=np.nan
        if prev and np.isfinite(r60):
            prev_room=((prev['H']-float(s.close_mid))/r60 if c['side']=='LONG' else (float(s.close_mid)-prev['L'])/r60)
            prev_dist=abs(float(s.close_mid)-prev['C'])/r60
        spread=float(e.open_ask-e.open_bid)
        rec.append({'ci':j,'date':c['date'],'year':int(c['year']),'wd':int(c['wd']),'side':c['side'],'entry_time':int(s['mod']),
                    'spread':spread,'spread_rel':spread/r60 if np.isfinite(r60) else np.nan,'body':body,'close_strength':close_strength,
                    'trend':trend,'mom60':mom,'ema_dist':ema_dist,'break_depth':break_depth,'prev_room':prev_room,'prev_dist':prev_dist,
                    'asia_ratio':float(c.get('asia_ratio',np.nan)),'asia_range':float(c.get('asia_range',np.nan))})
    return pd.DataFrame(rec)

def filter_defs(F, tp):
    def m(expr):return np.asarray(expr.fillna(False) if hasattr(expr,'fillna') else expr,dtype=bool)
    defs=[('ALL',np.ones(len(F),dtype=bool))]
    for v in [0.5,0.75,1.0,1.25]:defs.append((f'spread<={v}',m(F.spread<=v)))
    for v in [0.02,0.04,0.06]:defs.append((f'spreadRel<={v}',m(F.spread_rel<=v)))
    defs += [
      ('body>=0.40',m(F.body>=.40)),('body>=0.60',m(F.body>=.60)),
      ('close>=0.70',m(F.close_strength>=.70)),('close>=0.85',m(F.close_strength>=.85)),
      ('trend>=2',m(F.trend>=2)),('trend=3',m(F.trend>=3)),
      ('mom>=0',m(F.mom60>=0)),('mom>=0.15',m(F.mom60>=.15)),
      ('emaAligned',m(F.ema_dist>=0)),('prevRoom>=0',m(F.prev_room>=0)),
      ('prevDist>=0.25',m(F.prev_dist>=.25)),('prevDist>=0.50',m(F.prev_dist>=.50)),
      ('break>=0.05',m(F.break_depth>=.05)),('break>=0.10',m(F.break_depth>=.10)),
      ('break<=0.25',m(F.break_depth<=.25)),('asiaRatio<=0.55',m(F.asia_ratio<=.55)),
      ('asiaRatio<=0.80',m(F.asia_ratio<=.80)),('asiaRatio>=0.80',m(F.asia_ratio>=.80)),
      ('time<=12',m(F.entry_time<=720)),('time<=13',m(F.entry_time<=780)),('time>=13',m(F.entry_time>=780)),
      ('LONG',m(F.side=='LONG')),('SHORT',m(F.side=='SHORT')),
      ('Mon-Wed',m(F.wd<=2)),('Thu-Fri',m(F.wd>=3)),
      ('spread<=10%TP',m(F.spread<=0.10*tp)),('spread<=15%TP',m(F.spread<=0.15*tp)),
    ]
    base={n:a for n,a in defs}
    combos=[
      ('lowSpread+close',base['spread<=1.0'] & base['close>=0.70']),
      ('body+close',base['body>=0.40'] & base['close>=0.70']),
      ('trend+mom',base['trend>=2'] & base['mom>=0']),
      ('trend+close',base['trend>=2'] & base['close>=0.70']),
      ('trend+lowSpread',base['trend>=2'] & base['spread<=1.0']),
      ('break+trend',base['break>=0.05'] & base['trend>=2']),
      ('break+close',base['break>=0.05'] & base['close>=0.70']),
      ('prevRoom+trend',base['prevRoom>=0'] & base['trend>=2']),
      ('prevDist+trend',base['prevDist>=0.25'] & base['trend>=2']),
      ('early+trend',base['time<=13'] & base['trend>=2']),
      ('lowSpread+body',base['spread<=1.0'] & base['body>=0.40']),
      ('compression+trend',base['asiaRatio<=0.80'] & base['trend>=2']),
      ('TPspread+close',base['spread<=15%TP'] & base['close>=0.70']),
    ]
    return defs+combos

def score_sim(s):
    if s['final']<=START or s['executed']<20:return -1e9
    return math.log(s['final']/START)-1.25*s['maxdd']-0.35*s['max_day_loss']+0.08*s['positive_years']

def build_engine_rows(df, cs, F, engine, tp, sl, h):
    rows=[]
    for j,c in enumerate(cs):
        z=p.trade_result(df,c,float(tp),float(sl),h)
        if z:
            z=dict(z); z['engine']=engine; z['sl']=float(sl); z['tp']=float(tp); z['ci']=j; rows.append(z)
    return rows

def apply_mask(rows, mask):
    allowed=set(np.flatnonzero(mask).tolist())
    return [r for r in rows if int(r['ci']) in allowed]

def main():
    df=p.load(); C=p.candidates(df)
    cache={}; candidate_records=[]; shortlist={e:[] for e in ENGINES}
    feature_cache={}
    for eng,spec in ENGINES.items():
        cs=C.get(spec['key'],[]); F=feature_frame(df,cs); feature_cache[eng]=F
        original_train=max(1,int(F.year.isin(TRAIN_YEARS).sum()))
        min_train=max(25,int(original_train*.35))
        print(eng,'signals',len(cs),'train',original_train,'min filtered',min_train,flush=True)
        for tp in spec['tps']:
            for sl in spec['sls']:
                key0=(eng,float(tp),float(sl)); rows=build_engine_rows(df,cs,F,eng,tp,sl,spec['h']); cache[key0]=rows
                for fname,mask in filter_defs(F,float(tp)):
                    rr=apply_mask(rows,mask)
                    tr=[r for r in rr if int(r['year']) in TRAIN_YEARS]
                    if len(tr)<min_train:continue
                    sim=simulate(rr,TRAIN_YEARS); mod=simulate(rr,MODERN_TRAIN_YEARS); sc=score_sim(sim)
                    rec={'engine':eng,'tp':tp,'sl':sl,'filter':fname,'train_n':sim['executed'],'train_final':sim['final'],'train_multiple':sim['multiple'],
                         'train_dd':sim['maxdd'],'train_wr':sim['wr'],'train_pos_years':sim['positive_years'],'modern_final':mod['final'],'modern_dd':mod['maxdd'],
                         'train_score':sc,'retain_train':len(tr)/original_train}
                    candidate_records.append(rec)
        cg=pd.DataFrame([r for r in candidate_records if r['engine']==eng]).sort_values('train_score',ascending=False)
        # Diversify selections by (tp,sl,filter), selected only from pre-2023 data.
        picks=[]; seen=set()
        for _,r in cg.iterrows():
            sig=(r.tp,r.sl,r['filter'])
            if sig in seen:continue
            if r.train_final<=START:continue
            picks.append(r.to_dict());seen.add(sig)
            if len(picks)>=12:break
        # also add modern-regime leaders (2020-22 selection, OOS still untouched)
        for _,r in cg.sort_values(['modern_final','train_score'],ascending=False).iterrows():
            sig=(r.tp,r.sl,r['filter'])
            if sig in seen:continue
            if r.modern_final<=START:continue
            d=r.to_dict();d['selection']='modern2020_22';picks.append(d);seen.add(sig)
            if len(picks)>=18:break
        for d in picks:d.setdefault('selection','robust2018_22')
        shortlist[eng]=picks
    grid=pd.DataFrame(candidate_records).sort_values(['engine','train_score'],ascending=[True,False]); grid.to_csv(OUT/'train_grid.csv',index=False)

    # OOS evaluate only pre-2023 selected candidates.
    out=[]; filtered_cache={}
    for eng,picks in shortlist.items():
        F=feature_cache[eng]
        for d in picks:
            tp=float(d['tp']);sl=float(d['sl']);fname=d['filter'];rows=cache[(eng,tp,sl)]
            masks={n:a for n,a in filter_defs(F,tp)}; rr=apply_mask(rows,masks[fname]); filtered_cache[(eng,tp,sl,fname)]=rr
            oo=simulate(rr,OOS_YEARS); full=simulate(rr,set(range(2018,2027)))
            z=dict(d);z.update({'oos_final':oo['final'],'oos_multiple':oo['multiple'],'oos_dd':oo['maxdd'],'oos_wr':oo['wr'],'oos_trades':oo['executed'],
                                'oos_pos_years':oo['positive_years'],'full_final':full['final'],'full_dd':full['maxdd']})
            for y,v in oo['years'].items():z[f'oos_ret_{y}']=v['return']
            out.append(z)
    oos=pd.DataFrame(out).sort_values(['engine','oos_final'],ascending=[True,False]); oos.to_csv(OUT/'selected_oos.csv',index=False)

    # A6/12 explicit filter audit.
    a6=oos[(oos.engine=='A')&(oos.tp==6)&(oos.sl==12)].copy()
    if a6.empty:
        F=feature_cache['A']; rows=cache[('A',6.0,12.0)]; temp=[]
        for fname,mask in filter_defs(F,6.0):
            rr=apply_mask(rows,mask); tr=simulate(rr,TRAIN_YEARS); oo=simulate(rr,OOS_YEARS)
            if tr['executed']>=40: temp.append({'filter':fname,'train_final':tr['final'],'train_dd':tr['maxdd'],'oos_final':oo['final'],'oos_dd':oo['maxdd'],'oos_wr':oo['wr'],'oos_trades':oo['executed'],'oos_pos_years':oo['positive_years']})
        a6=pd.DataFrame(temp).sort_values('oos_final',ascending=False)
    a6.to_csv(OUT/'a6_12_filter_audit.csv',index=False)

    # Portfolio search uses only candidates selected from TRAIN. Search on TRAIN, then validate OOS.
    opts={}
    for eng,picks in shortlist.items():opts[eng]=picks[:10]
    port_train=[]
    choices={e:[None]+opts[e] for e in ENGINES}
    for a,b,c in itertools.product(choices['A'],choices['B'],choices['C']):
        chosen=[x for x in [a,b,c] if x is not None]
        if len(chosen)<2:continue
        rows=[]; desc=[]
        for d in chosen:
            eng=d['engine'];tp=float(d['tp']);sl=float(d['sl']);fname=d['filter'];F=feature_cache[eng]
            k=(eng,tp,sl,fname)
            if k not in filtered_cache:
                masks={n:x for n,x in filter_defs(F,tp)};filtered_cache[k]=apply_mask(cache[(eng,tp,sl)],masks[fname])
            rows+=filtered_cache[k];desc.append(f"{eng}:{tp:g}/{sl:g}|{fname}")
        s=simulate(rows,TRAIN_YEARS); sc=score_sim(s)
        port_train.append({'portfolio':' + '.join(desc),'components':len(chosen),'train_score':sc,'train_final':s['final'],'train_dd':s['maxdd'],'train_wr':s['wr'],'train_trades':s['executed'],'train_pos_years':s['positive_years']})
    pt=pd.DataFrame(port_train).sort_values('train_score',ascending=False).head(30)
    ports=[]
    # map descriptor back to rows
    for _,r in pt.iterrows():
        rows=[]
        for part in r.portfolio.split(' + '):
            eng,rest=part.split(':',1); ex,fn=rest.split('|',1); tp,sl=map(float,ex.split('/'))
            rows+=filtered_cache[(eng,tp,sl,fn)]
        oo=simulate(rows,OOS_YEARS); full=simulate(rows,set(range(2018,2027)))
        z=r.to_dict();z.update({'oos_final':oo['final'],'oos_multiple':oo['multiple'],'oos_dd':oo['maxdd'],'oos_wr':oo['wr'],'oos_trades':oo['executed'],
                               'oos_pos_years':oo['positive_years'],'oos_max_day_loss':oo['max_day_loss'],'full_final':full['final'],'full_dd':full['maxdd']})
        for y,v in oo['years'].items():z[f'oos_ret_{y}']=v['return']
        ports.append(z)
    ports=pd.DataFrame(ports).sort_values('oos_final',ascending=False);ports.to_csv(OUT/'portfolio_oos.csv',index=False)

    # Baselines requested/current.
    baselines=[]
    base_specs=[('ExpectancyV2',[('A',15,12,'ALL'),('B',20,20,'ALL'),('C',30,20,'ALL')]),('HighWR_A6+B5+C30',[('A',6,12,'ALL'),('B',5,20,'ALL'),('C',30,20,'ALL')]),('A6_only',[('A',6,12,'ALL')])]
    for name,parts in base_specs:
        rows=[]
        for eng,tp,sl,fn in parts:
            F=feature_cache[eng];masks={n:x for n,x in filter_defs(F,float(tp))};rows+=apply_mask(cache[(eng,float(tp),float(sl))],masks[fn])
        tr=simulate(rows,TRAIN_YEARS);oo=simulate(rows,OOS_YEARS);full=simulate(rows,set(range(2018,2027)))
        q={'name':name,'train_final':tr['final'],'train_dd':tr['maxdd'],'oos_final':oo['final'],'oos_dd':oo['maxdd'],'oos_wr':oo['wr'],'oos_trades':oo['executed'],'oos_pos_years':oo['positive_years'],'oos_max_day_loss':oo['max_day_loss'],'full_final':full['final'],'full_dd':full['maxdd']}
        for y,v in oo['years'].items():q[f'oos_ret_{y}']=v['return']
        baselines.append(q)
    pd.DataFrame(baselines).to_csv(OUT/'baselines.csv',index=False)

    summary={'assumptions':{'start_balance':START,'risk_per_trade':RISK,'lot_step':LOT_STEP,'usd_per_$move_per_1lot':USD_PER_DOLLAR_PER_LOT,'daily_sizing':'all trades opened on a date use beginning-of-day equity'},
             'best_selected_by_engine':{},'best_portfolio_oos':ports.head(8).to_dict('records'),'baselines':baselines}
    for eng in ENGINES:
        summary['best_selected_by_engine'][eng]=oos[oos.engine==eng].sort_values('oos_final',ascending=False).head(8).to_dict('records')
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':main()

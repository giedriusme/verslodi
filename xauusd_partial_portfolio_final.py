import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_compound_filter_search as s
import xauusd_b_partial_exit_search as b

OUT=Path('partial_portfolio_final_results');OUT.mkdir(exist_ok=True)
START=300.; VALUE=100.; TRAIN=set(range(2018,2023)); OOS=set(range(2023,2027)); FULL=set(range(2018,2027))

def lot(eq,sl):return s.lot_size(eq,sl)

def build_a(df,C,tp,sl,filter_name):
    cs=C[('asia_compression_break','r070')];F=s.feature_frame(df,cs);masks={n:m for n,m in s.filter_defs(F,tp)}
    rows=s.build_engine_rows(df,cs,F,'A',tp,sl,1440);return s.apply_mask(rows,masks[filter_name])

def build_b(df,C):
    cs0=C[('asia_break_10_16','thu_fri')];F=s.feature_frame(df,cs0);masks={n:m for n,m in s.filter_defs(F,20.)}
    allowed=set(np.flatnonzero(masks['mom>=0.15']).tolist());cs=[c for j,c in enumerate(cs0) if j in allowed];cs=sorted(cs,key=lambda x:x['date'])
    bases=[b.base_path(df,c) for c in cs];return cs,bases

def simulate(df,Arows,Bcs,Bbases,Bspec,years):
    amap={r['date']:r for r in Arows if int(r['year']) in years}; bidx={c['date']:j for j,c in enumerate(Bcs) if int(c['year']) in years}
    dates=sorted(set(amap)|set(bidx));eq=START;peak=eq;maxdd=0.;wins=losses=exe=0;yr={};maxrisk=0.;maxdayloss=0.
    for d in dates:
        bod=eq;day=0.;riskamt=0.;y=int(str(d)[:4])
        if d in amap:
            r=amap[d];la=lot(bod,float(r['sl']))
            if la>=.01:
                pa=float(r['pnl'])*VALUE*la;day+=pa;riskamt+=float(r['sl'])*VALUE*la;exe+=1;wins+=pa>0;losses+=pa<0
        if d in bidx:
            j=bidx[d];c=Bcs[j]
            if Bspec['kind']=='full':
                sl=Bspec['sl'];lb=lot(bod,sl)
                if lb>=.01:
                    pts=b.full_outcome(Bbases[j],Bspec['tp'],sl);pb=pts*VALUE*lb;day+=pb;riskamt+=sl*VALUE*lb;exe+=1;wins+=pb>0;losses+=pb<0
            else:
                tp1,tp2,sl,frac,mode,fb=Bspec['scheme'];lb=lot(bod,sl)
                if lb>=.01:
                    sp=b.split_lots(lb,frac)
                    if sp is None:
                        tp=tp1 if fb=='TP1' else tp2;pts=b.full_outcome(Bbases[j],tp,sl);pb=pts*VALUE*lb
                    else:
                        l1,l2=sp;p1,pr,state=b.partial_outcome(Bbases[j],tp1,tp2,sl,mode)
                        pb=(p1*VALUE*lb) if state in ('SL','TIME_PRE_TP1') else p1*VALUE*l1+pr*VALUE*l2
                    day+=pb;riskamt+=sl*VALUE*lb;exe+=1;wins+=pb>0;losses+=pb<0
        eq=max(0.,eq+day);peak=max(peak,eq);maxdd=max(maxdd,(peak-eq)/peak if peak else 1.)
        maxrisk=max(maxrisk,riskamt/bod if bod else 0.);maxdayloss=max(maxdayloss,max(0.,-day/bod if bod else 0.))
        yr.setdefault(y,{'start':bod,'end':eq})['end']=eq
        if eq<=0:break
    for y,v in yr.items():v['return']=v['end']/v['start']-1 if v['start'] else -1
    return {'final':eq,'multiple':eq/START,'dd':maxdd,'wr':wins/exe if exe else 0.,'trades':exe,'pos_years':sum(v['return']>0 for v in yr.values()),'maxrisk':maxrisk,'maxdayloss':maxdayloss,'years':yr}

def main():
    df=p.load();C=p.candidates(df);Bcs,Bbases=build_b(df,C)
    Aopts={'NONE':[], 'Astrong20_12':build_a(df,C,20.,12.,'close>=0.85'), 'Asmooth15_20':build_a(df,C,15.,20.,'close>=0.85')}
    Bopts={
      'Bfull20_25':{'kind':'full','tp':20.,'sl':25.},
      'Bpartial_trainwinner':{'kind':'partial','scheme':(5.,30.,20.,.7,'ORIG','TP2')},
      'Bpartial_smooth':{'kind':'partial','scheme':(4.,25.,25.,.7,'ORIG','TP2')},
      'Bpartial_balanced':{'kind':'partial','scheme':(4.,25.,20.,.5,'ORIG','TP2')},
    }
    out=[]
    for an,arows in Aopts.items():
      for bn,bs in Bopts.items():
        tr=simulate(df,arows,Bcs,Bbases,bs,TRAIN);oo=simulate(df,arows,Bcs,Bbases,bs,OOS);fu=simulate(df,arows,Bcs,Bbases,bs,FULL)
        z={'portfolio':an+' + '+bn,'train_final':tr['final'],'train_dd':tr['dd'],'train_wr':tr['wr'],'train_pos_years':tr['pos_years'],
           'oos_final':oo['final'],'oos_multiple':oo['multiple'],'oos_dd':oo['dd'],'oos_wr':oo['wr'],'oos_trades':oo['trades'],'oos_pos_years':oo['pos_years'],
           'oos_maxrisk':oo['maxrisk'],'oos_maxdayloss':oo['maxdayloss'],'full_final':fu['final'],'full_dd':fu['dd']}
        for y,v in oo['years'].items():z[f'ret_{y}']=v['return']
        out.append(z)
    q=pd.DataFrame(out);q['score']=q.oos_multiple/(1+3*q.oos_dd);q.sort_values('score',ascending=False).to_csv(OUT/'results.csv',index=False)
    with open(OUT/'summary.json','w') as f:json.dump(q.sort_values('score',ascending=False).to_dict('records'),f,indent=2,allow_nan=False)
    print(q.sort_values('score',ascending=False).to_string(index=False))
if __name__=='__main__':main()

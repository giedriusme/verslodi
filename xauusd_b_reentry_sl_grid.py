import json, math
from pathlib import Path
import pandas as pd
import xauusd_portfolio_discovery as p
import xauusd_b_reentry_search as r

OUT=Path('b_reentry_sl_grid_results');OUT.mkdir(exist_ok=True)
SL2S=[6.,8.,10.,12.,15.,20.]
TP2S=[4.,6.,8.,10.,12.,15.,20.,25.,30.]

def score(z):
    return math.log(max(z['final'],1e-9)/r.START)-1.35*z['dd']+.10*z['pos_years']

def main():
    df=p.load();C=p.candidates(df);Bcs=r.build_b(df,C);A=r.build_a(df,C)
    paths=[r.make_path(df,c) for c in Bcs];events=r.prep([q for q in paths if q is not None])
    rows=[]
    for sl2 in SL2S:
      r.SL2=sl2
      for tp2 in TP2S:
        bt=r.sim_b(events,tp2,r.TRAIN);bo=r.sim_b(events,tp2,r.OOS);bf=r.sim_b(events,tp2,r.FULL)
        at=r.sim_portfolio(A,events,tp2,r.TRAIN);ao=r.sim_portfolio(A,events,tp2,r.OOS);af=r.sim_portfolio(A,events,tp2,r.FULL)
        z={'tp2':tp2,'sl2':sl2,
           'b_train_final':bt['final'],'b_train_dd':bt['dd'],'b_train_wr':bt['wr'],'b_train_pos_years':bt['pos_years'],'b_train_score':score(bt),
           'b_oos_final':bo['final'],'b_oos_dd':bo['dd'],'b_oos_wr':bo['wr'],'b_oos_trades':bo['trades'],'b_oos_reentries':bo['reentries'],'b_oos_pos_years':bo['pos_years'],
           'ab_train_final':at['final'],'ab_train_dd':at['dd'],'ab_train_wr':at['wr'],'ab_train_pos_years':at['pos_years'],'ab_train_score':score(at),
           'ab_oos_final':ao['final'],'ab_oos_dd':ao['dd'],'ab_oos_wr':ao['wr'],'ab_oos_trades':ao['trades'],'ab_oos_reentries':ao['reentries'],'ab_oos_pos_years':ao['pos_years'],
           'ab_oos_maxrisk':ao['maxrisk'],'ab_oos_maxdayloss':ao['maxdayloss'],'b_full_final':bf['final'],'ab_full_final':af['final']}
        for y,v in bo['years'].items():z[f'b_ret_{y}']=v['return']
        for y,v in ao['years'].items():z[f'ab_ret_{y}']=v['return']
        rows.append(z)
    q=pd.DataFrame(rows)
    q.sort_values('b_train_score',ascending=False).to_csv(OUT/'by_b_train.csv',index=False)
    q.sort_values('ab_train_score',ascending=False).to_csv(OUT/'by_ab_train.csv',index=False)
    summary={'best_b_train':q.sort_values('b_train_score',ascending=False).head(10).to_dict('records'),
             'best_ab_train':q.sort_values('ab_train_score',ascending=False).head(10).to_dict('records')}
    with open(OUT/'summary.json','w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))
if __name__=='__main__':main()

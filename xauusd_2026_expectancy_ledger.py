from pathlib import Path
import pandas as pd
import xauusd_portfolio_discovery as p

OUT=Path('expectancy_v2_2026_ledger'); OUT.mkdir(exist_ok=True)

# Only enough history to seed the 20-session Asia median and previous-day context.
p.MONTHS=[(2025,m) for m in (11,12)] + [(2026,m) for m in range(1,9)]

SPECS=[
    ('A 15/12', ('asia_compression_break','r070'), 15.0, 12.0, 1440),
    ('B 20/20', ('asia_break_10_16','thu_fri'), 20.0, 20.0, 480),
    ('C 30/20', ('prevclose_cont_840','all'), 30.0, 20.0, 1440),
    ('A 6/12', ('asia_compression_break','r070'), 6.0, 12.0, 1440),
]

def fmt(x):
    if abs(x-round(x)) < 1e-9:
        return f'{int(round(x)):+d}'
    return f'{x:+.2f}'.rstrip('0').rstrip('.')

def main():
    df=p.load(); C=p.candidates(df)
    ledgers={}
    summaries=[]
    for name,key,tp,sl,h in SPECS:
        rows=[]
        for c in C.get(key,[]):
            if int(c['year']) != 2026: continue
            z=p.trade_result(df,c,tp,sl,h)
            if z:
                rows.append(z)
        rows=sorted(rows,key=lambda r:r['date'])
        ledgers[name]=rows
        pnl=[r['pnl'] for r in rows]
        summaries.append({
            'strategy':name,'trades':len(rows),
            'positive':sum(v>0 for v in pnl),'negative':sum(v<0 for v in pnl),
            'tp':sum(r['outcome']=='TP' for r in rows),'sl':sum(r['outcome']=='SL' for r in rows),
            'time':sum(r['outcome']=='TIME' for r in rows),'net_points':sum(pnl)
        })
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)

    dates=sorted(set(r['date'] for n in ['A 15/12','B 20/20','C 30/20'] for r in ledgers[n]))
    maps={n:{r['date']:fmt(r['pnl']) for r in ledgers[n]} for n in ledgers}
    combined=[]
    for d in dates:
        combined.append({'date':pd.Timestamp(d).strftime('%m.%d'),
                         'A 15/12':maps['A 15/12'].get(d,''),
                         'B 20/20':maps['B 20/20'].get(d,''),
                         'C 30/20':maps['C 30/20'].get(d,'')})
    pd.DataFrame(combined).to_csv(OUT/'expectancy_v2_calendar.csv',index=False)

    a6=[]
    for r in ledgers['A 6/12']:
        a6.append({'date':pd.Timestamp(r['date']).strftime('%m.%d'),'result':fmt(r['pnl']),'outcome':r['outcome']})
    pd.DataFrame(a6).to_csv(OUT/'a_6_12_calendar.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__': main()

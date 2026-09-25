import json, math
from pathlib import Path
import numpy as np
import pandas as pd

import xauusd_portfolio_discovery as p
import xauusd_partial_portfolio_final as pf
import xauusd_b_reentry_search as re
import xauusd_b_reentry_reclaim as rr

OUT = Path('annual_reset_results')
OUT.mkdir(exist_ok=True)
START = 300.0
YEARS = list(range(2018, 2027))

# Minimum equity needed to open 0.01 lot without exceeding the configured risk.
MIN_EQ = {
    'A Smooth + B Partial Balanced': 200.0,   # SL20 @ 10%
    'A Smooth + B Reclaim': 200.0,            # A/B1 SL20 @ 10%
    'B Partial Balanced': 200.0,               # SL20 @ 10%
    'B20/25 + Momentum': 250.0,                # SL25 @ 10%
    'B5/20 + Momentum': 200.0,                 # SL20 @ 10%
}


def make_reclaim_events(df, C):
    # Same Thu/Fri Asia breakout + mom>=0.15 signal used in the reclaim research.
    bcs = re.build_b(df, C)
    events = []
    for c in bcs:
        q = rr.make_path(df, c)
        if q is None:
            continue
        z = rr.first_trade(q)  # B1 TP4 / SL20, conservative same-M1 ordering
        z['path'] = q
        events.append(z)
    return events


def norm_partial(name, year, sim):
    final = float(sim['final'])
    return {
        'system': name,
        'year': int(year),
        'start': START,
        'final': final,
        'return_pct': (final / START - 1.0) * 100.0,
        'max_dd_pct': float(sim['dd']) * 100.0,
        'win_rate_pct': float(sim['wr']) * 100.0,
        'trades': int(sim['trades']),
        'max_day_loss_pct': float(sim.get('maxdayloss', 0.0)) * 100.0,
        'max_nominal_risk_pct': float(sim.get('maxrisk', 0.0)) * 100.0,
        'end_tradable': bool(final + 1e-9 >= MIN_EQ[name]),
        'min_equity_for_001': MIN_EQ[name],
    }


def norm_reclaim(name, year, sim):
    final = float(sim['final'])
    return {
        'system': name,
        'year': int(year),
        'start': START,
        'final': final,
        'return_pct': (final / START - 1.0) * 100.0,
        'max_dd_pct': float(sim['dd']) * 100.0,
        'win_rate_pct': float(sim['wr']) * 100.0,
        'trades': int(sim['trades']),
        'max_day_loss_pct': float(sim.get('maxdayloss', 0.0)) * 100.0,
        'max_nominal_risk_pct': float(sim.get('maxrisk', 0.0)) * 100.0,
        'reclaims': int(sim.get('reclaims', 0)),
        'end_tradable': bool(final + 1e-9 >= MIN_EQ[name]),
        'min_equity_for_001': MIN_EQ[name],
    }


def summarize(q):
    out = []
    for name, g in q.groupby('system', sort=False):
        mult = np.maximum(g['final'].to_numpy(float) / START, 1e-12)
        modern = g[g.year >= 2023]
        rec = {
            'system': name,
            'positive_years_2018_2026': int((g.final > START).sum()),
            'negative_years_2018_2026': int((g.final < START).sum()),
            'positive_years_2023_2026': int((modern.final > START).sum()),
            'stall_years': int((~g.end_tradable).sum()),
            'median_final': float(g.final.median()),
            'mean_final': float(g.final.mean()),
            'worst_final': float(g.final.min()),
            'best_final': float(g.final.max()),
            'geo_mean_multiple': float(np.exp(np.mean(np.log(mult)))),
            'median_dd_pct': float(g.max_dd_pct.median()),
            'worst_dd_pct': float(g.max_dd_pct.max()),
            'mean_wr_pct': float(g.win_rate_pct.mean()),
            'mean_trades': float(g.trades.mean()),
            'final_2026': float(g.loc[g.year == 2026, 'final'].iloc[0]),
            'return_2026_pct': float(g.loc[g.year == 2026, 'return_pct'].iloc[0]),
            'dd_2026_pct': float(g.loc[g.year == 2026, 'max_dd_pct'].iloc[0]),
        }
        # Robustness-oriented rank; not used to optimize parameters, only to order frozen finalists.
        rec['robust_score'] = (
            1.0 * rec['positive_years_2018_2026']
            + 0.5 * rec['positive_years_2023_2026']
            - 1.5 * rec['stall_years']
            + math.log(max(rec['geo_mean_multiple'], 1e-12))
            - 0.01 * rec['worst_dd_pct']
        )
        out.append(rec)
    return pd.DataFrame(out).sort_values(
        ['robust_score', 'positive_years_2018_2026', 'geo_mean_multiple'],
        ascending=[False, False, False]
    )


def main():
    print('Loading XAUUSD 2018-2026 data...', flush=True)
    df = p.load()
    C = p.candidates(df)

    # Shared frozen signals.
    Bcs, Bbases = pf.build_b(df, C)  # Thu/Fri + mom>=0.15
    Asmooth = pf.build_a(df, C, 15.0, 20.0, 'close>=0.85')
    reclaim_events = make_reclaim_events(df, C)

    B_PARTIAL = {'kind': 'partial', 'scheme': (4.0, 25.0, 20.0, 0.5, 'ORIG', 'TP2')}
    B_20_25 = {'kind': 'full', 'tp': 20.0, 'sl': 25.0}
    B_5_20 = {'kind': 'full', 'tp': 5.0, 'sl': 20.0}

    rows = []
    for year in YEARS:
        ys = {year}
        print('Year', year, flush=True)

        # 1) A Smooth + B Partial Balanced
        z = pf.simulate(df, Asmooth, Bcs, Bbases, B_PARTIAL, ys)
        rows.append(norm_partial('A Smooth + B Partial Balanced', year, z))

        # 2) A Smooth + B reclaim: B1 TP4/SL20 10%, retest, reclaim +2 <=120m,
        # B2 TP15/SL12 at 5% risk. Same sizing convention as validated research.
        z = rr.simulate_ab(Asmooth, reclaim_events, 2.0, 15.0, 12.0, 120, ys)
        rows.append(norm_reclaim('A Smooth + B Reclaim', year, z))

        # 3) B Partial Balanced alone
        z = pf.simulate(df, [], Bcs, Bbases, B_PARTIAL, ys)
        rows.append(norm_partial('B Partial Balanced', year, z))

        # 4) B20/25 + Momentum alone
        z = pf.simulate(df, [], Bcs, Bbases, B_20_25, ys)
        rows.append(norm_partial('B20/25 + Momentum', year, z))

        # 5) High-WR B5/20 + Momentum alone
        z = pf.simulate(df, [], Bcs, Bbases, B_5_20, ys)
        rows.append(norm_partial('B5/20 + Momentum', year, z))

    q = pd.DataFrame(rows)
    q = q.sort_values(['year', 'system'])
    q.to_csv(OUT / 'annual_resets.csv', index=False)

    pivot_final = q.pivot(index='year', columns='system', values='final')
    pivot_final.to_csv(OUT / 'final_balance_by_year.csv')
    pivot_ret = q.pivot(index='year', columns='system', values='return_pct')
    pivot_ret.to_csv(OUT / 'return_pct_by_year.csv')
    pivot_dd = q.pivot(index='year', columns='system', values='max_dd_pct')
    pivot_dd.to_csv(OUT / 'max_dd_pct_by_year.csv')

    rank = summarize(q)
    rank.to_csv(OUT / 'ranking.csv', index=False)

    summary = {
        'method': 'Each calendar year starts independently from EUR 300. Position size is rounded down to 0.01 lot. If minimum lot would exceed configured risk, trade is skipped; no artificial restart.',
        'data_note': '2026 is partial through available August 20 data.',
        'systems': {
            'A Smooth + B Partial Balanced': 'A TP15/SL20 close_strength>=0.85, 10% risk; B Thu/Fri mom>=0.15, SL20, 50% TP4 + 50% TP25 when splittable; 0.01 fallback uses TP25.',
            'A Smooth + B Reclaim': 'A as above; B1 TP4/SL20 10%; after TP4 retest original entry, reclaim +2 within 120m; B2 TP15/SL12 at 5% risk.',
            'B Partial Balanced': 'B partial only, same rules/fallback.',
            'B20/25 + Momentum': 'B Thu/Fri mom>=0.15, TP20/SL25, 10% risk.',
            'B5/20 + Momentum': 'B Thu/Fri mom>=0.15, TP5/SL20, 10% risk.',
        },
        'ranking': rank.to_dict('records'),
        'annual': q.to_dict('records'),
    }
    with open(OUT / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print('\nRANKING')
    print(rank.to_string(index=False))
    print('\nFINAL BALANCE BY YEAR')
    print(pivot_final.round(2).to_string())


if __name__ == '__main__':
    main()

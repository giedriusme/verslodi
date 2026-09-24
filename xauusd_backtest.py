import io
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

SOURCE_REPO = "kevingtlin/Market-Data-Lab"
RAW_BASE = f"https://raw.githubusercontent.com/{SOURCE_REPO}/main"
OUT = Path("backtest_results")
OUT.mkdir(exist_ok=True)

YEARS_MONTHS = [(y, m) for y in range(2023, 2027) for m in range(1, 13) if (y < 2026 or m <= 8)]
START_HOURS = [10, 14]
TPS = [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 12.0, 15.0]
SLS = [8.0, 9.0, 10.0, 11.0, 12.0, 12.5, 13.0, 15.0, 17.5, 20.0, 25.0]
BASE_TP, BASE_SL = 5.0, 15.0
TZ = ZoneInfo("Europe/Vilnius")

session = requests.Session()
session.headers.update({"User-Agent": "xauusd-backtest-research/1.0"})


def read_month(side, y, m):
    url = f"{RAW_BASE}/xauusd/{side}/m1/xauusd_{side}_m1_{y:04d}_{m:02d}.csv"
    r = session.get(url, timeout=90)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    expected = ["timestamp", "open", "high", "low", "close"]
    if list(d.columns) != expected:
        raise RuntimeError(f"Unexpected schema {url}: {list(d.columns)}")
    d = d.rename(columns={c: f"{c}_{side}" for c in expected if c != "timestamp"})
    return d


def load_data():
    frames = []
    for i, (y, m) in enumerate(YEARS_MONTHS, 1):
        print(f"Downloading {y}-{m:02d} ({i}/{len(YEARS_MONTHS)})", flush=True)
        bid = read_month("bid", y, m)
        ask = read_month("ask", y, m)
        d = bid.merge(ask, on="timestamp", how="inner", validate="one_to_one")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[f"{c}_mid"] = (df[f"{c}_bid"] + df[f"{c}_ask"]) / 2.0
    df["spread_open"] = df["open_ask"] - df["open_bid"]
    df["spread_close"] = df["close_ask"] - df["close_bid"]
    dt_utc = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["dt_utc"] = dt_utc
    df["dt_local"] = dt_utc.dt.tz_convert("Europe/Vilnius")
    df["local_date"] = df["dt_local"].dt.date
    df["hour"] = df["dt_local"].dt.hour.astype(np.int16)
    df["minute"] = df["dt_local"].dt.minute.astype(np.int16)
    df["year"] = df["dt_local"].dt.year.astype(np.int16)
    df["weekday"] = df["dt_local"].dt.dayofweek.astype(np.int8)
    return df


def first_cross(frame, ah, al):
    if frame.empty:
        return None
    hi = frame["high_mid"].to_numpy() >= ah
    lo = frame["low_mid"].to_numpy() <= al
    ih = np.flatnonzero(hi)
    il = np.flatnonzero(lo)
    h = int(ih[0]) if len(ih) else None
    l = int(il[0]) if len(il) else None
    if h is None and l is None:
        return None
    if h is not None and l is not None and h == l:
        return {"ambiguous": True, "pos": h, "direction": None}
    if l is None or (h is not None and h < l):
        return {"ambiguous": False, "pos": h, "direction": "LONG"}
    return {"ambiguous": False, "pos": l, "direction": "SHORT"}


def arr(frame, col):
    return frame[col].to_numpy(dtype=float, copy=True)


def build_candidates(df, start_hour):
    candidates = []
    stats = {"valid_days": 0, "incomplete_asia": 0, "no_trade": 0, "ambiguous": 0, "missing_start_bar": 0,
             "prebroken": 0, "fresh": 0, "both_sides_prebroken": 0}
    for day, g in df.groupby("local_date", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        # weekdays only; weekends can have stray/rollover rows but no normal session
        if pd.Timestamp(day).dayofweek >= 5:
            continue
        asia = g[(g["hour"] >= 1) & (g["hour"] < 9)]
        if len(asia) < 420:
            stats["incomplete_asia"] += 1
            continue
        stats["valid_days"] += 1
        ah = float(asia["high_mid"].max())
        al = float(asia["low_mid"].min())
        pre = g[(g["hour"] >= 9) & (g["hour"] < start_hour)]
        pre_evt = first_cross(pre, ah, al)
        both_pre = bool((pre["high_mid"] >= ah).any() and (pre["low_mid"] <= al).any())
        if both_pre:
            stats["both_sides_prebroken"] += 1

        if pre_evt is not None:
            if pre_evt["ambiguous"]:
                stats["ambiguous"] += 1
                continue
            direction = pre_evt["direction"]
            signal_row = pre.iloc[pre_evt["pos"]]
            start_rows = g[(g["hour"] == start_hour) & (g["minute"] == 0)]
            if start_rows.empty:
                stats["missing_start_bar"] += 1
                continue
            entry_bar = start_rows.iloc[0]
            raw_entry = float(entry_bar["close_mid"])
            # Conceptual chart entry is after the first 1m candle closes.
            future_raw = g[g["timestamp"] > entry_bar["timestamp"]]
            # Realistic executable entry is next candle open.
            next_rows = g[g["timestamp"] > entry_bar["timestamp"]]
            if next_rows.empty:
                stats["no_trade"] += 1
                continue
            exec_bar = next_rows.iloc[0]
            exec_entry = float(exec_bar["open_ask"] if direction == "LONG" else exec_bar["open_bid"])
            future_exec = g[g["timestamp"] >= exec_bar["timestamp"]]
            setup = "prebroken"
            stats["prebroken"] += 1
            entry_time_raw = entry_bar["dt_local"]
            entry_time_exec = exec_bar["dt_local"]
        else:
            post = g[g["hour"] >= start_hour]
            evt = first_cross(post, ah, al)
            if evt is None:
                stats["no_trade"] += 1
                continue
            if evt["ambiguous"]:
                stats["ambiguous"] += 1
                continue
            direction = evt["direction"]
            signal_row = post.iloc[evt["pos"]]
            raw_entry = ah if direction == "LONG" else al
            # Conservative spread estimate at an intra-minute trigger: max of open/close spread.
            spr = max(float(signal_row["spread_open"]), float(signal_row["spread_close"]), 0.0)
            exec_entry = raw_entry + spr / 2.0 if direction == "LONG" else raw_entry - spr / 2.0
            # OHLC does not reveal path inside breakout minute; evaluate exits from next minute.
            future_raw = g[g["timestamp"] > signal_row["timestamp"]]
            future_exec = future_raw
            if future_raw.empty:
                stats["no_trade"] += 1
                continue
            setup = "fresh"
            stats["fresh"] += 1
            entry_time_raw = signal_row["dt_local"]
            entry_time_exec = signal_row["dt_local"]

        cand = {
            "date": str(day),
            "year": int(pd.Timestamp(day).year),
            "weekday": int(pd.Timestamp(day).dayofweek),
            "direction": direction,
            "setup": setup,
            "both_pre": both_pre,
            "asia_high": ah,
            "asia_low": al,
            "asia_range": ah - al,
            "signal_time": str(signal_row["dt_local"]),
            "entry_time_raw": str(entry_time_raw),
            "entry_time_exec": str(entry_time_exec),
            "raw_entry": raw_entry,
            "exec_entry": exec_entry,
            "signal_spread": float(max(float(signal_row["spread_open"]), float(signal_row["spread_close"]), 0.0)),
            "raw_hi": arr(future_raw, "high_mid"),
            "raw_lo": arr(future_raw, "low_mid"),
            "raw_close": arr(future_raw, "close_mid"),
            "exec_bid_hi": arr(future_exec, "high_bid"),
            "exec_bid_lo": arr(future_exec, "low_bid"),
            "exec_bid_close": arr(future_exec, "close_bid"),
            "exec_ask_hi": arr(future_exec, "high_ask"),
            "exec_ask_lo": arr(future_exec, "low_ask"),
            "exec_ask_close": arr(future_exec, "close_ask"),
            "future_times_raw": [str(x) for x in future_raw["dt_local"].tolist()],
            "future_times_exec": [str(x) for x in future_exec["dt_local"].tolist()],
        }
        candidates.append(cand)
    return candidates, stats


def first_true(mask):
    x = np.flatnonzero(mask)
    return int(x[0]) if len(x) else None


def evaluate_one(c, tp, sl, mode="raw", details=False):
    direction = c["direction"]
    if mode == "raw":
        entry = c["raw_entry"]
        hi, lo, close = c["raw_hi"], c["raw_lo"], c["raw_close"]
        times = c["future_times_raw"]
    else:
        entry = c["exec_entry"]
        if direction == "LONG":
            hi, lo, close = c["exec_bid_hi"], c["exec_bid_lo"], c["exec_bid_close"]
        else:
            hi, lo, close = c["exec_ask_hi"], c["exec_ask_lo"], c["exec_ask_close"]
        times = c["future_times_exec"]
    if len(close) == 0:
        return None

    if direction == "LONG":
        tp_level, sl_level = entry + tp, entry - sl
        itp = first_true(hi >= tp_level)
        isl = first_true(lo <= sl_level)
        eod_pnl = float(close[-1] - entry)
        full_mfe = float(np.max(hi) - entry)
        full_mae = float(entry - np.min(lo))
    else:
        tp_level, sl_level = entry - tp, entry + sl
        itp = first_true(lo <= tp_level)
        isl = first_true(hi >= sl_level)
        eod_pnl = float(entry - close[-1])
        full_mfe = float(entry - np.min(lo))
        full_mae = float(np.max(hi) - entry)

    if itp is None and isl is None:
        outcome, pnl, ix = "EOD", eod_pnl, len(close) - 1
    elif itp is not None and (isl is None or itp < isl):
        outcome, pnl, ix = "TP", float(tp), itp
    else:
        # Same 1m candle touching TP and SL is conservatively counted as SL.
        outcome, pnl, ix = "SL", -float(sl), isl

    out = {"outcome": outcome, "pnl": pnl, "exit_idx": int(ix), "exit_time": times[ix] if 0 <= ix < len(times) else None,
           "full_mfe": max(0.0, full_mfe), "full_mae": max(0.0, full_mae)}
    if details:
        upto = int(ix) + 1
        if direction == "LONG":
            out["pre_exit_mfe"] = max(0.0, float(np.max(hi[:upto]) - entry))
            out["pre_exit_mae"] = max(0.0, float(entry - np.min(lo[:upto])))
        else:
            out["pre_exit_mfe"] = max(0.0, float(entry - np.min(lo[:upto])))
            out["pre_exit_mae"] = max(0.0, float(np.max(hi[:upto]) - entry))
    return out


def metrics(cands, tp, sl, mode, years=None, details=False):
    rows = []
    for c in cands:
        if years is not None and c["year"] not in years:
            continue
        ev = evaluate_one(c, tp, sl, mode=mode, details=details)
        if ev is None:
            continue
        row = {k: c[k] for k in ["date", "year", "weekday", "direction", "setup", "both_pre", "asia_range", "signal_spread"]}
        row.update(ev)
        rows.append(row)
    if not rows:
        return {}, pd.DataFrame()
    r = pd.DataFrame(rows)
    pnl = r["pnl"].to_numpy(float)
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    dd = peak[1:] - equity
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    streak = max_streak = 0
    for x in pnl:
        if x < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    wins = int((r["outcome"] == "TP").sum())
    sls = int((r["outcome"] == "SL").sum())
    eods = int((r["outcome"] == "EOD").sum())
    resolved = wins + sls
    m = {
        "n": int(len(r)), "tp_count": wins, "sl_count": sls, "eod_count": eods,
        "tp_rate_all": float(wins / len(r)),
        "win_rate_resolved": float(wins / resolved) if resolved else None,
        "positive_pnl_rate": float((pnl > 0).mean()),
        "net": float(pnl.sum()), "expectancy": float(pnl.mean()), "median_pnl": float(np.median(pnl)),
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "max_drawdown": float(dd.max()) if len(dd) else 0.0,
        "max_losing_streak": int(max_streak),
    }
    return m, r


def add_neighbor_score(grid):
    vals = []
    for _, row in grid.iterrows():
        nbr = grid[(grid["tp"].sub(row["tp"]).abs() <= 1.0) & (grid["sl"].sub(row["sl"]).abs() <= 2.5)]
        vals.append(float(nbr["train_expectancy"].mean()))
    grid = grid.copy()
    grid["neighbor_score"] = vals
    return grid


def make_grid(cands, start_hour, mode):
    rows = []
    for tp in TPS:
        for sl in SLS:
            train, _ = metrics(cands, tp, sl, mode, years={2023, 2024, 2025})
            oos, _ = metrics(cands, tp, sl, mode, years={2026})
            allm, _ = metrics(cands, tp, sl, mode, years=None)
            row = {"start_hour": start_hour, "mode": mode, "tp": tp, "sl": sl}
            for prefix, m in [("train", train), ("oos", oos), ("all", allm)]:
                for k, v in m.items():
                    row[f"{prefix}_{k}"] = v
            rows.append(row)
    grid = pd.DataFrame(rows)
    grid = add_neighbor_score(grid)
    return grid.sort_values(["neighbor_score", "train_expectancy"], ascending=False).reset_index(drop=True)


def qdict(series):
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return {}
    qs = s.quantile([0, .10, .25, .50, .75, .90, .95, .99, 1.0])
    return {str(k): float(v) for k, v in qs.items()}


def breakdown_table(cands, tp, sl, mode, start_hour, label):
    _, r = metrics(cands, tp, sl, mode, years=None, details=True)
    if r.empty:
        return pd.DataFrame()
    out = []
    groups = [
        ("year", ["year"]),
        ("direction", ["direction"]),
        ("setup", ["setup"]),
        ("weekday", ["weekday"]),
        ("year_direction", ["year", "direction"]),
        ("year_setup", ["year", "setup"]),
    ]
    for kind, cols in groups:
        for keys, g in r.groupby(cols):
            if not isinstance(keys, tuple): keys = (keys,)
            pnl = g["pnl"].to_numpy(float)
            rec = {"label": label, "start_hour": start_hour, "mode": mode, "tp": tp, "sl": sl, "group_type": kind,
                   "group": "|".join(map(str, keys)), "n": len(g), "tp_count": int((g.outcome == "TP").sum()),
                   "sl_count": int((g.outcome == "SL").sum()), "eod_count": int((g.outcome == "EOD").sum()),
                   "net": float(pnl.sum()), "expectancy": float(pnl.mean()), "positive_rate": float((pnl > 0).mean())}
            out.append(rec)
    return pd.DataFrame(out)


def main():
    df = load_data()
    data_summary = {
        "rows": int(len(df)),
        "first_utc": str(df["dt_utc"].iloc[0]),
        "last_utc": str(df["dt_utc"].iloc[-1]),
        "first_local": str(df["dt_local"].iloc[0]),
        "last_local": str(df["dt_local"].iloc[-1]),
        "spread_close_mean": float(df["spread_close"].mean()),
        "spread_close_median": float(df["spread_close"].median()),
        "spread_close_p90": float(df["spread_close"].quantile(.90)),
        "spread_close_p95": float(df["spread_close"].quantile(.95)),
        "spread_close_p99": float(df["spread_close"].quantile(.99)),
    }
    summary = {"data": data_summary, "rules": {
        "asia": "01:00<=LT<09:00 Europe/Vilnius DST-aware",
        "one_trade_per_day": True,
        "prebreak_direction": "first Asia high/low side crossed between 09:00 and start time",
        "if_both_same_minute": "skip as ambiguous",
        "prebroken_entry_raw": "close of start-time 1m candle; exits checked from next minute",
        "prebroken_entry_exec": "next 1m candle executable open (ask for long, bid for short)",
        "fresh_entry_raw": "Asia boundary; exits checked from next minute",
        "fresh_entry_exec": "Asia boundary +/- half conservative open/close spread; exits checked from next minute",
        "same_exit_candle_tp_sl": "SL first (conservative)",
        "eod": "if neither TP nor SL hits by local day end, close/mark at final executable close"
    }, "starts": {}}

    all_breakdowns = []
    for start_hour in START_HOURS:
        cands, diag = build_candidates(df, start_hour)
        print(f"Start {start_hour}: {len(cands)} candidates, diag={diag}", flush=True)
        summary["starts"][str(start_hour)] = {"diagnostics": diag, "candidate_count": len(cands)}

        for mode in ["raw", "exec"]:
            grid = make_grid(cands, start_hour, mode)
            grid.to_csv(OUT / f"grid_{start_hour}_{mode}.csv", index=False)
            baseline_train, _ = metrics(cands, BASE_TP, BASE_SL, mode, years={2023, 2024, 2025})
            baseline_oos, _ = metrics(cands, BASE_TP, BASE_SL, mode, years={2026})
            baseline_all, base_trades = metrics(cands, BASE_TP, BASE_SL, mode, years=None, details=True)

            # In-sample robust selection: maximize neighborhood expectancy; OOS not used for selection.
            eligible = grid[grid["train_n"] >= max(100, int(grid["train_n"].max() * 0.90))]
            best = eligible.sort_values(["neighbor_score", "train_expectancy"], ascending=False).iloc[0].to_dict()
            best_tp, best_sl = float(best["tp"]), float(best["sl"])
            best_train, _ = metrics(cands, best_tp, best_sl, mode, years={2023, 2024, 2025})
            best_oos, _ = metrics(cands, best_tp, best_sl, mode, years={2026})
            best_all, best_trades = metrics(cands, best_tp, best_sl, mode, years=None, details=True)

            summary["starts"][str(start_hour)][mode] = {
                "baseline_5_15": {"train": baseline_train, "oos": baseline_oos, "all": baseline_all},
                "robust_selected": {"tp": best_tp, "sl": best_sl, "neighbor_score": float(best["neighbor_score"]),
                                    "train": best_train, "oos": best_oos, "all": best_all},
                "top10_train_by_neighbor": grid.head(10)[["tp", "sl", "neighbor_score", "train_n", "train_tp_rate_all", "train_net", "train_expectancy", "train_profit_factor", "train_max_drawdown", "oos_n", "oos_tp_rate_all", "oos_net", "oos_expectancy", "oos_profit_factor", "oos_max_drawdown"]].to_dict("records")
            }

            for label, tp, sl, trades in [("baseline_5_15", BASE_TP, BASE_SL, base_trades), ("robust_selected", best_tp, best_sl, best_trades)]:
                br = breakdown_table(cands, tp, sl, mode, start_hour, label)
                if not br.empty: all_breakdowns.append(br)
                wins = trades[trades["outcome"] == "TP"] if not trades.empty else trades
                losses = trades[trades["outcome"] == "SL"] if not trades.empty else trades
                summary["starts"][str(start_hour)][mode][label + "_path_stats"] = {
                    "full_mfe_quantiles": qdict(trades["full_mfe"]) if not trades.empty else {},
                    "full_mae_quantiles": qdict(trades["full_mae"]) if not trades.empty else {},
                    "winner_pre_exit_mae_quantiles": qdict(wins["pre_exit_mae"]) if not wins.empty else {},
                    "loser_pre_exit_mfe_quantiles": qdict(losses["pre_exit_mfe"]) if not losses.empty else {},
                    "asia_range_quantiles": qdict(trades["asia_range"]) if not trades.empty else {},
                    "signal_spread_quantiles": qdict(trades["signal_spread"]) if not trades.empty else {},
                }

        # Save baseline detailed trade list for audit (raw + exec outcomes merged by date).
        _, rr = metrics(cands, BASE_TP, BASE_SL, "raw", years=None, details=True)
        _, ee = metrics(cands, BASE_TP, BASE_SL, "exec", years=None, details=True)
        if not rr.empty:
            rr.to_csv(OUT / f"trades_{start_hour}_baseline_raw.csv", index=False)
        if not ee.empty:
            ee.to_csv(OUT / f"trades_{start_hour}_baseline_exec.csv", index=False)

    if all_breakdowns:
        pd.concat(all_breakdowns, ignore_index=True).to_csv(OUT / "breakdowns.csv", index=False)
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("BACKTEST_COMPLETE")
    print(json.dumps(summary, ensure_ascii=False)[:12000])


if __name__ == "__main__":
    main()

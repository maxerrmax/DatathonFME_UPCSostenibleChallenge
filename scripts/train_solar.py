"""
Ajusta el model físic de solar (src/solar.py) i desa solar.json.

    python scripts/train_solar.py            -> dificil/arxiu/submission_hard/val_solar.json (sense els blocs de validació 15-21)
    python scripts/train_solar.py --final    -> dificil/arxiu/submission_hard/solar.json (tot el train)
    python scripts/train_solar.py --exclude-teamval  -> val_solar_teamval.json (sense la validació de l'equip)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from src.data import load_raw_hourly
from src.solar import CS_MIN, build_tables, fit_weights, save
import train_forecast as T


def main():
    df = load_raw_hourly()
    times, pv = df.index, df["pv_kw"].values
    blk = [T.block_of(t) for t in times]
    if "--final" in sys.argv:
        hold = np.zeros(len(df), bool); out = "solar.json"
    elif "--exclude-teamval" in sys.argv:
        sys.path.insert(0, str(ROOT / "estandard"))
        from data import split_val
        from src.data import load_hourly
        _, val = split_val(load_hourly(pv_scale=12))
        vs = set(val.index)
        hold = np.array([t in vs for t in times]); out = "val_solar_teamval.json"
        blk = [(t.year, t.month, "val") if h else (t.year, t.month, (t.day - 1) // 7)
                        for t, h in zip(times, hold)]
    else:
        hold = np.array([b[2] == 2 for b in blk]); out = "val_solar.json"
    keep = ~hold
    cs, ktc = build_tables(times, pv, keep)
    csv = cs[times.dayofyear.values - 1, times.hour.values]
    kt = np.where(csv > CS_MIN, np.clip(pv / np.maximum(csv, 1e-9), 0, 1.3), np.nan)
    mon, hr = times.month.values, times.hour.values
    n = len(df)
    samples = {k: [] for k in range(24)}
    for i in np.where(keep)[0]:
        b = blk[i]
        kn = kt[i - 1] if i > 0 and blk[i - 1] == b else np.nan
        js = [j for j in range(i - 24, i) if j >= 0 and blk[j] == b and csv[j] > CS_MIN and pv[j] == pv[j]]
        den = csv[js].sum() if js else 0.0
        kd = pv[js].sum() / den if den > 0 else np.nan
        for k in range(24):
            j = i + k
            if j >= n or blk[j] != b or pv[j] != pv[j]:
                break
            samples[k].append((pv[j], csv[j], kn, kd, ktc[mon[j] - 1, hr[j]]))
    w_now, w_d1 = fit_weights(samples)
    save(ROOT / "dificil/arxiu/submission_hard" / out, cs, ktc, w_now, w_d1)
    print("desat", out, "| w_now:", np.round(w_now, 2).tolist(), "| w_d1:", np.round(w_d1, 2).tolist())


if __name__ == "__main__":
    main()

"""
Completa l'historial inicial (hist0) del v8 amb TOTES les dades disponibles de cada sèrie, sense reentrenar.

El v8 construeix hist0 amb el dataset horari després d'eliminar les hores on falta algun sensor; per això
acaba el 16/08 i no té novembre, i les setmanes de test 24-30/11 i 25-31/08 comencen sense historial.
Aquí cada sèrie (preu, CO2, consum, solar ×12, irradiància, temperatura, vent) s'omple per separat.

Ús:
    python scripts/extend_hist0_v8.py <pkl_entrada> <pkl_sortida> [--exclude-val]
        --exclude-val: treu les hores de la validació de l'equip (per a proves amb models `fit`)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "estandard"))

import joblib
import numpy as np
import pandas as pd

from src.data import load_raw_hourly

PV_SCALE = 12.0
COLS = {"price": "precio_eur_per_kwh", "co2": "co2_kg_per_kwh", "load": "load_kw", "pv": "pv_kw",
        "irr": "irradiancia_wm2", "temp": "temp_ambient_c", "vent": "vent_ms"}


def epoch_hours(idx):
    return ((idx - pd.Timestamp("1970-01-01")) // pd.Timedelta("1h")).astype(int)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dump = joblib.load(src)
    old = dump["hist0"]
    raw = load_raw_hourly()
    raw["pv_kw"] = raw["pv_kw"] * PV_SCALE
    if "--exclude-val" in sys.argv:
        from data import split_val
        from src.data import load_hourly
        _, val = split_val(load_hourly(pv_scale=PV_SCALE))
        raw = raw.drop(index=raw.index.intersection(val.index))
    new = {}
    for k, c in COLS.items():
        s = raw[c].dropna()
        eh = epoch_hours(s.index).values.astype(np.int64)
        new[k] = (eh, s.values.astype(float))
        # comprovació: a les hores comunes, els valors han de coincidir amb els del v8
        oh, ov = np.asarray(old[k][0]), np.asarray(old[k][1], float)
        m = dict(zip(eh.tolist(), s.values.tolist()))
        common = [(v, m[h]) for h, v in zip(oh.tolist(), ov.tolist()) if h in m and np.isfinite(v)]
        diff = max((abs(a - b) for a, b in common), default=0.0)
        fmt = lambda e: (pd.Timestamp("1970-01-01") + pd.Timedelta(hours=int(e))).strftime("%Y-%m-%d %H")
        print(f"{k:5s} v8: {len(oh):5d} h ({fmt(oh.min())} -> {fmt(oh.max())}) | nou: {len(eh):5d} h "
              f"({fmt(eh.min())} -> {fmt(eh.max())}) | dif. màx. a hores comunes: {diff:.2e}")
    dump["hist0"] = new
    joblib.dump(dump, dst, compress=3)
    print("desat", dst, f"{dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

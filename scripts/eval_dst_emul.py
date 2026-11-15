"""
Simula el canvi d'hora en setmanes de validació per mesurar si l'agent el gestiona bé (el train no en té cap).

Al test, les setmanes 25-31/3 i 25-31/10 contenen el canvi d'hora: el consum segueix el rellotge (horari humà),
però el sol (i amb ell la solar, la meteo, el CO2 i en part el preu) es desplaça 1 h respecte del rellotge,
mentre que els models s'han entrenat amb l'horari del mes (estiu a octubre, hivern a març).

Emulació sobre la validació de l'equip (models `fit`):
  - "atrassar" (com l'octubre): setmanes d'hora d'estiu (17/09, 18/10): a l'hora de rellotge h hi posem el sol de h+1
    i forcem dst=0 (hivern) aquells dies;
  - "avançar" (com el març): setmanes d'hora d'hivern (15/02, 18/03): a l'hora h hi posem el sol de h-1 i forcem dst=1.
El consum no es mou. És una aproximació (el preu també té una part que segueix el rellotge).

Ús:
    python scripts/eval_dst_emul.py <dir_agent>['|{json params}'] [<dir2>['|{...}'] ...] [--noshift]
        --noshift: les mateixes 4 setmanes sense canvi d'hora (referència)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dificil")); sys.path.insert(0, str(ROOT / "estandard"))

SUN = ["pv_kw", "irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms", "co2_kg_per_kwh", "precio_eur_per_kwh"]
NOSHIFT = "--noshift" in sys.argv
WEEKS = {"2025-09-17": ("atrassar", -1, 0), "2025-10-18": ("atrassar", -1, 0),
         "2026-02-15": ("avançar", +1, 1), "2026-03-18": ("avançar", +1, 1)}


def emulated():
    from hdata import load_hard
    from data import split_val
    val = split_val(load_hard())[1]
    idx = val.index
    starts = [0] + [i for i in range(1, len(idx)) if idx[i] - idx[i - 1] != pd.Timedelta("1h")]
    ends = starts[1:] + [len(idx)]
    parts, ovr, names = [], {}, []
    for s, e in zip(starts, ends):
        b = val.iloc[s:e].copy()
        k = str(b.index[0].date())
        if k not in WEEKS:
            continue
        _, sh, d = WEEKS[k]
        cols = [c for c in SUN if c in b.columns]
        if NOSHIFT:
            parts.append(b); names.append((k, len(b))); continue
        b[cols] = b[cols].shift(-1 if sh < 0 else 1).ffill().bfill()
        for day in np.unique(((b.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1h")).values // 24):
            ovr[int(day)] = d
        parts.append(b); names.append((k, len(b)))
    return pd.concat(parts), ovr, names


def one(spec):
    d, params = spec
    sys.path.insert(0, d)
    df, ovr, names = emulated()
    from henv import HardEnv
    spec_ = importlib.util.spec_from_file_location("agent_eval", os.path.join(d, "agent.py"))
    m = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(m)
    import feats
    feats._OVR = ovr
    if hasattr(feats, "_dst_bounds"):
        feats._dst_bounds.cache_clear()
    ag = m.BatteryAgent(params) if params else m.BatteryAgent()
    env = HardEnv(df.reset_index()); o = env.reset(); done = False
    while not done:
        o, done = env.step(ag.act(o))
    tot = env.evaluate_results()
    per, p0 = [], 0
    for k, n in names:
        mask = np.zeros(len(df), bool); mask[p0:p0 + n] = True; p0 += n
        per.append((k, env.evaluate_results(mask=mask)["score"]))
    return os.path.basename(d.rstrip("/")), params, tot["score"], tot["co2_estalviat_pct"], tot["cost_estalviat_pct"], per


if __name__ == "__main__":
    specs = []
    for a in [x for x in sys.argv[1:] if not x.startswith("--")]:
        d, _, js = a.partition("|")
        specs.append((os.path.abspath(d), json.loads(js) if js else {}))
    with Pool(min(len(specs), os.cpu_count() or 1)) as p:
        for name, prm, s, c, k, per in p.map(one, specs):
            print(f"{s:6.2f}  CO2={c:6.2f}  cost={k:6.2f}  {name} {json.dumps(prm)}  | " +
                  "  ".join(f"{w[5:]}:{v:5.2f}" for w, v in per))

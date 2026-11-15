"""Prova l'agent amb dades trencades: NaN, consum 0, llistes curtes, claus absents, salts de temps."""
import sys, time, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from eval_agent_hard import load_agent_cls
from src.data import load_hourly, validation_weeks
from src.hard_env import run, SPEC_STD

A = load_agent_cls()
val = validation_weeks(load_hourly(pv_scale=12))
md = ROOT / "dificil/arxiu/submission_hard"

# 1) obs patològiques directes (agent nou a cada cas)
ts = pd.Timestamp("2026-03-25 10:00")
base = {"timestamp": ts, "hour": 10, "day_of_week": 2, "soc_kwh": 3000.0, "soc_pct": 0.5, "preu_actual": 0.1,
        "co2_actual": 0.08, "preus_properes_24h": [0.1, 0.12, 0.11], "co2_properes_24h": [0.08, 0.07, 0.09],
        "noves_dades_5min": {"ultim_consum_kw": 900.0, "ultima_solar_kw": 300.0, "irradiancia_wm2": 500.0,
                             "temp_ambient_c": 18.0, "temp_panel_c": 25.0, "vent_ms": 2.0},
        "cap_max_kwh": 6000.0, "power_max_kw": 1500.0}
cases = {
    "normal": {},
    "NaN previsions": {"preus_properes_24h": [float("nan")] * 3, "co2_properes_24h": [None, float("nan"), 0.1]},
    "llistes buides": {"preus_properes_24h": [], "co2_properes_24h": []},
    "sense noves_dades": {"noves_dades_5min": {}},
    "consum NaN": {"noves_dades_5min": {"ultim_consum_kw": float("nan"), "ultima_solar_kw": None}},
    "soc NaN": {"soc_kwh": float("nan")},
    "soc fora de rang": {"soc_kwh": 9999.0},
    "preu negatiu i enorme": {"preus_properes_24h": [-0.05, 5.0, 0.1]},
    "sense cap/power": {"cap_max_kwh": None, "power_max_kw": None},
}
for name, upd in cases.items():
    o = dict(base); o.update(upd)
    if "cap_max_kwh" in upd and upd["cap_max_kwh"] is None:
        o.pop("cap_max_kwh"); o.pop("power_max_kw")
    a = A(model_dir=md).act(o)
    print(f"{name:24s} -> {a:9.2f} kW  {'OK' if isinstance(a, float) and math.isfinite(a) else 'ERROR'}")

# 2) setmana sense dades de sensors (consum 0), com podria passar al novembre del test
broken = val.copy()
blk = broken.index[(broken.index.month == 3)]
broken.loc[blk, ["load_kw", "pv_kw"]] = 0.0
r_ok = run(A(model_dir=md), val); r_b = run(A(model_dir=md), broken)
print(f"validació normal: {r_ok['score']:.2f}  infr={r_ok['infraccions']} | amb una setmana a 0: {r_b['score']:.2f} infr={r_b['infraccions']}")

# 3) Mode Estàndard (2.000 kWh, previsió 24 h, solar real)
val1 = validation_weeks(load_hourly(pv_scale=1))
t = time.time(); r_s = run(A(model_dir=md), val1, spec=SPEC_STD)
print(f"Mode Estàndard: {r_s['score']:.2f} infr={r_s['infraccions']} ({time.time()-t:.0f}s)")

"""Compara agents al Mode Difícil sobre les setmanes de validació (dies 15-21 de cada mes)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data import load_hourly, validation_weeks
from src.hard_env import run
from src.agents import KitHeuristic, OracleMPC, NaiveForecastMPC

df = load_hourly(pv_scale=12)
val = validation_weeks(df)
base = np.maximum(df.load_kw - df.pv_kw, 0)
kc, kp = (base * df.co2_kg_per_kwh).mean(), (base * df.precio_eur_per_kwh).mean()
agents = {
    "Heurística kit x3": lambda: KitHeuristic(),
    "MPC previsió ingènua (ahir)": lambda: NaiveForecastMPC(kc, kp),
    "MPC previsió perfecta 24h": lambda: OracleMPC(val, kc, kp),
}
for name, mk in agents.items():
    t = time.time(); r = run(mk(), val)
    print(f"{name:32s} score={r['score']:6.2f}  CO2={r['co2_estalviat_pct']:6.2f}%  cost={r['cost_estalviat_pct']:6.2f}%  infr={r['infraccions']}  ({time.time()-t:.0f}s)")

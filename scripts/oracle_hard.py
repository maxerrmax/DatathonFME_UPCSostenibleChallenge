"""Cota superior del Mode Difícil: despatx òptim coneixent tot el futur (consum, solar, preu, CO2)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data import load_hourly, validation_weeks
from src.optim import solve_dispatch, score_weights, score_from_actions

df = load_hourly(pv_scale=12)
val = validation_weeks(df)
net = (val.load_kw - val.pv_kw).values
print(f"hores validació: {len(val)} | hores amb excedent solar: {(net<0).sum()} ({(net<0).mean():.1%})"
      f" | excedent total: {(-net[net<0]).sum()/1000:.0f} MWh | net mín {net.min():.0f} kW")
w = score_weights(val)
for cap, pmax, name in [(6000, 1500, "Mode Difícil"), (2000, 500, "bateria estàndard + solar x12")]:
    t = time.time()
    p = solve_dispatch(w, net, cap / 2, cap, pmax)
    print(name, {k: round(v, 2) for k, v in score_from_actions(val, p).items()}, f"{time.time()-t:.1f}s")

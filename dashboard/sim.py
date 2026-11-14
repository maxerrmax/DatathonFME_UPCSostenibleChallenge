"""
dashboard/sim.py
Simulación de agentes sobre los datos de train con el simulador oficial
(starter_kit/environment.py) y cálculo de la cota superior teórica
(programación lineal con previsión perfecta de todo el periodo).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import diags, hstack, identity, vstack

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "starter_kit"
SPEC = json.loads((STARTER / "battery_spec.json").read_text(encoding="utf-8"))

ALPHA, BETA = 0.7, 0.3


def _load_env_module():
    spec = importlib.util.spec_from_file_location("campus_env", STARTER / "environment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_agents() -> dict[str, Path]:
    """Busca en el repo todos los .py que definen `class BatteryAgent`."""
    found = {}
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT)
        if rel.parts[0] in {".git", "dashboard", "data"} or ".ipynb_checkpoints" in rel.parts:
            continue
        try:
            if "class BatteryAgent" in p.read_text(encoding="utf-8", errors="ignore"):
                found[str(rel)] = p
        except OSError:
            continue
    return found


def load_agent(path: Path):
    """Importa un agent.py y devuelve una instancia nueva de BatteryAgent."""
    folder = str(path.parent)
    added = folder not in sys.path
    if added:
        sys.path.insert(0, folder)  # para que el agente pueda importar sus módulos locales
    try:
        name = f"agent_{abs(hash(str(path)))}_{time.time_ns()}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.BatteryAgent()
    finally:
        if added:
            sys.path.remove(folder)


def run_agent(agent, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Ejecuta el agente hora a hora con el simulador oficial.

    frame: columnas load_kw, pv_kw, precio_eur_per_kwh, co2_kg_per_kwh, indexado por Fecha.
    """
    env_mod = _load_env_module()
    env = env_mod.CampusBatteryEnv(frame, spec=dict(SPEC))
    obs = env.reset()
    done = False
    t0 = time.perf_counter()
    while not done:
        action = agent.act(obs)
        obs, done = env.step(action)
    elapsed = time.perf_counter() - t0
    res = env.evaluate_results(alpha=ALPHA, beta=BETA)
    res["segundos"] = round(elapsed, 2)

    soc_prev = np.r_[env.soc_init, np.asarray(env.soc_series)[:-1]]
    soc_now = np.asarray(env.soc_series)
    p_real = np.where(soc_now >= soc_prev,
                      (soc_now - soc_prev) / env.eta_c,
                      (soc_now - soc_prev) * env.eta_d)
    trace = pd.DataFrame({
        "consigna_kw": env.actions,
        "bateria_kw": p_real,
        "soc_kwh": soc_now,
        "red_kw": env.grid,
        "infracciones": env.infraccions_per_step,
    }, index=frame.index)
    return res, trace


def optimal_lp(frame: pd.DataFrame, soc0_frac: float | None = None) -> tuple[dict, pd.DataFrame]:
    """Cota superior: LP con previsión perfecta de precio, CO2, consumo y solar.

    Maximiza exactamente el score oficial (0,7·ΔCO2% + 0,3·ΔCoste%), sin infracciones.
    Variables: carga c_t ≥ 0, descarga d_t ≥ 0, energía s_t ∈ [0, cap].
    """
    n = len(frame)
    cap, pmax = float(SPEC["capacity_kwh"]), float(SPEC["power_kw"])
    eta_c = eta_d = float(SPEC["efficiency"]) ** 0.5
    soc0 = (SPEC["soc_init"] if soc0_frac is None else soc0_frac) * cap

    load = frame["load_kw"].to_numpy(float)
    pv = frame["pv_kw"].to_numpy(float)
    price = frame["precio_eur_per_kwh"].to_numpy(float)
    co2 = frame["co2_kg_per_kwh"].to_numpy(float)
    base = np.maximum(load - pv, 0.0)
    co2_base, cost_base = float(base @ co2), float(base @ price)

    # Peso de cada kWh importado en el score (en %)
    w = 100 * (ALPHA * co2 / co2_base + BETA * price / cost_base)
    c_obj = np.r_[w, -w, np.zeros(n)]  # minimizamos w·(c − d)

    # s_t − s_{t−1} − η_c·c_t + d_t/η_d = 0  (s_{−1} = soc0)
    D = identity(n, format="csr") - diags([np.ones(n - 1)], [-1], shape=(n, n), format="csr")
    A_eq = hstack([-eta_c * identity(n), (1 / eta_d) * identity(n), D], format="csr")
    b_eq = np.zeros(n)
    b_eq[0] = soc0

    # No descargar más que la demanda neta (la exportación no se paga)
    bounds = ([(0, pmax)] * n
              + [(0, min(pmax, b)) for b in base]
              + [(0, cap)] * n)
    sol = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not sol.success:
        raise RuntimeError(sol.message)

    ch, dis, soc = sol.x[:n], sol.x[n:2 * n], sol.x[2 * n:]
    p = ch - dis
    grid = np.maximum(load - pv + p, 0.0)
    co2_ag, cost_ag = float(grid @ co2), float(grid @ price)
    d_co2 = (co2_base - co2_ag) / co2_base
    d_cost = (cost_base - cost_ag) / cost_base
    res = {
        "score": round(max(0.0, (ALPHA * d_co2 + BETA * d_cost) * 100), 2),
        "co2_estalviat_pct": round(d_co2 * 100, 2),
        "cost_estalviat_pct": round(d_cost * 100, 2),
        "co2_estalviat_kg": round(co2_base - co2_ag, 1),
        "cost_estalviat_eur": round(cost_base - cost_ag, 2),
        "co2_baseline_kg": round(co2_base, 1),
        "cost_baseline_eur": round(cost_base, 2),
        "infraccions": 0,
    }
    trace = pd.DataFrame({
        "consigna_kw": p, "bateria_kw": p, "soc_kwh": soc,
        "red_kw": grid, "infracciones": 0,
    }, index=frame.index)
    return res, trace


def daily_cycle_value(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada día: hora más limpia/sucia y margen de CO2 y precio de un ciclo.

    Un ciclo compensa si  valor_descarga · η  >  valor_carga  (η = 0,9 ida y vuelta).
    """
    eta = float(SPEC["efficiency"])
    g = df.dropna(subset=["co2_kg_per_kwh", "precio_eur_per_kwh"]).groupby("fecha_dia")
    out = pd.DataFrame({
        "co2_min": g["co2_kg_per_kwh"].min(),
        "co2_max": g["co2_kg_per_kwh"].max(),
        "precio_min": g["precio_eur_per_kwh"].min(),
        "precio_max": g["precio_eur_per_kwh"].max(),
        "hora_co2_min": g["co2_kg_per_kwh"].idxmin().dt.hour,
        "hora_co2_max": g["co2_kg_per_kwh"].idxmax().dt.hour,
        "hora_precio_min": g["precio_eur_per_kwh"].idxmin().dt.hour,
        "hora_precio_max": g["precio_eur_per_kwh"].idxmax().dt.hour,
    })
    out["margen_co2"] = out["co2_max"] * eta - out["co2_min"]
    out["margen_precio"] = out["precio_max"] * eta - out["precio_min"]
    return out

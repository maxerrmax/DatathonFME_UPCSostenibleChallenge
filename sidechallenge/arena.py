"""Simulation engine for UPC Energy Intelligence.

Single source of truth for every battery number shown in the dashboard:

* ``official_eval`` replicates ``starter_kit/environment.py`` step by step
  (clipping, infractions, SOC limits, score formula). A regression test checks
  that both give identical results.
* ``lp_schedule`` is a perfect-information linear programme with a numerical
  safety margin, so its schedules never trigger clipping infractions.
* ``run_arena`` back-tests policies on pseudo-test weeks with the official
  ``CampusBatteryEnv`` (agents only see ``obs``, exactly as in the challenge).

The dashboard never writes to the participant's agent: it imports it read-only.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STARTER = ROOT / "starter_kit"
COLS = ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]
ALPHA = 0.7
SOC_MARGIN = 1e-3     # kWh kept away from 0 and capacity in LP schedules
POWER_MARGIN = 1e-6   # kW kept away from the inverter limit
OFFICIAL_PENALTY = 0.002   # value hard-coded in starter_kit/environment.py (0.2 points)
STATEMENT_PENALTY = 0.01   # value written in ENUNCIAT.md (1.00 point)

try:
    SPEC: dict = json.loads((STARTER / "battery_spec.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    SPEC = {"capacity_kwh": 2000, "power_kw": 500, "efficiency": 0.9, "soc_init": 0.5,
            "soc_min": 0.0, "soc_max": 1.0, "dt_hours": 1}


def battery(cap: float | None = None, power: float | None = None, eta: float | None = None) -> dict:
    cap = float(cap if cap is not None else SPEC["capacity_kwh"])
    return {"cap": cap, "power": float(power if power is not None else SPEC["power_kw"]),
            "eta": float(eta if eta is not None else SPEC["efficiency"]),
            "soc0": float(SPEC.get("soc_init", 0.5)) * cap,
            "soc_min": float(SPEC.get("soc_min", 0.0)) * cap,
            "soc_max": float(SPEC.get("soc_max", 1.0)) * cap}


# --------------------------------------------------------------------------
# Official physics and score
# --------------------------------------------------------------------------
def official_eval(frame: pd.DataFrame, actions, bat: dict | None = None,
                  alpha: float = ALPHA, beta: float | None = None) -> dict:
    """Exact replica of CampusBatteryEnv.step + evaluate_results for a schedule."""
    bat = bat or battery()
    beta = 1 - alpha if beta is None else beta
    load = frame["load_kw"].to_numpy(float); pv = frame["pv_kw"].to_numpy(float)
    price = frame["precio_eur_per_kwh"].to_numpy(float); co2 = frame["co2_kg_per_kwh"].to_numpy(float)
    eta_c = eta_d = math.sqrt(bat["eta"])
    soc = bat["soc0"]; clipped = violations = 0
    grid = np.zeros(len(load)); applied = np.zeros(len(load)); socs = np.zeros(len(load)); infr = np.zeros(len(load), int)
    for t, req in enumerate(np.asarray(actions, float)):
        now = 0
        p = float(np.clip(req, -bat["power"], bat["power"]))
        if p != req:
            clipped += 1; now += 1
        if p > 0:
            max_c = (bat["soc_max"] - soc) / eta_c
            pa = min(p, max(0.0, max_c))
            if pa < p:
                clipped += 1; now += 1
            soc += pa * eta_c
        elif p < 0:
            max_d = (soc - bat["soc_min"]) * eta_d
            pa = -min(-p, max(0.0, max_d))
            if pa > p:
                clipped += 1; now += 1
            soc += pa / eta_d
        else:
            pa = 0.0
        if soc < bat["soc_min"] - 1e-6 or soc > bat["soc_max"] + 1e-6:
            violations += 1; now += 1
        soc = min(max(soc, bat["soc_min"]), bat["soc_max"])
        net = load[t] - pv[t] + pa
        grid[t] = max(net, 0.0); applied[t] = pa; socs[t] = soc; infr[t] = now
    base = np.maximum(load - pv, 0.0)
    cost, cost0 = float(grid @ price), float(base @ price)
    kg, kg0 = float(grid @ co2), float(base @ co2)
    d_co2 = (kg0 - kg) / kg0 if kg0 > 0 else 0.0
    d_cost = (cost0 - cost) / cost0 if cost0 > 0 else 0.0
    n_infr = int(clipped + violations)
    raw = alpha * d_co2 + beta * d_cost
    return {
        "score": round(max(0.0, (raw - OFFICIAL_PENALTY * n_infr) * 100), 2),
        "score_statement_penalty": round(max(0.0, (raw - STATEMENT_PENALTY * n_infr) * 100), 2),
        "score_raw": round(raw * 100, 3),
        "co2_saved_pct": round(d_co2 * 100, 3), "cost_saved_pct": round(d_cost * 100, 3),
        "co2_saved_kg": round(kg0 - kg, 2), "cost_saved_eur": round(cost0 - cost, 2),
        "co2_kg": round(kg, 2), "cost_eur": round(cost, 2),
        "co2_baseline_kg": round(kg0, 2), "cost_baseline_eur": round(cost0, 2),
        "infractions": n_infr, "soc_end_kwh": round(float(socs[-1]) if len(socs) else bat["soc0"], 2),
        "_soc": socs, "_applied": applied, "_grid": grid, "_infr": infr,
    }


def public(result: dict) -> dict:
    """Drop numpy arrays (keys starting with '_') before JSON serialisation."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def safe_action(p: float, soc: float, bat: dict) -> float:
    """Clamp a requested power so the official simulator never clips it."""
    eta_c = eta_d = math.sqrt(bat["eta"])
    hi = min(bat["power"], (bat["soc_max"] - soc) / eta_c) - POWER_MARGIN
    lo = -min(bat["power"], (soc - bat["soc_min"]) * eta_d) + POWER_MARGIN
    return float(min(max(p, min(lo, 0.0)), max(hi, 0.0)))


# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------
def weights(price: np.ndarray, co2: np.ndarray, alpha: float, scale_co2: float, scale_cost: float) -> np.ndarray:
    """Marginal score value (points) of importing 1 kWh in each hour."""
    return 100 * (alpha * co2 / scale_co2 + (1 - alpha) * price / scale_cost)


def lp_schedule(load, pv, price, co2, soc0: float, *, alpha: float = ALPHA, bat: dict | None = None,
                scale_co2: float | None = None, scale_cost: float | None = None,
                soc_end_min: float | None = None, terminal_value: float = 0.0) -> np.ndarray:
    """Perfect-information dispatch minimising the weighted 70/30 import.

    Grid import is an explicit variable g >= max(0, net + p), so solar surplus,
    if any, is handled exactly as in the official balance (export is worth 0).
    """
    bat = bat or battery()
    load, pv, price, co2 = (np.asarray(x, float) for x in (load, pv, price, co2))
    n = len(load); net = load - pv
    base = np.maximum(net, 0)
    scale_co2 = scale_co2 or float(base @ co2) or 1.0
    scale_cost = scale_cost or float(base @ price) or 1.0
    w = weights(price, co2, alpha, scale_co2, scale_cost)
    ec = ed = math.sqrt(bat["eta"])
    I = sparse.eye(n, format="csr")
    D = I - sparse.diags([np.ones(n - 1)], [-1], shape=(n, n), format="csr")
    Z = sparse.csr_matrix((n, n))
    # variables x = [c, d, s, g]
    A_eq = sparse.hstack([-ec * I, (1 / ed) * I, D, Z], format="csr")
    b_eq = np.zeros(n); b_eq[0] = soc0
    A_ub = sparse.hstack([I, -I, Z, -I], format="csr")     # c - d - g <= -net
    b_ub = -net
    pmax = bat["power"] - POWER_MARGIN
    lo_s, hi_s = bat["soc_min"] + SOC_MARGIN, bat["soc_max"] - SOC_MARGIN
    s_bounds = [(lo_s, hi_s)] * n
    if soc_end_min is not None:
        s_bounds[-1] = (min(max(soc_end_min, lo_s), hi_s), hi_s)
    bounds = [(0, pmax)] * n + [(0, pmax)] * n + s_bounds + [(0, None)] * n
    cost = np.r_[np.zeros(n), np.zeros(n), np.zeros(n), w]
    cost[3 * n - 1] -= terminal_value
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("LP infeasible: " + str(res.message))
    return res.x[:n] - res.x[n:2 * n]


# --------------------------------------------------------------------------
# Policies (all expose act(obs) exactly like a BatteryAgent)
# --------------------------------------------------------------------------
class IdleAgent:
    name = "Sense bateria"
    def act(self, obs):
        return 0.0


class WindowAgent:
    """Fixed hourly windows (the kind of rule a human reads off a heatmap)."""
    def __init__(self, charge_hours, discharge_hours, power=500.0, name="Regla horària"):
        self.charge, self.discharge, self.power, self.name = set(charge_hours), set(discharge_hours), power, name
        self.bat = battery()
    def act(self, obs):
        h = obs["hour"]; soc = obs["soc_kwh"]
        if h in self.charge:
            return safe_action(self.power, soc, self.bat)
        if h in self.discharge:
            last_net = obs["noves_dades_5min"]["ultim_consum_kw"] - obs["noves_dades_5min"]["ultima_solar_kw"]
            return safe_action(-min(self.power, max(last_net, 0.0)), soc, self.bat)
        return 0.0


class MPCAgent:
    """Receding-horizon LP over the 24 h price/CO2 forecasts the challenge provides.

    forecast='naive': load/PV for the next 24 h = same hour yesterday (else last
    observation). forecast='perfect': true future load/PV (an upper bound used only
    to measure how much a better load forecast could be worth).
    """
    def __init__(self, forecast: str, scale_co2: float, scale_cost: float, truth: pd.DataFrame | None = None,
                 terminal_fraction: float = 0.5, name: str | None = None):
        self.forecast, self.truth = forecast, truth
        self.scale_co2, self.scale_cost = scale_co2, scale_cost
        self.terminal_fraction = terminal_fraction
        self.bat = battery(); self.t = 0; self.hist: list[tuple[int, float, float]] = []
        self.name = name or ("MPC 24 h · previsió " + ("perfecta" if forecast == "perfect" else "ingènua"))
    def _net_forecast(self, obs, n: int) -> tuple[np.ndarray, np.ndarray]:
        if self.forecast == "perfect" and self.truth is not None:
            sl = self.truth.iloc[self.t:self.t + n]
            return sl["load_kw"].to_numpy(float), sl["pv_kw"].to_numpy(float)
        last = obs["noves_dades_5min"]
        by_hour = {h: (l, p) for h, l, p in self.hist[-24:]}
        hours = [(obs["hour"] + k) % 24 for k in range(n)]
        load = np.array([by_hour.get(h, (last["ultim_consum_kw"], 0))[0] for h in hours])
        pv = np.array([by_hour.get(h, (0, last["ultima_solar_kw"]))[1] for h in hours])
        return load, pv
    def act(self, obs):
        if self.t > 0:
            last = obs["noves_dades_5min"]
            self.hist.append(((obs["hour"] - 1) % 24, last["ultim_consum_kw"], last["ultima_solar_kw"]))
        price = np.asarray(obs["preus_properes_24h"], float); co2 = np.asarray(obs["co2_properes_24h"], float)
        n = len(price)
        load, pv = self._net_forecast(obs, n)
        load, pv = load[:n], pv[:n]
        if len(load) < n:
            n = len(load); price, co2 = price[:n], co2[:n]
        w = weights(price, co2, ALPHA, self.scale_co2, self.scale_cost)
        tv = self.terminal_fraction * math.sqrt(self.bat["eta"]) * float(np.median(w))
        try:
            plan = lp_schedule(load, pv, price, co2, obs["soc_kwh"], bat=self.bat, scale_co2=self.scale_co2,
                               scale_cost=self.scale_cost, terminal_value=tv)
            action = float(plan[0])
        except RuntimeError:
            action = 0.0
        self.t += 1
        return safe_action(action, obs["soc_kwh"], self.bat)


class ScheduleAgent:
    """Replays a precomputed open-loop schedule (used for the perfect-information oracle)."""
    def __init__(self, schedule, name):
        self.schedule, self.name, self.t = np.asarray(schedule, float), name, 0
    def act(self, obs):
        a = float(self.schedule[self.t]) if self.t < len(self.schedule) else 0.0
        self.t += 1
        return a


def agent_path() -> Path | None:
    env = os.getenv("UPC_AGENT_PATH")
    candidates = [Path(env)] if env else [ROOT / "src" / "agent.py", STARTER / "agent.py"]
    for p in candidates:
        p = p if p.is_absolute() else ROOT / p
        if p.is_file():
            return p
    return None


def load_agent_factory(path: Path) -> Callable[[], Any]:
    """Import the team's BatteryAgent read-only; a fresh instance per week."""
    spec = importlib.util.spec_from_file_location("upc_team_agent_" + hashlib.md5(str(path).encode()).hexdigest()[:8], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.BatteryAgent


def _env_class():
    spec = importlib.util.spec_from_file_location("upc_official_environment", STARTER / "environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.CampusBatteryEnv


def run_official(agent, week: pd.DataFrame) -> dict:
    """Run any act(obs) policy inside the official environment (no leakage)."""
    Env = _env_class()
    env = Env(week[["Fecha", *COLS]].set_index("Fecha"))
    obs = env.reset(); done = False; errors = 0
    while not done:
        try:
            action = float(agent.act(obs))
            if not math.isfinite(action):
                raise ValueError("non-finite action")
        except Exception:
            errors += 1; action = 0.0
        obs, done = env.step(action)
    out = env.evaluate_results()
    raw = ALPHA * out["co2_estalviat_pct"] + (1 - ALPHA) * out["cost_estalviat_pct"]
    return {"score": out["score"], "score_statement_penalty": round(max(0.0, raw - 100 * STATEMENT_PENALTY * out["infraccions"]), 2),
            "co2_saved_pct": out["co2_estalviat_pct"], "cost_saved_pct": out["cost_estalviat_pct"],
            "co2_saved_kg": out["co2_estalviat_kg"], "cost_saved_eur": out["cost_estalviat_eur"],
            "co2_baseline_kg": out["co2_baseline_kg"], "cost_baseline_eur": out["cost_baseline_eur"],
            "infractions": out["infraccions"], "agent_errors": errors,
            "soc": [round(float(s), 1) for s in env.soc_series], "actions": [round(float(a), 1) for a in env.actions]}


# --------------------------------------------------------------------------
# Pseudo-test protocol
# --------------------------------------------------------------------------
def complete_days(df: pd.DataFrame) -> list[str]:
    ok = df.dropna(subset=COLS)
    sizes = ok.groupby("date")["hour"].nunique()
    return sorted(sizes[sizes == 24].index)


def pseudo_test_weeks(df: pd.DataFrame) -> list[dict]:
    """Per month, the latest 7 consecutive complete days available in train.

    The hidden test set is the last 7 days of every month, so the 7 days right
    before it are the closest seasonal proxy for each test week.
    """
    days = set(complete_days(df))
    weeks = []
    for (_, month), part in df.groupby([df["Fecha"].dt.year, df["Fecha"].dt.month]):
        dates = sorted({_dt.date.fromisoformat(d) for d in part["date"].unique()})
        end, first = dates[-1], dates[0]
        while end - _dt.timedelta(days=6) >= first:
            span = [(end - _dt.timedelta(days=k)).isoformat() for k in range(6, -1, -1)]
            if all(d in days for d in span):
                frame = df[df["date"].isin(span)].dropna(subset=COLS).sort_values("Fecha").reset_index(drop=True)
                weeks.append({"id": span[0], "label": f"{span[0]} → {span[-1]}", "month": int(month), "frame": frame})
                break
            end -= _dt.timedelta(days=1)
    return weeks


def pooled(results: list[dict]) -> dict:
    kg0 = sum(r["co2_baseline_kg"] for r in results); eur0 = sum(r["cost_baseline_eur"] for r in results)
    kg = sum(r["co2_saved_kg"] for r in results); eur = sum(r["cost_saved_eur"] for r in results)
    infr = sum(r["infractions"] for r in results)
    d_co2 = 100 * kg / kg0 if kg0 else 0.0; d_cost = 100 * eur / eur0 if eur0 else 0.0
    raw = ALPHA * d_co2 + (1 - ALPHA) * d_cost
    return {"score": round(max(0.0, raw - 100 * OFFICIAL_PENALTY * infr), 2),
            "score_statement_penalty": round(max(0.0, raw - 100 * STATEMENT_PENALTY * infr), 2),
            "co2_saved_pct": round(d_co2, 3), "cost_saved_pct": round(d_cost, 3),
            "co2_saved_t": round(kg / 1000, 3), "cost_saved_eur": round(eur, 2), "infractions": int(infr)}


def scales(weeks: list[dict]) -> tuple[float, float]:
    frames = pd.concat([w["frame"] for w in weeks])
    base = np.maximum(frames["load_kw"] - frames["pv_kw"], 0)
    return float(base @ frames["co2_kg_per_kwh"]), float(base @ frames["precio_eur_per_kwh"])


def train_scales(df: pd.DataFrame, weeks: list[dict]) -> tuple[float, float]:
    """Baseline CO2 and cost per evaluated hour, estimated from train days outside the weeks."""
    held = set().union(*[set(w["frame"]["date"]) for w in weeks]) if weeks else set()
    rest = df[~df["date"].isin(held)].dropna(subset=COLS)
    base = np.maximum(rest["load_kw"] - rest["pv_kw"], 0)
    hours = sum(len(w["frame"]) for w in weeks) or len(rest)
    return float(base @ rest["co2_kg_per_kwh"]) / len(rest) * hours, float(base @ rest["precio_eur_per_kwh"]) / len(rest) * hours


def signal_profile(df: pd.DataFrame) -> np.ndarray:
    clean = df.dropna(subset=["precio_eur_per_kwh", "co2_kg_per_kwh"])
    s = 0.7 * clean["co2_kg_per_kwh"] / clean["co2_kg_per_kwh"].mean() + 0.3 * clean["precio_eur_per_kwh"] / clean["precio_eur_per_kwh"].mean()
    return s.groupby(clean["hour"]).mean().reindex(range(24)).to_numpy(float)


def rule_windows(df: pd.DataFrame, k: int = 4) -> tuple[list[int], list[int]]:
    """Cheapest-k and dirtiest-k hours of the average 70/30 signal (charge before discharge)."""
    sig = signal_profile(df)
    order = np.argsort(sig)
    charge = sorted(int(h) for h in order[:k])
    later = [int(h) for h in order[::-1] if h > max(charge)]
    discharge = sorted(later[:k]) or sorted(int(h) for h in order[::-1][:k])
    return charge, discharge


# --------------------------------------------------------------------------
# Arena (background, cached)
# --------------------------------------------------------------------------
class Job:
    def __init__(self):
        self.lock = threading.Lock(); self.state = "idle"; self.result: dict | None = None
        self.error: str | None = None; self.progress = ""; self.started = 0.0
    def snapshot(self) -> dict:
        with self.lock:
            return {"status": self.state, "progress": self.progress, "error": self.error,
                    "elapsed_s": round(time.time() - self.started, 1) if self.started else 0}


ARENA = Job()
STUDIES = Job()


def _data_key(df: pd.DataFrame, extra: str = "") -> str:
    h = hashlib.sha1(pd.util.hash_pandas_object(df[["Fecha", *COLS]], index=False).values.tobytes())
    h.update(extra.encode())
    return h.hexdigest()[:16]


CACHE_DIR = HERE / ".cache"


def _cached(name: str, key: str, compute: Callable[[], dict]) -> dict:
    path = CACHE_DIR / f"{name}-{key}.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    result = compute()
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass
    return result


def compute_arena(df: pd.DataFrame, progress: Callable[[str], None] = lambda s: None) -> dict:
    weeks = pseudo_test_weeks(df)
    if not weeks:
        raise RuntimeError("No hi ha setmanes completes per fer el backtest")
    s_co2, s_cost = scales(weeks)          # oracle: exact pooled normalisation of the evaluated weeks
    t_co2, t_cost = train_scales(df, weeks)  # causal policies: hourly train averages, no peeking at test weeks
    held = set().union(*[set(w["frame"]["date"]) for w in weeks])
    charge, discharge = rule_windows(df[~df["date"].isin(held)])  # rule learnt outside the evaluated weeks
    apath = agent_path()
    policies: list[tuple[str, str, Callable[[dict], Any]]] = [
        ("idle", "Sense bateria", lambda w: IdleAgent()),
        ("rule", f"Regla horària (càrrega {charge[0]:02d}–{charge[-1]:02d} h, descàrrega {discharge[0]:02d}–{discharge[-1]:02d} h)",
         lambda w: WindowAgent(charge, discharge)),
    ]
    agent_note = None
    if apath:
        try:
            Factory = load_agent_factory(apath)
            label = "BatteryAgent · " + str(apath.relative_to(ROOT) if apath.is_relative_to(ROOT) else apath.name)
            policies.append(("team", label, lambda w: Factory()))
        except Exception as exc:  # the team's file may be mid-edit
            agent_note = f"No s'ha pogut importar {apath.name}: {type(exc).__name__}: {exc}"
    policies += [
        ("mpc_naive", "MPC 24 h · previsió ingènua de consum", lambda w: MPCAgent("naive", t_co2, t_cost)),
        ("mpc_perfect", "MPC 24 h · consum futur conegut", lambda w: MPCAgent("perfect", t_co2, t_cost, truth=w["frame"])),
        ("oracle", "Òptim setmanal · informació perfecta", lambda w: ScheduleAgent(
            lp_schedule(w["frame"]["load_kw"], w["frame"]["pv_kw"], w["frame"]["precio_eur_per_kwh"],
                        w["frame"]["co2_kg_per_kwh"], battery()["soc0"], scale_co2=s_co2, scale_cost=s_cost),
            "oracle")),
    ]
    out = []
    for key, label, make in policies:
        per_week = []
        t0 = time.time()
        for i, w in enumerate(weeks):
            progress(f"{label} · setmana {i + 1}/{len(weeks)}")
            r = run_official(make(w), w["frame"])
            r.update({"week": w["id"], "label": w["label"], "month": w["month"]})
            per_week.append(r)
        out.append({"key": key, "label": label, "pooled": pooled(per_week), "weeks": per_week,
                    "seconds": round(time.time() - t0, 1)})
    by = {p["key"]: p["pooled"]["score"] for p in out}
    regret = []
    if "oracle" in by and "mpc_perfect" in by:
        regret.append({"step": "Informació perfecta de tota la setmana → MPC de 24 h", "points": round(by["oracle"] - by["mpc_perfect"], 2),
                       "meaning": "Valor de veure més enllà de 24 h (miopia de l'horitzó)."})
    if "mpc_perfect" in by and "mpc_naive" in by:
        regret.append({"step": "Consum futur conegut → previsió ingènua", "points": round(by["mpc_perfect"] - by["mpc_naive"], 2),
                       "meaning": "Valor màxim que podria aportar un millor model de previsió de consum/solar."})
    if "mpc_naive" in by and "team" in by:
        regret.append({"step": "MPC ingènua → BatteryAgent", "points": round(by["mpc_naive"] - by["team"], 2),
                       "meaning": "Diferència atribuïble a l'estructura de la política de l'agent."})
    net_min = float(min((w["frame"]["load_kw"] - w["frame"]["pv_kw"]).min() for w in weeks))
    return {"weeks": [{"id": w["id"], "label": w["label"], "month": w["month"]} for w in weeks],
            "policies": out, "regret": regret, "rule": {"charge": charge, "discharge": discharge},
            "agent_path": str(apath) if apath else None, "agent_note": agent_note,
            "min_net_load_kw": round(net_min, 1),
            "protocol": "Per a cada mes, els últims 7 dies complets de train (just abans de la finestra oculta de test). "
                        "Cada setmana comença amb SOC = 50% i s'avalua amb starter_kit/environment.py.",
            "leakage_warning": "Si l'agent s'ha calibrat amb aquestes mateixes dates, el resultat és in-sample i optimista.",
            "penalty_note": "environment.py resta 0,2 punts per infracció; l'ENUNCIAT diu 1,00. Es mostren tots dos."}


def compute_studies(df: pd.DataFrame, progress: Callable[[str], None] = lambda s: None) -> dict:
    """Pareto frontier, sizing sensitivity and a daily value calendar (perfect information)."""
    weeks = pseudo_test_weeks(df)
    s_co2, s_cost = scales(weeks)
    pareto = []
    for alpha in [round(float(a), 1) for a in np.linspace(0, 1, 11)]:
        progress(f"Frontera de Pareto · α = {alpha}")
        res = []
        for w in weeks:
            f = w["frame"]
            plan = lp_schedule(f["load_kw"], f["pv_kw"], f["precio_eur_per_kwh"], f["co2_kg_per_kwh"], battery()["soc0"],
                               alpha=alpha, scale_co2=s_co2, scale_cost=s_cost)
            res.append(official_eval(f, plan))
        p = pooled(res)
        pareto.append({"alpha": alpha, "co2_saved_pct": p["co2_saved_pct"], "cost_saved_pct": p["cost_saved_pct"],
                       "score_70_30": p["score"]})
    sizing = []
    for cap in (500, 1000, 2000, 3000, 4000):
        for power in (250, 500, 750, 1000):
            progress(f"Dimensionat · {cap} kWh / {power} kW")
            bat = battery(cap, power)
            res = []
            for w in weeks:
                f = w["frame"]
                plan = lp_schedule(f["load_kw"], f["pv_kw"], f["precio_eur_per_kwh"], f["co2_kg_per_kwh"], bat["soc0"],
                                   bat=bat, scale_co2=s_co2, scale_cost=s_cost)
                res.append(official_eval(f, plan, bat=bat))
            p = pooled(res)
            sizing.append({"capacity_kwh": cap, "power_kw": power, "score": p["score"],
                           "co2_saved_t": p["co2_saved_t"], "cost_saved_eur": p["cost_saved_eur"]})
    calendar = []
    days = complete_days(df)
    for i, d in enumerate(days):
        if i % 25 == 0:
            progress(f"Calendari de valor · {i}/{len(days)} dies")
        f = df[df["date"] == d].sort_values("Fecha")
        bat = battery()
        plan = lp_schedule(f["load_kw"], f["pv_kw"], f["precio_eur_per_kwh"], f["co2_kg_per_kwh"], bat["soc0"],
                           soc_end_min=bat["soc0"])
        r = official_eval(f, plan)
        sig = 0.7 * f["co2_kg_per_kwh"] / f["co2_kg_per_kwh"].mean() + 0.3 * f["precio_eur_per_kwh"] / f["precio_eur_per_kwh"].mean()
        calendar.append({"date": d, "score": r["score"], "co2_saved_kg": r["co2_saved_kg"], "cost_saved_eur": r["cost_saved_eur"],
                         "spread": round(float(sig.max() - sig.min()), 3)})
    cal = pd.DataFrame(calendar)
    spread_corr = float(cal[["score", "spread"]].corr().iloc[0, 1]) if len(cal) > 3 else None
    return {"pareto": pareto, "sizing": sizing, "calendar": calendar,
            "calendar_spread_corr": round(spread_corr, 3) if spread_corr is not None else None,
            "weeks": len(weeks),
            "note": "Estudis amb informació perfecta (cota superior). Serveixen per comparar escenaris, no com a previsió del leaderboard."}


def start_background(df: pd.DataFrame) -> None:
    """Launch both studies once per dataset; results are cached on disk."""
    apath = agent_path()
    agent_stamp = f"{apath}:{apath.stat().st_mtime_ns}" if apath else "none"
    for job, name, fn, extra in ((ARENA, "arena", compute_arena, agent_stamp), (STUDIES, "studies", compute_studies, "v1")):
        with job.lock:
            if job.state in ("running", "ready"):
                continue
            job.state, job.started, job.error = "running", time.time(), None

        def work(job=job, name=name, fn=fn, extra=extra):
            def progress(msg):
                with job.lock:
                    job.progress = msg
            try:
                result = _cached(name, _data_key(df, extra), lambda: fn(df, progress))
                with job.lock:
                    job.result, job.state, job.progress = result, "ready", "Complet"
            except Exception as exc:
                traceback.print_exc()
                with job.lock:
                    job.state, job.error = "error", f"{type(exc).__name__}: {exc}"
        threading.Thread(target=work, daemon=True, name=f"upc-{name}").start()


def reset_arena() -> None:
    """Force a recomputation (e.g. after the team edits agent.py)."""
    with ARENA.lock:
        ARENA.state, ARENA.result = "idle", None


# --------------------------------------------------------------------------
# Decision Lab: one day, user windows vs cyclic optimum
# --------------------------------------------------------------------------
def _window(start: int, end: int) -> list[int]:
    """Hours in [start, end] inclusive, wrapping around midnight."""
    start, end = int(start) % 24, int(end) % 24
    return list(range(start, end + 1)) if start <= end else list(range(start, 24)) + list(range(0, end + 1))


def simulate_day(day: pd.DataFrame, charge: tuple[int, int], discharge: tuple[int, int], power: float) -> dict:
    bat = battery()
    ch, dis = set(_window(*charge)), set(_window(*discharge))
    overlap = sorted(ch & dis)
    ch -= dis  # discharge wins on overlap; reported to the user
    load = day["load_kw"].to_numpy(float); pv = day["pv_kw"].to_numpy(float)
    soc = bat["soc0"]; eta = math.sqrt(bat["eta"]); plan = []
    for t, h in enumerate(day["hour"].astype(int)):
        if h in ch:
            p = safe_action(power, soc, bat)
        elif h in dis:
            # Closed cycle: only energy charged today is discharged (SOC never below SOC0).
            floor_kw = max(0.0, (soc - bat["soc0"]) * eta - POWER_MARGIN)
            p = safe_action(-min(power, max(load[t] - pv[t], 0.0), floor_kw), soc, bat)
        else:
            p = 0.0
        soc += p * eta if p >= 0 else p / eta
        plan.append(p)
    manual = official_eval(day, plan)
    opt_plan = lp_schedule(load, pv, day["precio_eur_per_kwh"], day["co2_kg_per_kwh"], bat["soc0"], soc_end_min=bat["soc0"])
    optimal = official_eval(day, opt_plan)
    base = np.maximum(load - pv, 0)
    w = weights(day["precio_eur_per_kwh"].to_numpy(float), day["co2_kg_per_kwh"].to_numpy(float), ALPHA,
                float(base @ day["co2_kg_per_kwh"]), float(base @ day["precio_eur_per_kwh"]))
    # w[t] = score points gained by avoiding 1 kWh of import at hour t (that day's normalisation).
    return {"manual": manual, "optimal": optimal, "manual_actions": manual["_applied"], "optimal_actions": optimal["_applied"],
            "weights": w, "overlap_hours": overlap, "charge_hours": sorted(ch), "discharge_hours": sorted(dis)}

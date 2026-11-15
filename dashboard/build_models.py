"""
dashboard/build_models.py
Precomputes everything the "model comparison" pages of the dashboard need, so the app itself
never has to run an agent:

    python dashboard/build_models.py            # arena export + oracle + sizing + traces (a few minutes)
    python dashboard/build_models.py --no-traces # skip the hour-by-hour agent runs

Outputs (dashboard/model_data/):
    arena.json    every strategy of dificil/arena/results/store*.json: score, CO2, cost, per-week scores,
                  plus the perfect-foresight optimum per fold and per week
    traces.parquet hour-by-hour battery power / SOC / grid import of a few agents and of the optimum (fold 1)
    runs.json     summary of those runs (score, kg CO2, EUR, ms per step measured alone, one thread)
    sizing.json   perfect-foresight score for a grid of battery sizes, with and without the x12 solar plant

Single process, one thread (CLAUDE.md: never two heavy jobs at once on this 4-core machine).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "model_data"
sys.path.insert(0, str(ROOT / "dificil"))
sys.path.insert(0, str(ROOT / "estandard"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dificil" / "arena"))

FOLDS = (1, 2, 3)
# Agents traced hour by hour on fold 1 (arena strategy name -> what it stands for)
TRACED = {
    "v9": "v9",
    "v12": "v12",
    "stoch_s12": "nucli de la v14 (MPC estocàstic, 12 escenaris)",
    "stoch8_nn_pre": "nucli de la v13 (MPC estocàstic + CO₂/preu neuronal)",
}
ETA = 0.9 ** 0.5


def week_ids(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.to_series().diff() != pd.Timedelta("1h")).cumsum().values


def simulate(actions_or_agent, val: pd.DataFrame):
    """Runs an agent (or a fixed action vector) through the official physics. Returns env, ms per step."""
    from henv import HardEnv
    env = HardEnv(val.reset_index())
    obs, done, t, t0 = env.reset(), False, 0, time.perf_counter()
    while not done:
        if hasattr(actions_or_agent, "act"):
            a = actions_or_agent.act(obs)
            a = float(a) if np.isfinite(a) else 0.0
        else:
            # The LP sits exactly on the SOC limits; floating-point noise there would count as an infraction,
            # so the fixed plan is clamped to what is physically feasible this hour (changes nothing else).
            soc, cap = obs["soc_kwh"], env.cap
            a = float(np.clip(actions_or_agent[t], -max(soc - 1e-6, 0.0) * ETA, max(cap - soc - 1e-6, 0.0) / ETA))
        obs, done = env.step(a)
        t += 1
    return env, 1000 * (time.perf_counter() - t0) / max(t, 1)


def env_trace(env, val: pd.DataFrame) -> pd.DataFrame:
    soc = np.asarray(env.soc_series, float)
    prev = np.r_[env.soc_init, soc[:-1]]
    p = np.where(soc >= prev, (soc - prev) / ETA, (soc - prev) * ETA)
    return pd.DataFrame({"bateria_kw": p, "soc_kwh": soc, "red_kw": np.asarray(env.grid, float),
                         "infr": np.asarray(env.infraccions_per_step, int)}, index=val.index)


def env_summary(env, val: pd.DataFrame) -> dict:
    r = env.evaluate_results()
    gid = week_ids(val.index)
    weeks = {str(val.index[gid == g][0].date()): env.evaluate_results(mask=(gid == g))["score"] for g in np.unique(gid)}
    return {"score": r["score"], "co2": r["co2_estalviat_pct"], "cost": r["cost_estalviat_pct"],
            "infr": r["infraccions"], "co2_kg": r["co2_estalviat_kg"], "cost_eur": r["cost_estalviat_eur"],
            "co2_base_kg": r["co2_baseline_kg"], "cost_base_eur": r["cost_baseline_eur"], "weeks": weeks}


def export_arena(df) -> dict:
    from folds import split_fold
    from horacle import oracle
    res = ROOT / "dificil" / "arena" / "results"
    rows: dict[str, dict] = {}
    for f in ("store_kaggle.json", "store.json"):            # the local store wins when both have a run
        if not (res / f).exists():
            continue
        for key, r in json.loads((res / f).read_text()).items():
            if "score" not in r:
                continue
            rows.setdefault(r["name"], {})[str(r["fold"])] = {
                "score": r["score"], "co2": r["co2"], "cost": r["cost"], "infr": r["infr"], "weeks": r["weeks"]}
    rows = {n: d for n, d in rows.items() if all(str(k) in d for k in FOLDS)}
    orc = {}
    for k in FOLDS:
        val = split_fold(df, k)[1]
        _, P = oracle(val)
        env, _ = simulate(P, val)
        s = env_summary(env, val)
        orc[str(k)] = {"score": s["score"], "co2": s["co2"], "cost": s["cost"], "weeks": s["weeks"], "hours": len(val)}
    return {"strategies": rows, "oracle": orc}


def sizing(df) -> list[dict]:
    """Perfect-foresight value of the battery for several sizes (power = capacity / 4, as in both modes)."""
    from horacle import oracle
    out = []
    days = len(df) / 24
    for pv_scale in (1, 12):
        d = df.copy()
        d["pv_kw"] = d["pv_kw"] / 12 * pv_scale
        net = (d.load_kw - d.pv_kw).clip(lower=0)
        co2_base, cost_base = float((net * d.co2_kg_per_kwh).sum()), float((net * d.precio_eur_per_kwh).sum())
        for cap in (1000, 2000, 4000, 6000, 8000, 12000):
            r, _ = oracle(d, cap=float(cap), pmax=cap / 4, soc0=cap / 2)
            out.append({"pv_scale": pv_scale, "cap_kwh": cap, "score": r["score"], "co2_pct": r["co2"],
                        "cost_pct": r["cost"], "co2_t_year": co2_base * r["co2"] / 100 / 1000 * 365 / days,
                        "eur_year": cost_base * r["cost"] / 100 * 365 / days,
                        "co2_base_t_year": co2_base / 1000 * 365 / days, "eur_base_year": cost_base * 365 / days})
            print(f"  sizing pv x{pv_scale} {cap} kWh -> {r['score']:.2f}", flush=True)
    return out


def traces(df):
    import arena
    from folds import split_fold
    from horacle import oracle
    val = split_fold(df, 1)[1]
    frames, runs = [], {}
    _, P = oracle(val)
    env, _ = simulate(P, val)
    frames.append(env_trace(env, val).assign(model="optimum"))
    runs["optimum"] = {**env_summary(env, val), "label": "Previsió perfecta (cota superior)", "ms": None}
    for name, label in TRACED.items():
        try:
            agent = arena.load_agent(name, {"models": "models_f17.pkl"})
            env, ms = simulate(agent, val)
        except Exception as e:                               # an arena strategy may need files that are not here
            print(f"  {name}: skipped ({type(e).__name__}: {e})", flush=True)
            continue
        frames.append(env_trace(env, val).assign(model=name))
        runs[name] = {**env_summary(env, val), "label": label, "ms": round(ms, 1)}
        print(f"  {name}: {runs[name]['score']:.2f}  {ms:.1f} ms/step", flush=True)
    ctx = val[["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]]
    tr = pd.concat(frames).join(ctx)
    tr.index.name = "Fecha"
    return tr.reset_index(), runs


if __name__ == "__main__":
    from hdata import load_hard
    OUT.mkdir(exist_ok=True)
    data = load_hard()
    print("arena + optimum per fold", flush=True)
    (OUT / "arena.json").write_text(json.dumps(export_arena(data)))
    if "--optimum" not in sys.argv:
        print("battery sizing", flush=True)
        (OUT / "sizing.json").write_text(json.dumps(sizing(data)))
    if "--optimum" in sys.argv:                              # refresh only the optimum rows of existing traces
        from folds import split_fold
        from horacle import oracle
        val = split_fold(data, 1)[1]
        env, _ = simulate(oracle(val)[1], val)
        old = pd.read_parquet(OUT / "traces.parquet")
        ctx = val[["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]]
        new = env_trace(env, val).assign(model="optimum").join(ctx)
        new.index.name = "Fecha"
        pd.concat([new.reset_index(), old[old["model"] != "optimum"]], ignore_index=True).to_parquet(OUT / "traces.parquet")
        runs = json.loads((OUT / "runs.json").read_text(encoding="utf-8"))
        runs["optimum"] = {**env_summary(env, val), "label": "Previsió perfecta (cota superior)", "ms": None}
        (OUT / "runs.json").write_text(json.dumps(runs, ensure_ascii=False), encoding="utf-8")
    elif "--no-traces" not in sys.argv:
        print("hour-by-hour traces (fold 1)", flush=True)
        tr, runs = traces(data)
        tr.to_parquet(OUT / "traces.parquet")
        (OUT / "runs.json").write_text(json.dumps(runs, ensure_ascii=False), encoding="utf-8")
    print("done ->", OUT)

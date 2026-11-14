"""
Avaluació d'agents amb el simulador OFICIAL (starter_kit/environment.py), en una sola
simulació contínua com al servidor. Calcula també la puntuació sobre meitats "pública"/"privada"
de cada setmana i el gap respecte a l'oracle.
"""
from __future__ import annotations
import ast, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter_kit"))
from environment import CampusBatteryEnv  # noqa: E402
sys.path.pop(0)
sys.path.insert(0, str(ROOT / "estandard"))
from oracle import oracle_actions, score_from_actions  # noqa: E402

AGENTS = ROOT / "estandard" / "agents"

# mateixa llista que check_submission.py
FORBIDDEN = {"os", "sys", "shutil", "subprocess", "socket", "urllib", "requests", "http", "ftplib",
             "paramiko", "telnetlib", "pty", "posix", "ctypes", "importlib", "inspect", "gc",
             "builtins", "fcntl", "signal", "threading", "multiprocessing"}


def ast_ok(folder: Path) -> str | None:
    for f in folder.glob("*.py"):
        if f.name.startswith("train"):
            continue  # scripts d'entrenament no es lliuren
        for n in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                names = [n.module]
            for m in names:
                if m.split(".")[0] in FORBIDDEN:
                    return f"{f.name}: import prohibit '{m}'"
    return None


def load_agent(folder: Path, **kw):
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(f"ag_{folder.name}_{time.time_ns()}", folder / "agent.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod.BatteryAgent(**kw) if kw else mod.BatteryAgent()


def public_mask(df: pd.DataFrame) -> np.ndarray:
    """Suposició: el 50% públic són les primeres 84 h de cada setmana (no documentat)."""
    gid = (df.index.to_series().diff() != pd.Timedelta("1h")).cumsum().values
    pos = pd.Series(1, index=df.index).groupby(gid).cumsum().values - 1
    return (pos % 168) < 84


def evaluate(agent, df: pd.DataFrame) -> dict:
    env = CampusBatteryEnv(df.reset_index()[["Fecha", "load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]])
    obs = env.reset(); done = False; t0 = time.time()
    while not done:
        obs, done = env.step(agent.act(obs))
    dt = time.time() - t0
    res = env.evaluate_results()
    # l'enunciat web diu -1 punt per infracció (l'environment local resta 0.2)
    res["score_web"] = round(max(0.0, res["co2_estalviat_pct"] * 0.7 + res["cost_estalviat_pct"] * 0.3
                                 - res["infraccions"]), 3)
    m = public_mask(df)
    res["score_public"] = env.evaluate_results(mask=m)["score"]
    res["score_private"] = env.evaluate_results(mask=~m)["score"]
    res["runtime_s"] = round(dt, 2)
    res["ms_per_step"] = round(1000 * dt / len(df), 3)
    res["cicles_equivalents"] = round(float(np.sum(np.clip(env.actions, 0, None))) / 2000.0, 1)
    return res


def oracle_score(df: pd.DataFrame) -> float:
    P, _ = oracle_actions(df)
    return score_from_actions(df, P)["score"]

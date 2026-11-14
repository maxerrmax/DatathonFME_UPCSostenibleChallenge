"""Cerca de paràmetres de l'agent (en paral·lel) sobre les setmanes de validació."""
import sys, json
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eval_agent_hard import load_agent_cls
from src.data import load_hourly, validation_weeks
from src.hard_env import run

VAL = validation_weeks(load_hourly(pv_scale=12))
MODEL_DIR = sys.argv[1]

def one(params):
    r = run(load_agent_cls()(model_dir=MODEL_DIR, params=params), VAL)
    return params, r["score"], r["co2_estalviat_pct"], r["cost_estalviat_pct"], r["infraccions"]

if __name__ == "__main__":
    grid = [json.loads(g) for g in sys.argv[2:]]
    with Pool(4) as p:
        for params, s, c, k, i in sorted(p.map(one, grid), key=lambda x: -x[1]):
            print(f"{s:6.2f}  CO2={c:6.2f}  cost={k:6.2f}  infr={i}  {json.dumps(params)}")

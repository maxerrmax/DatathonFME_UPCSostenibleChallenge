"""Avalua l'agent del Mode Difícil a validació (models entrenats només amb 'fit')."""
import sys, json, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pipeline"))
from hdata import load_hard; from data import split_val
from henv import run
from horacle import oracle


def load_agent(params):
    sys.path.insert(0, str(HERE / "agent_hard"))
    spec = importlib.util.spec_from_file_location("ah", HERE / "agent_hard" / "agent.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.path.pop(0)
    return m.BatteryAgent(params)


if __name__ == "__main__":
    df = load_hard(); fit, val = split_val(df)
    print("oracle val", oracle(val)[0])
    grid = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [{}]
    for g in grid:
        prm = {"models": "models_fit.pkl", **g}
        r = run(load_agent(prm), val)
        print(g, {k: r[k] for k in ["score", "co2_estalviat_pct", "cost_estalviat_pct", "infraccions", "ms_per_step"]}, flush=True)

"""
Avalua un agent del Mode Difícil "tal com es lliura" (carpeta amb agent.py, feats.py, parametres.json i el .pkl)
sobre la validació de l'equip (els darrers 7 dies de cada bloc mensual del train).

Ús:
    python scripts/eval_agent_dir.py <carpeta_agent> ['{"param": valor}' ...]

Per comparar en igualdat de condicions, el .pkl ha de ser el de models `fit` (entrenats sense la validació):
    python dificil/train_v7.py fit            -> dificil/agent_hard/models_fit7.pkl
    mkdir /tmp/v9fit && cp dificil/agent_hard/{agent.py,feats.py,parametres.json} /tmp/v9fit/
    cp dificil/agent_hard/models_fit7.pkl /tmp/v9fit/models_all7.pkl
    python scripts/eval_agent_dir.py /tmp/v9fit        # referència v9: 20,52
Amb diversos jocs de paràmetres s'executen en paral·lel (1 fil per procés).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dificil")); sys.path.insert(0, str(ROOT / "estandard"))

D = None
VAL = None


def _init(d):
    global D, VAL
    D = d
    sys.path.insert(0, d)
    from hdata import load_hard
    from data import split_val
    VAL = split_val(load_hard())[1]


def one(params):
    from henv import run
    spec = importlib.util.spec_from_file_location("agent_eval", os.path.join(D, "agent.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    r = run(m.BatteryAgent(params) if params else m.BatteryAgent(), VAL)
    return params, r["score"], r["co2_estalviat_pct"], r["cost_estalviat_pct"], r["infraccions"], r["ms_per_step"]


if __name__ == "__main__":
    d = os.path.abspath(sys.argv[1])
    grid = [json.loads(a) for a in sys.argv[2:]] or [{}]
    with Pool(min(len(grid), os.cpu_count() or 1), initializer=_init, initargs=(d,)) as p:
        for prm, s, c, k, i, ms in sorted(p.map(one, grid), key=lambda x: -x[1]):
            print(f"{s:6.2f}  CO2={c:6.2f}  cost={k:6.2f}  infr={i}  {ms} ms/pas  {json.dumps(prm)}")

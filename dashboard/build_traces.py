"""
dashboard/build_traces.py
Executa TOTES les estratègies de l'arena (les de `model_data/arena.json`) hora a hora sobre el plec 1 de validació i
les desa perquè la pàgina «Cara a cara» pugui comparar-ne qualsevol parella.

    python dashboard/build_traces.py                # les que encara no s'han calculat (reprèn on es va deixar)
    python dashboard/build_traces.py --all          # recalcula totes
    python dashboard/build_traces.py --procs 4      # nombre de processos (un fil per procés, com mana CLAUDE.md)

Cada estratègia es desa a `model_data/_cache/<nom>.pkl` (clau = hash del codi de l'arena); al final es fusionen a
`traces.parquet` i `runs.json`. Si una estratègia falla (li falten fitxers), s'omet i es diu.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_models as bm  # noqa: E402  (posa els camins de dificil/ i scripts/ a sys.path)

OUT = bm.OUT
CACHE = OUT / "_cache"


def _code_hash(name: str) -> str:
    import arena
    return arena.code_hash(name)


def work(name: str):
    import arena
    from folds import split_fold
    from hdata import load_hard
    try:
        val = split_fold(load_hard(), 1)[1]
        agent = arena.load_agent(name, {"models": "models_f17.pkl"})
        env, ms = bm.simulate(agent, val)
        res = {"hash": _code_hash(name), "trace": bm.env_trace(env, val), "summary": bm.env_summary(env, val),
               "ms": round(ms, 1)}
        CACHE.mkdir(exist_ok=True)
        (CACHE / f"{name}.pkl").write_bytes(pickle.dumps(res))
        return name, res["summary"]["score"], None
    except Exception as e:                                   # noqa: BLE001
        return name, None, f"{type(e).__name__}: {e}"


def merge(names: list[str]) -> None:
    from folds import split_fold
    from hdata import load_hard
    from horacle import oracle
    val = split_fold(load_hard(), 1)[1]
    env, _ = bm.simulate(oracle(val)[1], val)
    frames = [bm.env_trace(env, val).assign(model="optimum")]
    runs = {"optimum": {**bm.env_summary(env, val), "label": "Previsió perfecta (cota superior)", "ms": None}}
    for n in names:
        f = CACHE / f"{n}.pkl"
        if not f.exists():
            continue
        r = pickle.loads(f.read_bytes())
        frames.append(r["trace"].assign(model=n))
        runs[n] = {**r["summary"], "label": n, "ms": r["ms"]}
    ctx = val[["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]]
    tr = pd.concat(frames).join(ctx)
    tr.index.name = "Fecha"
    tr.reset_index().to_parquet(OUT / "traces.parquet")
    (OUT / "runs.json").write_text(json.dumps(runs, ensure_ascii=False), encoding="utf-8")
    print(f"merged {len(runs) - 1} agents + optimum")


if __name__ == "__main__":
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
    arena_names = sorted(json.loads((OUT / "arena.json").read_text())["strategies"])
    todo = []
    for n in arena_names:
        f = CACHE / f"{n}.pkl"
        if "--all" in sys.argv or not f.exists() or pickle.loads(f.read_bytes()).get("hash") != _code_hash(n):
            todo.append(n)
    print(f"{len(todo)} per calcular de {len(arena_names)}", flush=True)
    with Pool(procs) as p:
        for n, s, err in p.imap_unordered(work, todo):
            print(f"  {n}: {'ERROR ' + err if err else f'{s:.2f}'}", flush=True)
    merge(arena_names)

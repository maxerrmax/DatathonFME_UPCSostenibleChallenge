"""
Torneig d'estratègies del Mode Difícil sobre validació multi-plec.

    python scripts/arena.py                       # totes les estratègies de dificil/arena/, plecs 1-3
    python scripts/arena.py v12 v9 my_strat       # només aquestes
    python scripts/arena.py my_strat --smoke 72   # prova ràpida: 72 h del plec 1, un sol procés
    python scripts/arena.py --procs 8 --folds 1,2

Cada estratègia: dificil/arena/<nom>/agent.py amb BatteryAgent(params). El plec k fa servir els models
dificil/agent_hard/models_f<k>7.pkl (entrenats amb `python dificil/train_v7.py f<k>`, sense aquella setmana).
Els resultats es desen a dificil/arena/results/store.json (clau = nom, plec i hash del codi) i no es recalculen.
Paral·lelisme: 1 fil per procés (CLAUDE.md).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "dificil" / "arena"
RES = ARENA / "results"
sys.path.insert(0, str(ROOT / "dificil")); sys.path.insert(0, str(ROOT / "estandard")); sys.path.insert(0, str(ARENA))

REF = "v12"
_DF = None


def strategies():
    return sorted(p.parent.name for p in ARENA.glob("*/agent.py") if not p.parent.name.startswith(("_", ".")))


def code_hash(name):
    h = hashlib.sha1()
    files = sorted((ARENA / name).glob("*.py")) + [ARENA / "common.py", ROOT / "dificil/agent_hard/agent.py",
                                                    ROOT / "dificil/agent_hard/feats.py",
                                                    ROOT / "dificil/agent_hard/parametres.json"]
    src = (ARENA / name / "agent.py").read_text(encoding="utf-8")
    if "combo(" in src or "load_strategy(" in src:   # combinacions: el codi de les estratègies que encadena
        files += [f for f in sorted(ARENA.glob("*/*.py")) if f.parent.name in src]
    files += sorted((ARENA / name).glob("*.npz"))
    for dep in ("nn_pre",):                          # pesos d'estratègies encadenades
        if dep in src and dep != name:
            files += sorted((ARENA / dep).glob("*.npz"))
    if name.startswith("nn_") or "nn_" in src:                       # pesos de les xarxes (dificil/arena/nn_co2/*.npz)
        files += sorted((ARENA / "nn_co2").glob("*.npz"))
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def fold_data(k, smoke=0):
    global _DF
    from hdata import load_hard
    from folds import split_fold
    if _DF is None:
        _DF = load_hard()
    val = split_fold(_DF, k)[1]
    return val.iloc[:smoke] if smoke else val


def load_agent(name, params):
    spec = importlib.util.spec_from_file_location(f"arena_{name}", ARENA / name / "agent.py")
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ARENA / name))
    try:
        spec.loader.exec_module(m)
    finally:
        sys.path.pop(0)
    return m.BatteryAgent(params)


def run_one(job):
    name, k, extra, smoke = job
    from henv import HardEnv
    try:
        val = fold_data(k, smoke)
        agent = load_agent(name, {"models": f"models_f{k}7.pkl", **extra})
        env = HardEnv(val.reset_index()); o = env.reset(); d = False; t0 = time.time(); n = 0
        while not d:
            a = agent.act(o)
            if not np.isfinite(a):
                a = 0.0
            o, d = env.step(a); n += 1
        ms = 1000 * (time.time() - t0) / max(n, 1)
        tot = env.evaluate_results()
        gid = (val.index.to_series().diff() != pd.Timedelta("1h")).cumsum().values
        weeks = {str(val.index[gid == g][0].date()): env.evaluate_results(mask=(gid == g))["score"] for g in np.unique(gid)}
        return {"name": name, "fold": k, "score": tot["score"], "co2": tot["co2_estalviat_pct"],
                "cost": tot["cost_estalviat_pct"], "infr": tot["infraccions"], "ms": round(ms, 2), "weeks": weeks}
    except Exception:
        return {"name": name, "fold": k, "error": traceback.format_exc()[-2000:]}


def oracle_scores(folds):
    from horacle import oracle
    out = {}
    for k in folds:
        out[k] = oracle(fold_data(k))[0]["score"]
    return out


def report(cache, names, folds):
    rows = []
    ref = {k: cache.get(f"{REF}|{k}") for k in folds}
    for n in names:
        rs = [cache.get(f"{n}|{k}") for k in folds]
        if any(r is None or "error" in r for r in rs):
            err = next((r["error"].strip().splitlines()[-1] for r in rs if r and "error" in r), "pendent")
            rows.append((n, None, err)); continue
        sc = [r["score"] for r in rs]
        diffs, wins = [], 0
        for r, k in zip(rs, folds):
            if ref[k] and "weeks" in ref[k]:
                for w, s in r["weeks"].items():
                    if w in ref[k]["weeks"]:
                        dlt = s - ref[k]["weeks"][w]; diffs.append(dlt); wins += dlt > 1e-9
        rows.append((n, dict(mean=np.mean(sc), folds=sc, worst=min(np.array(sc) - np.array([ref[k]["score"] for k in folds]))
                             if all(ref.values()) else float("nan"),
                             co2=np.mean([r["co2"] for r in rs]), cost=np.mean([r["cost"] for r in rs]),
                             infr=sum(r["infr"] for r in rs), ms=max(r["ms"] for r in rs),
                             dmean=np.mean(diffs) if diffs else float("nan"),
                             tstat=(np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 2 and np.std(diffs) > 0 else float("nan")),
                             win=f"{wins}/{len(diffs)}"), None))
    ok = sorted([r for r in rows if r[1]], key=lambda r: -r[1]["mean"])
    hdr = f"{'estratègia':<22}{'mitjana':>8} " + " ".join(f"{'plec'+str(k):>7}" for k in folds) + \
          f"{'Δ v12':>8}{'pitjor Δ':>9}{'t':>6}{'setm>v12':>10}{'CO2%':>7}{'cost%':>7}{'infr':>5}{'ms/pas':>8}"
    lines = [hdr, "-" * len(hdr)]
    for n, d, _ in ok:
        lines.append(f"{n:<22}{d['mean']:8.3f} " + " ".join(f"{s:7.2f}" for s in d["folds"]) +
                     f"{d['dmean']:+8.3f}{d['worst']:+9.3f}{d['tstat']:6.1f}{d['win']:>10}{d['co2']:7.2f}{d['cost']:7.2f}{d['infr']:5d}{d['ms']:8.1f}")
    for n, _, e in rows:
        if _ is None:
            lines.append(f"{n:<22} ERROR/PENDENT: {e}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--folds", default="1,2,3")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--smoke", type=int, default=0, help="hores del plec 1 per a una prova ràpida (sense cache)")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--store", default="store.json", help="fitxer de resultats dins de dificil/arena/results/")
    ap.add_argument("--report", action="store_true", help="només l'informe, sense executar res")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    names = a.names or strategies()
    extra = json.loads(a.params)
    if a.smoke:
        for n in names:
            r = run_one((n, 1, extra, a.smoke))
            print(json.dumps({k: v for k, v in r.items() if k != "weeks"}, ensure_ascii=False))
        sys.exit(0)
    folds = [int(x) for x in a.folds.split(",")]
    RES.mkdir(parents=True, exist_ok=True)
    keyed = {}
    for n in set(names) | {REF}:
        hsh = code_hash(n) + (hashlib.sha1(a.params.encode()).hexdigest()[:6] if extra else "")
        for k in folds:
            keyed[(n, k)] = hsh
    store = {}
    SF = RES / a.store
    if SF.exists():
        store = json.loads(SF.read_text())
    todo = [] if a.report else [(n, k, extra, 0) for (n, k), h in keyed.items() if a.rerun or store.get(f"{n}|{k}|{h}") is None]
    print(f"{len(todo)} execucions pendents amb {min(a.procs, max(1, len(todo)))} processos", flush=True)
    if todo:
        with Pool(min(a.procs, len(todo))) as p:
            for r in p.imap_unordered(run_one, todo):
                key = f"{r['name']}|{r['fold']}|{keyed[(r['name'], r['fold'])]}"
                store[key] = r
                SF.write_text(json.dumps(store))
                msg = f"  {r['name']:<22} plec {r['fold']}: " + (f"{r['score']:.3f} ({r['ms']} ms/pas)" if "score" in r else "ERROR")
                print(msg, flush=True)
    view = {f"{n}|{k}": store.get(f"{n}|{k}|{h}") for (n, k), h in keyed.items()}
    orc = oracle_scores(folds)
    out = report(view, sorted(set(names) | {REF}), folds)
    out += "\n\nÒptim (previsió perfecta) per plec: " + "  ".join(f"plec{k} {v:.2f}" for k, v in orc.items())
    print(out)
    (RES / SF.name.replace("store", "leaderboard").replace(".json", ".txt")).write_text(out, encoding="utf-8")

"""
Pipeline Mode Difícil, en dues etapes (tot avaluat a 'validació', models entrenats només amb 'fit'):

  Etapa 1 — previsió: entrena cada família de models (preu, CO2, consum, solar), simula la validació
            gravant les previsions de cada hora i en mesura l'error (MAE) i el score amb la decisió per defecte.
  Etapa 2 — decisió: reutilitza les previsions gravades i prova configuracions de l'MPC en paral·lel.

Ús:  python dificil/hpipe.py stage1 [familia ...]
     python dificil/hpipe.py stage2 familia1 [familia2 ...]
"""
import sys, json, time, itertools, pickle
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import Parallel, delayed

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "agent_hard")); sys.path.insert(0, str(HERE.parent / "estandard"))
from hdata import load_hard
from data import split_val
from henv import run
from horacle import oracle
from heval import load_agent
from train_forecast import build, epoch_hours, COLS
from feats import AvgModel

CACHE, RES = HERE / "cache", HERE / "results"
KEYS = ["price", "co2", "load", "pv"]


def lgbm(**kw):
    base = dict(n_estimators=500, learning_rate=0.04, num_leaves=63, min_child_samples=20,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=2)
    base.update(kw); return lambda: lgb.LGBMRegressor(**base)


FAMILIES = {
    "lgbm_base": lgbm(),
    "lgbm_petit": lgbm(n_estimators=300, num_leaves=15, min_child_samples=60),
    "lgbm_gran": lgbm(n_estimators=1500, learning_rate=0.02, num_leaves=127, min_child_samples=10),
    "xgb": lambda: xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.9, n_jobs=2, tree_method="hist"),
    "hgb": lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63),
}


def fit_family(name, D):
    if name == "naive":
        return {}
    if name == "ensemble":
        parts = [fit_family(n, D) for n in ("lgbm_base", "xgb", "hgb")]
        conv = lambda m: lgb.Booster(model_str=m) if isinstance(m, str) else m
        return {k: AvgModel([conv(p[k]) for p in parts]) for k in parts[0]}
    mk = FAMILIES[name]; out = {}
    for k, (X, y) in D.items():
        lag = X[:, 5]; ok = np.isfinite(lag)
        m1 = mk(); m1.fit(X, y)
        m2 = mk(); m2.fit(X[ok], (y - lag)[ok])
        for nm, m in ((k + "_dir", m1), (k + "_res", m2)):
            out[nm] = m.booster_.model_to_string() if hasattr(m, "booster_") else m
    return out


def truth_tables(val):
    eh = epoch_hours(val.index)
    return {k: dict(zip(eh, val[COLS[k]].values)) for k in KEYS}


def forecast_mae(table, truth):
    errs = {k: [] for k in KEYS}
    for t, arrs in table.items():
        for i, k in enumerate(KEYS):
            for h, v in enumerate(arrs[i]):
                y = truth[k].get(t + h)
                if y is not None and (k in ("load", "pv") or h >= 3):
                    errs[k].append(abs(v - y))
    return {k: round(float(np.mean(v)), 4) for k, v in errs.items()}


def stage1(names):
    df = load_hard(); fit, val = split_val(df)
    from train_forecast import make_cs
    from feats import set_cs
    CS = make_cs(fit); set_cs(CS)
    D = build(fit); truth = truth_tables(val)
    rows = []
    for name in names:
        t0 = time.time()
        rec = {}
        if name == "perfecte":
            prm = {"_models": {}, "_cheat": (truth, KEYS), "_record": rec}
        else:
            prm = {"_models": {**fit_family(name, D), "pv_cs": CS}, "_record": rec}
        t_fit = time.time() - t0
        r = run(load_agent(prm), val)
        pickle.dump(rec, open(CACHE / f"fc_{name}.pkl", "wb"))
        row = {"familia": name, "score": r["score"], "co2_%": r["co2_estalviat_pct"], "cost_%": r["cost_estalviat_pct"],
               "infr": r["infraccions"], "ms_pas": r["ms_per_step"], "entrenament_s": round(t_fit, 1),
               **{f"MAE_{k}": v for k, v in forecast_mae(rec, truth).items()}}
        print(json.dumps(row), flush=True); rows.append(row)
    out = RES / "etapa1_previsio.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    allr = pd.concat([old[~old.familia.isin(names)] if len(old) else old, pd.DataFrame(rows)]).sort_values("score", ascending=False)
    allr.to_csv(out, index=False); (RES / "etapa1_previsio.md").write_text(allr.to_markdown(index=False))
    print(allr.to_string(index=False))


GRID = {"k_dis": [0.0, 0.25, 0.5, 0.75, 1.0], "sigma_mode": ["global", "hod"],
        "k_future": [0.0, 0.15, 0.3], "horizon": [16, 24]}


def _one(fam, cfg, val):
    tbl = pickle.load(open(CACHE / f"fc_{fam}.pkl", "rb"))
    r = run(load_agent({"_models": {}, "_fc_table": tbl, **cfg}), val)
    return {"familia": fam, **cfg, "score": r["score"], "co2_%": r["co2_estalviat_pct"],
            "cost_%": r["cost_estalviat_pct"], "infr": r["infraccions"]}


def stage2(fams):
    df = load_hard(); _, val = split_val(df)
    cfgs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    jobs = [(f, c) for f in fams for c in cfgs]
    print(len(jobs), "configuracions", flush=True)
    rows = Parallel(n_jobs=2, verbose=0)(delayed(_one)(f, c, val) for f, c in jobs)
    res = pd.DataFrame(rows).sort_values("score", ascending=False)
    res.to_csv(RES / "etapa2_decisio.csv", index=False)
    (RES / "etapa2_decisio.md").write_text(res.head(40).to_markdown(index=False))
    print(res.head(15).to_string(index=False))
    for p in GRID:
        print(res.groupby("familia" if p == "k_dis" else p)["score"].mean().round(3).to_dict() if p != "k_dis" else
              res.groupby("k_dis")["score"].mean().round(3).to_dict(), "<-", p)


if __name__ == "__main__":
    if sys.argv[1] == "stage1":
        stage1(sys.argv[2:] or ["perfecte", "naive", "lgbm_base", "lgbm_petit", "lgbm_gran", "xgb", "hgb", "ensemble"])
    else:
        stage2(sys.argv[2:])

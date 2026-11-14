"""
src/train_lib.py
Funcions d'entrenament / validacio offline per al Mode Dificil.
Les fa servir train_pipeline.ipynb. NO forma part de submission.zip.
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

try:
    from tqdm.auto import tqdm
except Exception:                       # sense tqdm: iterador normal
    def tqdm(it=None, **kw):
        return it

import agent as A
from data_prep import PV_SCALE
from hard_env import HardModeEnv, SPEC_HARD, run_hard

COLMAP = {"load": "load_kw", "pv": "pv_kw", "irr": "irradiancia_wm2", "tamb": "temp_ambient_c",
          "tpan": "temp_panel_c", "price": "precio_eur_per_kwh", "co2": "co2_kg_per_kwh"}
TARGETS = ("load", "pv", "price", "co2")
K_MIN = {"load": 0, "pv": 0, "price": 3, "co2": 3}     # preu/CO2: k=0..2 son exactes


# ----------------------------------------------------------------------------------
# Dades -> arrays sobre l'eix absolut de l'agent
# ----------------------------------------------------------------------------------
def to_arrays(hourly: pd.DataFrame) -> dict:
    """Taula horaria neta -> dict d'arrays (N_HOURS,) amb NaN on no hi ha dada.
    Imputacio de forats curts (<= 3 h) per interpolacio lineal; els forats llargs
    (novembre 2025, setmanes de test) NO s'interpolen ni es propaguen."""
    idx = A.full_index()
    out = {}
    for name, col in COLMAP.items():
        s = hourly[col].reindex(idx)
        if name in A.TELEMETRY:
            s = interpolate_short(s, 3)
        out[name] = s.values.astype(float)
    return out


def interpolate_short(s: pd.Series, max_gap: int = 3) -> pd.Series:
    """Interpola nomes forats interiors de longitud <= max_gap hores."""
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    run = isna.groupby(grp).transform("size")
    filled = s.interpolate(limit_area="inside")
    return s.where(~(isna & (run <= max_gap)), filled)


def impute_profile(s: pd.Series) -> pd.Series:
    """Estrategia alternativa (per comparar): mediana mes / dia-setmana / hora."""
    key = [s.index.month, s.index.dayofweek, s.index.hour]
    med = s.groupby(key).transform("median")
    return s.fillna(med)


def clear_sky_table(pv: np.ndarray, q: float = 0.97, win: int = 15) -> np.ndarray:
    """Envolupant de cel clar (367, 24): quantil alt de pv per (dia de l'any +-win, hora)."""
    idx = A.full_index()
    df = pd.DataFrame({"pv": pv, "doy": idx.dayofyear, "h": idx.hour}).dropna()
    tab = np.full((367, 24), np.nan)
    for h in range(24):
        d = df[df.h == h]
        if d.empty:
            continue
        doy = d.doy.values
        v = d.pv.values
        for day in range(1, 367):
            dist = np.abs(doy - day)
            dist = np.minimum(dist, 366 - dist)
            m = dist <= win
            if m.sum() >= 5:
                tab[day, h] = np.quantile(v[m], q)
    # forats (novembre): interpolacio circular al llarg de l'any
    for h in range(24):
        col = pd.Series(np.concatenate([tab[1:, h]] * 3))
        col = col.interpolate(limit_direction="both")
        tab[1:, h] = col.values[366:732]
    tab[0] = tab[1]
    tab = np.nan_to_num(tab, nan=0.0)
    tab[tab < 5.0] = 0.0
    return tab


# ----------------------------------------------------------------------------------
# Blocs de validacio (mirall del test ocult)
# ----------------------------------------------------------------------------------
def validation_blocks(env_df: pd.DataFrame):
    """Per cada mes de train: els darrers 168 h consecutius disponibles (dies ~17-23),
    es a dir, la setmana immediatament anterior a la setmana oculta."""
    blocks = []
    for per, g in env_df.groupby(env_df.index.to_period("M")):
        if len(g) < 336:
            continue
        end = g.index.max()
        start = end - pd.Timedelta(hours=167)
        blk = env_df.loc[start:end]
        if len(blk) >= 150:
            blocks.append((start, end))
    return blocks


def synthetic_november(env_df: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Bloc sintetic 17-23 Nov 2025: preu i CO2 REALS de novembre + consum / solar / meteo
    = mitjana de dues setmanes reals alineades per dia de la setmana (13-19 Oct i 1-7 Des)."""
    idx = pd.date_range("2025-11-17", periods=168, freq="1h")
    a = env_df.loc[NOV_SOURCES[0][0]:NOV_SOURCES[0][1]]
    b = env_df.loc[NOV_SOURCES[1][0]:NOV_SOURCES[1][1]]
    cols = ["load_kw", "pv_kw", "temp_ambient_c", "temp_panel_c", "vent_ms", "irradiancia_wm2"]
    a = a.reindex(pd.date_range("2025-10-13", periods=168, freq="1h"))[cols].interpolate(limit_direction="both")
    b = b.reindex(pd.date_range("2025-12-01", periods=168, freq="1h"))[cols].interpolate(limit_direction="both")
    out = pd.DataFrame(0.5 * (a.values + b.values), index=idx, columns=cols)
    out["precio_eur_per_kwh"] = hourly["precio_eur_per_kwh"].reindex(idx).values
    out["co2_kg_per_kwh"] = hourly["co2_kg_per_kwh"].reindex(idx).values
    out.index.name = "Fecha"
    return out[env_df.columns]


def make_validation_frame(env_df, hourly):
    """12 blocs de 168 h concatenats (11 reals + novembre sintetic), ordre cronologic."""
    blocks = validation_blocks(env_df)
    parts = [env_df.loc[s:e] for s, e in blocks]
    nov = synthetic_november(env_df, hourly)
    parts.append(nov)
    blocks.append((nov.index.min(), nov.index.max()))
    order = np.argsort([p.index.min() for p in parts])
    parts = [parts[i] for i in order]
    blocks = [blocks[i] for i in order]
    return pd.concat(parts), blocks


# Setmanes reals amb que es construeix el bloc sintetic de novembre: tambe s'han d'amagar
# de l'entrenament i de l'historial de l'agent, o la validacio de novembre seria optimista.
NOV_SOURCES = [(pd.Timestamp("2025-10-13"), pd.Timestamp("2025-10-19 23:00")),
               (pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-07 23:00"))]


def mask_blocks(arrays: dict, blocks) -> dict:
    """Copia dels arrays amb els blocs de validacio amagats (NaN). Si hi ha el bloc sintetic
    de novembre, tambe s'amaga la telemetria de les seves dues setmanes font."""
    out = {k: v.copy() for k, v in arrays.items()}
    if any(s.month == 11 for s, _ in blocks):
        for s, e in NOV_SOURCES:
            i0, i1 = A.hour_index(s), A.hour_index(e) + 1
            for k in A.TELEMETRY:
                out[k][i0:i1] = np.nan
    for s, e in blocks:
        i0, i1 = A.hour_index(s), A.hour_index(e) + 1
        for k in out:
            out[k][i0:i1] = np.nan
    return out


# ----------------------------------------------------------------------------------
# Datasets supervisats (mateix codi de features que l'agent)
# ----------------------------------------------------------------------------------
def nov_gap_mask() -> np.ndarray:
    """Hores t la finestra de lags de les quals (fins a 168 h) creua el forat Oct->Des 2025."""
    m = np.zeros(A.N_HOURS, dtype=bool)
    m[A.hour_index("2025-11-01"):A.hour_index("2025-12-08")] = True
    return m


def build_dataset(arrays: dict, cs_table: np.ndarray, target: str, t_range=None):
    """Retorna X (n, F), y (n,), t (n,), k (n,) per a totes les parelles (t, k) amb
    objectiu conegut. Les features es construeixen amb agent.Features (identic a online)."""
    idx = A.full_index()
    cal = A.calendar_matrix(idx)
    cs = A.clear_sky_series(cs_table, idx)
    feat = A.Features(arrays, cal, cs)
    y_all = arrays[target]
    anchor = arrays["price"]        # hores amb mercat conegut = hores de train
    gap = nov_gap_mask()
    if t_range is None:
        t_range = range(200, A.N_HOURS - A.HORIZON - 2)
    Xs, ys, ts, ks = [], [], [], []
    kmin = K_MIN[target]
    for t in t_range:
        if anchor[t] != anchor[t]:
            continue
        if target in ("load", "pv"):
            if gap[t]:
                continue                       # no entrenem a traves del forat de novembre
            if y_all[t - 1] != y_all[t - 1]:
                continue                       # sense telemetria recent (inici de segment)
        else:
            if anchor[t + 2] != anchor[t + 2] or anchor[t - 1] != anchor[t - 1]:
                continue
        y = y_all[t:t + A.HORIZON]
        ok = (y == y)
        ok[:kmin] = False
        if not ok.any():
            continue
        X = feat.build(target, t)
        Xs.append(X[ok]); ys.append(y[ok])
        ts.append(np.full(int(ok.sum()), t)); ks.append(A._K[ok])
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(ts), np.concatenate(ks))


# ----------------------------------------------------------------------------------
# Families de models
# ----------------------------------------------------------------------------------
def gpu_available() -> bool:
    try:
        import subprocess
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False


LGB_PARAMS = dict(objective="l1", learning_rate=0.1, num_leaves=31, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=7, num_threads=4)


def fit_lgb(X, y, n_rounds=400, params=None, names=None):
    import lightgbm as lgb
    p = dict(LGB_PARAMS)
    p.update(params or {})
    ds = lgb.Dataset(X, y, feature_name=list(names) if names else "auto")
    return lgb.train(p, ds, num_boost_round=n_rounds)


def fit_xgb(X, y, n_rounds=400, use_gpu=False):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=n_rounds, learning_rate=0.05, max_depth=7, subsample=0.8,
                         colsample_bytree=0.8, objective="reg:absoluteerror", tree_method="hist",
                         device="cuda" if use_gpu else "cpu", n_jobs=4, random_state=7)
    return m.fit(X, y)


def fit_catboost(X, y, n_rounds=400, use_gpu=False):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=n_rounds, learning_rate=0.08, depth=7, loss_function="MAE",
                          task_type="GPU" if use_gpu else "CPU", verbose=0, random_seed=7,
                          metric_period=n_rounds, allow_writing_files=False,   # sense avisos GPU
                          thread_count=4)
    return m.fit(X, y)


def fit_rf(X, y):
    from sklearn.ensemble import RandomForestRegressor
    Xf = np.where(np.isnan(X), -999.0, X)
    m = RandomForestRegressor(n_estimators=60, min_samples_leaf=5, max_features=0.5, n_jobs=4,
                              random_state=7).fit(Xf, y)
    m._nanfill = True
    return m


def fit_ridge(X, y, k, alpha=10.0):
    """Ridge directe: un vector de pesos per horitzo k. Retorna l'especificacio numpy
    {W (24,F), b (24,), mu, sd} que l'agent pot avaluar sense dependencies."""
    from sklearn.linear_model import Ridge
    mu = np.nanmean(X, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    Xi = np.where(np.isnan(X), mu, X)
    sd = Xi.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (Xi - mu) / sd
    F = X.shape[1]
    W = np.zeros((A.HORIZON, F)); b = np.zeros(A.HORIZON)
    for kk in range(A.HORIZON):
        m = k == kk
        if m.sum() < 50:
            continue
        r = Ridge(alpha=alpha).fit(Z[m], y[m])
        W[kk] = r.coef_; b[kk] = r.intercept_
    return {"W": W, "b": b, "mu": mu, "sd": sd}


def predict_ridge(spec, X, k):
    Z = (np.where(np.isnan(X), spec["mu"], X) - spec["mu"]) / spec["sd"]
    return (Z * spec["W"][k]).sum(axis=1) + spec["b"][k]


def predict_any(model, X, k):
    if isinstance(model, dict):
        return predict_ridge(model, X, k)
    if getattr(model, "_nanfill", False):
        return model.predict(np.where(np.isnan(X), -999.0, X))
    return model.predict(X)


def seasonal_naive(X, names):
    """Baseline: valor de la mateixa hora d'ahir (lag 24 h de l'hora objectiu)."""
    col = [n for n in names if n.endswith("_h24")][0]
    j = list(names).index(col)
    fb = [n for n in names if n in ("l1", "pv1", "y2")][0]
    p = X[:, j].copy()
    p[np.isnan(p)] = X[np.isnan(p), list(names).index(fb)]
    return p


def mae(y, p):
    m = ~np.isnan(p) & ~np.isnan(y)
    return float(np.mean(np.abs(y[m] - p[m])))


def benchmark_target(target, arrays_tr, arrays_full, cs_table, blocks, families, use_gpu=False,
                     n_rounds=400):
    """Entrena cada familia sense els blocs de validacio i mesura MAE sobre els blocs.
    Les features de validacio fan servir l'historial real (com passara al test)."""
    names = A.FEATURE_NAMES[target]
    Xtr, ytr, ttr, ktr = build_dataset(arrays_tr, cs_table, target)
    t_val = np.concatenate([np.arange(A.hour_index(s), A.hour_index(e) + 1) for s, e in blocks])
    Xva, yva, tva, kva = build_dataset(arrays_full, cs_table, target, t_range=t_val)
    in_blk = np.isin(tva + kva, t_val)
    Xva, yva, kva = Xva[in_blk], yva[in_blk], kva[in_blk]
    rows, models = [], {}
    bar = tqdm(list(families), desc=f"models [{target}]", leave=False)
    for fam in bar:
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(fam)
        t0 = time.time()
        if fam == "seasonal_naive":
            model = None; p = seasonal_naive(Xva, names)
        elif fam == "ridge":
            model = fit_ridge(Xtr, ytr, ktr); p = predict_any(model, Xva, kva)
        elif fam == "lightgbm_stacked":
            model = fit_lgb(Xtr, ytr, n_rounds, names=names); p = model.predict(Xva)
        elif fam == "lightgbm_buckets":
            model = fit_lgb_buckets(Xtr, ytr, ktr, target); p = predict_buckets(model, Xva, kva)
            rdg = fit_ridge(Xtr, ytr, ktr)             # k >= 12: Ridge (com fa l'agent)
            p = np.where(np.isnan(p), predict_ridge(rdg, Xva, kva), p)
        elif fam == "xgboost":
            model = fit_xgb(Xtr, ytr, n_rounds, use_gpu); p = model.predict(Xva)
        elif fam == "catboost":
            model = fit_catboost(Xtr, ytr, n_rounds, use_gpu); p = model.predict(Xva)
        elif fam == "random_forest":
            model = fit_rf(Xtr, ytr); p = predict_any(model, Xva, kva)
        else:
            raise ValueError(fam)
        short = kva <= 5
        rows.append({"target": target, "model": fam, "MAE": mae(yva, p),
                     "MAE_k<=5": mae(yva[short], p[short]), "MAE_k>5": mae(yva[~short], p[~short]),
                     "n_train": len(ytr), "n_val": len(yva), "fit_s": round(time.time() - t0, 1)})
        models[fam] = model
    return pd.DataFrame(rows), models


LGB_OBJECTIVE = {"load": "l2", "pv": "l1", "price": "l2", "co2": "l1"}


def fit_lgb_buckets(X, y, k, target, n_rounds=150, params=None):
    """LightGBM per grup d'horitzo (agent.BUCKETS). Retorna [(k0, k1, booster), ...]."""
    out = []
    p = {"objective": LGB_OBJECTIVE[target]}
    p.update(params or {})
    for k0, k1 in A.BUCKETS:
        m = (k >= k0) & (k <= k1)
        if m.sum() < 200:
            continue
        out.append((k0, k1, fit_lgb(X[m], y[m], n_rounds, params=p, names=A.FEATURE_NAMES[target])))
    return out


def predict_buckets(models, X, k):
    p = np.full(len(k), np.nan)
    for k0, k1, bst in models:
        m = (k >= k0) & (k <= k1)
        if m.any():
            p[m] = bst.predict(X[m])
    return p


def train_final_models(arrays, cs_table, n_rounds=150, lgb_params=None, cap_qs=(0.2,)):
    # lgb_params: None, o dict per objectiu {"load": {"n_rounds": 200, "num_leaves": 31, ...}}
    """Entrena LightGBM per grups d'horitzo + Ridge (reserva) per als 4 objectius. Retorna
    l'especificacio serialitzable de l'agent (cadenes de model LightGBM + pesos numpy)."""
    out = {}
    for target in tqdm(TARGETS, desc="entrenament LightGBM+Ridge", leave=False):
        X, y, t, k = build_dataset(arrays, cs_table, target)
        lp = dict((lgb_params or {}).get(target, {}))
        nr = int(lp.pop("n_rounds", n_rounds))
        bk = fit_lgb_buckets(X, y, k, target, nr, lp or None)
        out[target] = {"lgb": [(k0, k1, b.model_to_string()) for k0, k1, b in bk],
                       "ridge": fit_ridge(X, y, k), "features": list(A.FEATURE_NAMES[target])}
    out["cap"] = train_cap_models(arrays, cs_table, cap_qs)
    return out


def train_cap_models(arrays, cs_table, qs, n_rounds=150):
    """Models quantil a k=0: consum al quantil q i solar al quantil 1-q. La seva diferencia
    es una cota inferior prudent del consum net -> sostre de descarrega sense exportacio."""
    out = {}
    if not len(qs):
        return out
    Xl, yl, _, kl = build_dataset(arrays, cs_table, "load")
    Xp, yp, _, kp = build_dataset(arrays, cs_table, "pv")
    Xl, yl, Xp, yp = Xl[kl == 0], yl[kl == 0], Xp[kp == 0], yp[kp == 0]
    for q in qs:
        bl = fit_lgb(Xl, yl, n_rounds, params={"objective": "quantile", "alpha": float(q)},
                     names=A.FEATURE_NAMES["load"])
        bp = fit_lgb(Xp, yp, n_rounds, params={"objective": "quantile", "alpha": 1.0 - float(q)},
                     names=A.FEATURE_NAMES["pv"])
        out[float(q)] = (bl.model_to_string(), bp.model_to_string())
    return out


# ----------------------------------------------------------------------------------
# Optuna: hiperparametres dels models i del control
# ----------------------------------------------------------------------------------
def optuna_tune_lgb(target, arrays_tr, arrays_full, cs_table, blocks, n_trials=20, seed=7):
    """Cerca bayesiana (TPE) dels hiperparametres de LightGBM per a un objectiu, minimitzant
    el MAE fora de mostra sobre els blocs de validacio (horitzons amb arbres, k <= 11).
    Retorna (millors_parametres, study). El primer assaig son els valors per defecte."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xtr, ytr, _, ktr = build_dataset(arrays_tr, cs_table, target)
    t_val = np.concatenate([np.arange(A.hour_index(s), A.hour_index(e) + 1) for s, e in blocks])
    Xva, yva, tva, kva = build_dataset(arrays_full, cs_table, target, t_range=t_val)
    ok = np.isin(tva + kva, t_val) & (kva <= A.BUCKETS[-1][1])
    Xva, yva, kva = Xva[ok], yva[ok], kva[ok]

    def objective(trial):
        p = {"learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
             "num_leaves": trial.suggest_int("num_leaves", 15, 63),
             "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
             "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
             "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10.0, log=True)}
        nr = trial.suggest_int("n_rounds", 80, 300)      # limitat per la latencia de l'agent
        bk = fit_lgb_buckets(Xtr, ytr, ktr, target, nr, p)
        return mae(yva, predict_buckets(bk, Xva, kva))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({"learning_rate": LGB_PARAMS["learning_rate"], "num_leaves": LGB_PARAMS["num_leaves"],
                         "min_data_in_leaf": LGB_PARAMS["min_data_in_leaf"],
                         "feature_fraction": LGB_PARAMS["feature_fraction"],
                         "bagging_fraction": LGB_PARAMS["bagging_fraction"],
                         "lambda_l2": LGB_PARAMS["lambda_l2"], "n_rounds": 150})
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return dict(study.best_params), study


CONTROL_SPACE = {
    "sigma_scale": ("float", 0.0, 1.5), "cap_q": ("cat", [0.0, 0.1, 0.2, 0.3]),
    "buf_kw": ("float", 0.0, 40.0), "gamma": ("float", 0.8, 1.1), "w_co2": ("float", 0.6, 0.8),
    "net_bias_kw": ("float", -40.0, 40.0), "blend_load": ("float", 0.2, 0.8),
    "blend_pv": ("float", 0.2, 0.8), "blend_price": ("float", 0.2, 0.8),
    "blend_co2": ("float", 0.2, 0.8),
}


def _control_params(flat):
    """Parametres plans d'Optuna -> parametres de l'agent."""
    p = {k: v for k, v in flat.items() if not k.startswith("blend_")}
    p["blend_w"] = {t: float(flat.get("blend_" + t, 0.5)) for t in TARGETS}
    return p


def split_scores(env, blocks):
    """Puntuacio oficial sobre els blocs parells (A) i senars (B) per separat."""
    ts = pd.DatetimeIndex(env.df["Fecha"])
    out = []
    for half in (0, 1):
        m = np.zeros(len(ts), dtype=bool)
        for i, (s, e) in enumerate(blocks):
            if i % 2 == half:
                m |= np.asarray((ts >= s) & (ts <= e))
        out.append(env.evaluate_results(mask=m)["score"])
    return out


def optuna_tune_control(artifact, val_df, blocks, n_trials=60, base=None, seed=7):
    """Cerca bayesiana (TPE) dels parametres del control contra la metrica oficial en Mode
    Dificil (aproximacio parametrica de la funcio de cost, ajustada sobre el simulador).

    Antisobreajust: nomes s'accepta una configuracio si millora la de per defecte a les DUES
    meitats dels blocs (parells i senars) i el guany total supera `min_gain`.
    Retorna (parametres_escollits, study, taula)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = dict(base or {})
    D0 = A.DEFAULT_PARAMS
    default_flat = {"sigma_scale": D0["sigma_scale"], "cap_q": D0["cap_q"], "buf_kw": D0["buf_kw"],
                    "gamma": D0["gamma"], "w_co2": D0["w_co2"], "net_bias_kw": D0["net_bias_kw"],
                    "blend_load": 0.5, "blend_pv": 0.5, "blend_price": 0.5, "blend_co2": 0.5}
    caps = set((artifact["models"].get("cap") or {}).keys())

    def objective(trial):
        flat = {}
        for name, spec in CONTROL_SPACE.items():
            if spec[0] == "cat":
                choices = [c for c in spec[1] if c == 0.0 or c in caps] if name == "cap_q" else spec[1]
                flat[name] = trial.suggest_categorical(name, choices)
            else:
                flat[name] = trial.suggest_float(name, spec[1], spec[2])
        res, env = simulate(artifact, val_df, {**base, **_control_params(flat)}, return_env=True)
        a, b = split_scores(env, blocks)
        trial.set_user_attr("score_A", a); trial.set_user_attr("score_B", b)
        trial.set_user_attr("infraccions", res["infraccions"])
        trial.set_user_attr("ms", res["ms_per_step_mean"])
        return res["score"] - 1000.0 * (res["infraccions"] > 0)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(default_flat)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    rows = [{"trial": t.number, "score": t.value, "score_A": t.user_attrs.get("score_A"),
             "score_B": t.user_attrs.get("score_B"), "ms/pas": t.user_attrs.get("ms"), **t.params}
            for t in study.trials if t.value is not None]
    tab = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    d = tab[tab.trial == 0].iloc[0]
    min_gain = 0.05
    okm = (tab.score_A > d.score_A) & (tab.score_B > d.score_B) & (tab.score >= d.score + min_gain)
    chosen = tab[okm].iloc[0] if okm.any() else d
    flat = {k: (float(chosen[k]) if k != "cap_q" else float(chosen[k])) for k in CONTROL_SPACE}
    tab.attrs["chosen_trial"] = int(chosen.trial)
    return {**base, **_control_params(flat)}, study, tab


# ----------------------------------------------------------------------------------
# Artefacte per a l'agent
# ----------------------------------------------------------------------------------
def baseline_refs(env_df: pd.DataFrame):
    """Mitjanes ponderades pel baseline (sense bateria): normalitzen l'objectiu 70/30."""
    base = np.maximum(env_df["load_kw"] - env_df["pv_kw"], 0.0)
    return (float((base * env_df["co2_kg_per_kwh"]).sum() / base.sum()),
            float((base * env_df["precio_eur_per_kwh"]).sum() / base.sum()))


def residual_sigma(models, arrays_full, cs_table, blocks, prefer="blend"):
    """Desviacio (RMSE) de l'error de prediccio per horitzo k, mesurada fora de mostra sobre
    els blocs de validacio amb models que no els han vist: consum en kW i solar com a
    fraccio de l'envolupant de cel clar. Alimenta el DP estocastic de l'agent."""
    idx = A.full_index()
    cs = A.clear_sky_series(cs_table, idx)
    out = {}
    t_val = np.concatenate([np.arange(A.hour_index(s), A.hour_index(e) + 1) for s, e in blocks])
    for tg, key in (("load", "load"), ("pv", "pv_frac")):
        X, y, t, k = build_dataset(arrays_full, cs_table, tg, t_range=t_val)
        ok = np.isin(t + k, t_val)
        X, y, t, k = X[ok], y[ok], t[ok], k[ok]
        # mateixa combinacio que agent.Forecaster, vectoritzada per grup d'horitzo
        import lightgbm as lgb
        p = np.full(len(y), np.nan)
        if prefer != "ridge":
            for k0, k1, txt in models[tg]["lgb"]:
                m = (k >= k0) & (k <= k1)
                if m.any():
                    p[m] = lgb.Booster(model_str=txt).predict(X[m])
        r = predict_ridge(models[tg]["ridge"], X, k)
        bw = 0.5 if prefer == "blend" else 0.0
        p = np.where(np.isnan(p), r, (1.0 - bw) * p + bw * r)
        err = y - p
        sig = np.zeros(A.HORIZON)
        for kk in range(A.HORIZON):
            m = (k == kk) & ~np.isnan(err)
            if tg == "pv":
                c = cs[t[m] + kk]
                day = c > 50.0
                sig[kk] = float(np.sqrt(np.mean((err[m][day] / c[day]) ** 2))) if day.any() else 0.0
            else:
                sig[kk] = float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else 0.0
        out[key] = sig
    return out


def make_artifact(arrays, cs_table, models, env_df, params=None, sigma=None):
    co2_ref, price_ref = baseline_refs(env_df)
    last = max(np.flatnonzero(~np.isnan(v)).max() for v in arrays.values()) + 1
    m = (~np.isnan(arrays["pv"])) & (arrays["pv"] > 50) & ~np.isnan(arrays["irr"])
    irr_per_pv = float(np.median(arrays["irr"][m] / arrays["pv"][m])) if m.any() else 0.6
    p = {"co2_ref": co2_ref, "price_ref": price_ref}
    p.update(params or {})
    return {"hist": {k: v[:last].astype(np.float32) for k, v in arrays.items()},
            "cs_table": cs_table.astype(np.float32), "models": models,
            "irr_per_pv": irr_per_pv, "params": p, "pv_scale": PV_SCALE,
            "sigma": {k: np.asarray(v, dtype=float) for k, v in (sigma or {}).items()}}


# ----------------------------------------------------------------------------------
# Simulacio i cotes
# ----------------------------------------------------------------------------------
def simulate(artifact, df_env, params=None, spec=None, return_env=False, desc=None):
    art = dict(artifact)
    art["params"] = dict(artifact.get("params", {}))
    art["params"].update(params or {})
    agent = A.BatteryAgent(artifact=art)
    assert agent.ready, "l'agent no ha pogut carregar l'artefacte"
    out = run_hard(agent, df_env, spec=spec or SPEC_HARD, return_env=return_env, desc=desc)
    (out[0] if return_env else out)["fallbacks"] = agent.n_fallback
    return out


class ModelAgent(A.BatteryAgent):
    """Mateix agent i mateix control, pero amb les prediccions de qualsevol familia de
    models (per comparar families amb la metrica oficial; no es limita la latencia)."""

    def set_family(self, family, models):
        self.family = family
        self.fam_models = models        # dict objectiu -> model

    def _predict_all(self, t, Xl, Xp):
        f = self.feat
        Xs = {"load": Xl, "pv": Xp, "price": f.market(t, "price", "co2"),
              "co2": f.market(t, "co2", "price")}
        out = []
        for tg in TARGETS:
            X = Xs[tg]
            if self.family == "seasonal_naive":
                p = seasonal_naive(X, A.FEATURE_NAMES[tg])
            else:
                m = self.fam_models[tg]
                if isinstance(m, list):
                    p = predict_buckets(m, X, A._K)
                    r = self.fam_models.get(tg + "_ridge")
                    if r is not None:
                        p = np.where(np.isnan(p), predict_ridge(r, X, A._K), p)
                else:
                    p = np.asarray(predict_any(m, X, A._K), dtype=float)
            out.append(p)
        return tuple(out)


def simulate_family(artifact, df_env, family, models, params=None):
    art = dict(artifact)
    art["params"] = dict(artifact.get("params", {}))
    art["params"].update(params or {})
    ag = ModelAgent(artifact=art)
    assert ag.ready
    ag.set_family(family, models)
    return run_hard(ag, df_env, desc=f"sim {family}")


class OracleAgent:
    """Cota superior: mateix LP amb coneixement perfecte de tota la serie."""

    def __init__(self, df_env, spec=None):
        spec = spec or SPEC_HARD
        self.plan = oracle_plan(df_env, spec)
        self.i = 0
        self.cap = spec["capacity_kwh"]; self.pmax = spec["power_kw"]

    def act(self, obs):
        a = float(self.plan[self.i]); self.i += 1
        eta = 0.9 ** 0.5; soc = float(obs["soc_kwh"]); eps = 1e-4
        mc = max(0.0, min(self.pmax, (self.cap - soc) / eta - eps))
        md = max(0.0, min(self.pmax, soc * eta - eps))
        return float(max(-md, min(mc, a)))


def oracle_plan(df_env, spec):
    n = len(df_env)
    eta = spec["efficiency"] ** 0.5
    cap = spec["capacity_kwh"]; pmax = spec["power_kw"]; soc0 = spec["soc_init"] * cap
    net = (df_env["load_kw"] - df_env["pv_kw"]).values
    co2_ref, price_ref = baseline_refs(df_env)
    w = 0.7 * df_env["co2_kg_per_kwh"].values / co2_ref + 0.3 * df_env["precio_eur_per_kwh"].values / price_ref
    # variables: c (n), d (n), g (n), s (n)
    Aeq = lil_matrix((n, 4 * n)); beq = np.zeros(n)
    Aub = lil_matrix((n, 4 * n)); bub = -net
    for i in range(n):
        Aeq[i, 3 * n + i] = 1.0; Aeq[i, i] = -eta; Aeq[i, n + i] = 1.0 / eta
        if i > 0:
            Aeq[i, 3 * n + i - 1] = -1.0
        else:
            beq[i] = soc0
        Aub[i, i] = 1.0; Aub[i, n + i] = -1.0; Aub[i, 2 * n + i] = -1.0
    c = np.concatenate([np.full(2 * n, 1e-6), w, np.zeros(n)])
    bounds = [(0, pmax)] * (2 * n) + [(0, None)] * n + [(0, cap)] * n
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return res.x[:n] - res.x[n:2 * n]


def leakage_test(arrays, cs_table, n_trials=40, seed=0):
    """Test de no-filtracio: pertorba TOT el futur no permes (telemetria >= t, mercat >= t+3)
    i comprova que les features de l'agent no canvien ni un bit."""
    rng = np.random.default_rng(seed)
    idx = A.full_index()
    cal = A.calendar_matrix(idx)
    cs = A.clear_sky_series(cs_table, idx)
    ok_t = np.flatnonzero(~np.isnan(arrays["price"]))
    ok_t = ok_t[(ok_t > 400) & (ok_t < A.N_HOURS - 60)]
    worst = 0.0
    for t in rng.choice(ok_t, n_trials, replace=False):
        f0 = A.Features(arrays, cal, cs)
        pert = {k: v.copy() for k, v in arrays.items()}
        for k in A.TELEMETRY:
            pert[k][t:] = rng.normal(size=A.N_HOURS - t) * 1e3
        for k in ("price", "co2"):
            pert[k][t + 3:] = rng.normal(size=A.N_HOURS - t - 3) * 1e3
        f1 = A.Features(pert, cal, cs)
        for target in TARGETS:
            a, b = f0.build(target, int(t)), f1.build(target, int(t))
            d = np.nanmax(np.abs(np.where(np.isnan(a) & np.isnan(b), 0.0, a - b)))
            if not (np.isnan(a) == np.isnan(b)).all():
                d = np.inf
            worst = max(worst, float(d))
    return worst

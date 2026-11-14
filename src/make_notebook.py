"""
src/make_notebook.py
Genera train_pipeline.ipynb AUTOCONTINGUT: incrusta agent.py, data_prep.py, hard_env.py,
train_lib.py i una copia de starter_kit/environment.py en cel.les %%writefile, de manera
que a Kaggle nomes cal pujar el notebook + els CSV de data/train.

    python src/make_notebook.py            # regenera el notebook despres de tocar el codi
"""
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


def embed(path, name):
    code(f"%%writefile hm_build/{name}\n" + Path(path).read_text(encoding="utf-8"))


md("""
# Repte UPC Sostenible — Mode Difícil: entrenament, validació i exportació

Notebook **autocontingut**: funciona igual a **Kaggle**, a **Google Colab** i en local. Fa tot el pipeline:

1. Escriu els mòduls (`agent.py`, simulador, preprocessat, llibreria d'entrenament) a `hm_build/`.
2. Preprocessa `data/train` (PV ×12), tracta el forat de novembre 2025 i compara imputacions.
3. Construeix el **nostre conjunt de test**: 12 blocs de 168 h (la darrera setmana disponible de cada mes + novembre sintètic).
4. Entrena i **compara famílies de models** (seasonal-naive, Ridge, LightGBM, XGBoost, CatBoost, RandomForest) per MAE *i* per la puntuació oficial en Mode Difícil.
5. Ajusta amb **Optuna** els hiperparàmetres de LightGBM i els del control contra `CampusBatteryEnv` (6.000 kWh / ±1.500 kW / η 0,90 / horitzó 3 h).
6. Reentrena amb totes les dades, exporta `model_bateria.pkl` + `agent.py` i genera `submission.zip`.

**Dades** (els 6 CSV de `data/train/`, solts o dins d'un `.zip`): es localitzen automàticament.

* **Kaggle:** pugeu-los com a *Dataset* i afegiu-lo al notebook (`/kaggle/input/**`). No cal Internet.
* **Colab:** arrossegueu els CSV o un `data.zip` al panell de fitxers (`/content`), o deseu-los a Google Drive i poseu `MOUNT_DRIVE = True`. Si no es troben, el notebook obre un diàleg per pujar-los.
* **Local:** s'usa `../data/train` del repositori.

La GPU (CUDA) és opcional: es detecta i s'usa per a XGBoost/CatBoost. El resultat és `hm_build/submission.zip`.
""")
code("""
# ---------------- Configuració ----------------
DATA_DIR = None        # None = autodetecta (Kaggle, Colab, Drive, zip o repo local); o una ruta
MOUNT_DRIVE = False    # Colab: True per muntar Google Drive i buscar-hi les dades
FAST = False           # True = menys arbres i sense RandomForest/CatBoost (prova ràpida)
CHOSEN = "blend"       # model que s'exporta a l'agent: "blend" (LightGBM+Ridge), "lgb" o "ridge"
N_TRIALS_LGB = 20      # assaigs d'Optuna per objectiu (hiperparametres de LightGBM); 0 = no ajustar
N_TRIALS_CTRL = 60     # assaigs d'Optuna per als parametres del control; 0 = valors per defecte
FAMILIES = ["seasonal_naive", "ridge", "lightgbm_stacked", "lightgbm_buckets", "xgboost",
            "catboost", "random_forest"]
import os, sys, subprocess, zipfile
os.makedirs("hm_build", exist_ok=True)
IN_COLAB = "google.colab" in sys.modules or os.path.isdir("/content/sample_data")
IN_KAGGLE = os.path.isdir("/kaggle/input")
for pkg in ("lightgbm", "xgboost", "catboost", "optuna", "tqdm"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])
warnings_off = os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _walk(root, depth):
    # os.walk amb profunditat limitada (Drive pot ser enorme)
    root = os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("hm_build", "sample_data", "__pycache__")]
        if d.count(os.sep) - base >= depth:
            dirs[:] = []
        yield d, files


def locate_data(data_dir=None):
    # Retorna la carpeta que conte contador.csv. Ordre: ruta explicita, carpetes conegudes,
    # un .zip amb els CSV (s'extreu a hm_data/), i a Colab un dialeg per pujar fitxers.
    need = "contador.csv"
    if data_dir and os.path.exists(os.path.join(data_dir, need)):
        return os.path.abspath(data_dir)
    if IN_COLAB and MOUNT_DRIVE and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    roots = [("data/train", 1), ("../data/train", 1), ("hm_data", 4), (".", 3), ("..", 2),
             ("/kaggle/input", 5), ("/content", 3), ("/content/drive/MyDrive", 4)]
    roots = [(r, d) for r, d in roots if os.path.isdir(r)]
    zips = []
    for r, depth in roots:
        for d, files in _walk(r, depth):
            if need in files:
                return d
            zips += [os.path.join(d, f) for f in files if f.lower().endswith(".zip")]

    def unzip(path):
        try:
            with zipfile.ZipFile(path) as z:
                if any(os.path.basename(m) == need for m in z.namelist()):
                    z.extractall("hm_data")
                    return True
        except Exception:
            pass
        return False

    for zp in zips:
        if "submission" not in os.path.basename(zp).lower() and unzip(zp):
            return locate_data("hm_data") if os.path.exists("hm_data/" + need) else next(
                d for d, files in _walk("hm_data", 5) if need in files)
    if IN_COLAB:
        from google.colab import files
        print("No s'han trobat les dades: pugeu els 6 CSV de data/train (o un .zip que els contingui).")
        up = files.upload()
        os.makedirs("hm_data", exist_ok=True)
        for name in up:
            if name.lower().endswith(".zip"):
                unzip(name)
            else:
                os.replace(name, os.path.join("hm_data", os.path.basename(name)))
        for d, files_ in _walk("hm_data", 5):
            if need in files_:
                return d
    raise FileNotFoundError("No trobo contador.csv. Poseu DATA_DIR a la carpeta amb els CSV de data/train.")


DATA_DIR = locate_data(DATA_DIR)
missing = [f for f in ("contador.csv", "inversor.csv", "meteo.csv", "precio_pvpc.csv", "co2_red.csv")
           if not os.path.exists(os.path.join(DATA_DIR, f))]
assert not missing, f"Falten fitxers a {DATA_DIR}: {missing}"
print("Entorn:", "Colab" if IN_COLAB else "Kaggle" if IN_KAGGLE else "local", "| dades:", DATA_DIR)
""")
md("## 1. Mòduls del projecte (s'escriuen a `hm_build/`)\n`environment.py` és una còpia literal del simulador oficial de `starter_kit/`.")
embed(ROOT / "starter_kit" / "environment.py", "environment.py")
embed(SRC / "agent.py", "agent.py")
embed(SRC / "data_prep.py", "data_prep.py")
embed(SRC / "hard_env.py", "hard_env.py")
embed(SRC / "train_lib.py", "train_lib.py")
code("""
import sys, time, json, zipfile, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, "hm_build")
for m in ("environment", "agent", "data_prep", "hard_env", "train_lib"):
    sys.modules.pop(m, None)
import agent as A, data_prep as D, hard_env as HE, train_lib as T
from environment import CampusBatteryEnv, run_agent_simulation
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
USE_GPU = T.gpu_available()
if FAST:
    FAMILIES = [f for f in FAMILIES if f not in ("catboost", "random_forest")]
print("GPU:", USE_GPU, "| data:", DATA_DIR)
""")
md("""
## 2. Preprocessat (PV ×12) i cobertura de dades

* `build_hourly`: sèrie neta per a *features* (els `0.0` del comptador = caiguda de sensor → `NaN`).
* `build_env_frame`: taula «oficial» del simulador, calcada de `starter_kit/tutorial.ipynb` (els `0.0` es conserven, `dropna`), amb PV ×12.
""")
code("""
hourly = D.build_hourly(DATA_DIR)          # features
env_df = D.build_env_frame(DATA_DIR)       # veritat del simulador (df_hard)
cov = hourly.notna().groupby(hourly.index.to_period("M")).sum()
print("Hores amb dada per mes (novembre 2025 sense comptador/inversor/meteo; inversor fins 16/08/2026):")
display(cov[["precio_eur_per_kwh", "co2_kg_per_kwh", "load_kw", "pv_kw", "irradiancia_wm2"]])
print("vent_ms sempre 0:", float(hourly["vent_ms"].abs().max()) == 0.0,
      "| hores env:", len(env_df), "| net<0 (excedent solar):", f"{((env_df.load_kw - env_df.pv_kw) < 0).mean():.1%}")
""")
md("### 2.1 Comparació d'estratègies d'imputació\nAmaguem valors coneguts de `load_kw` (forats curts d'1–3 h i forats llargs de 48 h) i mesurem l'error de cada estratègia. Regla resultant: interpolació només per a forats ≤ 3 h; mai a través de forats llargs (novembre, setmanes ocultes), que es deixen a `NaN` per als lags.")
code("""
rng = np.random.default_rng(0)
s = hourly["load_kw"].copy()
known = np.flatnonzero(s.notna().values)
rows = []
for name, glen, n in (("forat curt (1-3 h)", (1, 3), 300), ("forat llarg (48 h)", (48, 48), 12)):
    hid = s.copy(); idxs = []
    for st in rng.choice(known[200:-200], n, replace=False):
        L = int(rng.integers(glen[0], glen[1] + 1)); hid.iloc[st:st + L] = np.nan; idxs.extend(range(st, st + L))
    idxs = np.array(sorted(set(idxs))); truth = s.iloc[idxs]
    cands = {"interpolacio lineal (sense limit)": hid.interpolate(limit_area="inside"),
             "interpolacio <= 3 h": T.interpolate_short(hid, 3),
             "mediana mes/dia-setmana/hora": T.impute_profile(hid)}
    for k, v in cands.items():
        p = v.iloc[idxs]; ok = p.notna() & truth.notna()
        rows.append({"cas": name, "estrategia": k, "MAE_kW": round(float((p[ok] - truth[ok]).abs().mean()), 1) if ok.any() else np.nan,
                     "cobertura": f"{ok.sum() / max(1, truth.notna().sum()):.0%}"})
display(pd.DataFrame(rows))
""")
md("## 3. Features, calendari i test de no-filtració\nLes features les construeix **el mateix codi** de `agent.py` (`agent.Features`) offline i online. El test pertorba tot el futur no permès (telemetria ≥ t, mercat ≥ t+3) i exigeix diferència exactament 0.")
code("""
arr = T.to_arrays(hourly)                                  # historic complet (NaN als forats)
cal = pd.DataFrame(A.calendar_matrix(hourly.index), index=hourly.index, columns=A.CAL_NAMES)
print("Hores marcades per flag:"); display(cal[[c for c in A.CAL_NAMES if c.startswith("is_")]].sum().astype(int).to_frame("hores").T)
val_df, blocks = T.make_validation_frame(env_df, hourly)
arr_tr = T.mask_blocks(arr, blocks)                        # train sense els blocs de validacio
cs_tr = T.clear_sky_table(arr_tr["pv"]); cs_full = T.clear_sky_table(arr["pv"])
leak = T.leakage_test(arr, cs_full, n_trials=40)
print("Test de leakage (max |diferencia| de features en pertorbar el futur):", leak); assert leak == 0.0
for tg in T.TARGETS: print(f"{tg:6s} {len(A.FEATURE_NAMES[tg])} features")
""")
md("""
## 4. El nostre conjunt de test: 12 blocs × 168 h en Mode Difícil

Per a cada mes, la **darrera setmana disponible a train** (la immediatament anterior a la setmana oculta), més un bloc **sintètic de novembre 2025** (preu i CO₂ reals; consum/solar/meteo = mitjana de dues setmanes reals d'octubre i desembre). Es concatenen sense reiniciar el SOC, igual que el test ocult.
""")
code("""
display(pd.DataFrame([(s, e, len(val_df.loc[s:e])) for s, e in blocks], columns=["inici", "fi", "hores"]))
env_tr = env_df.drop(val_df.index, errors="ignore")
for s_, e_ in T.NOV_SOURCES: env_tr = env_tr.drop(env_tr.loc[s_:e_].index)   # fonts del novembre sintetic
SPEC_HARD = HE.SPEC_HARD; print(SPEC_HARD)
ORACLE = "oracle (LP amb futur perfecte, cota superior)"
ref = {ORACLE: T.run_hard(T.OracleAgent(val_df), val_df),
       "heuristica de reserva (sense model)": T.run_hard(A.BatteryAgent(artifact={}), val_df)}
display(pd.DataFrame(ref).T[["score", "co2_estalviat_pct", "cost_estalviat_pct", "infraccions"]])
""")
md("## 5. Comparació de famílies de models — error de predicció (MAE)\nTots s'entrenen **sense** els blocs de validació i s'avaluen sobre ells (horitzons k=0..23 per a consum i solar, k=3..23 per a preu i CO₂).")
code("""
n_rounds = 150 if FAST else 400
bench, fam_models = [], {f: {} for f in FAMILIES}
for tg in T.tqdm(T.TARGETS, desc="objectius"):
    res, models = T.benchmark_target(tg, arr_tr, arr, cs_tr, blocks, FAMILIES, use_gpu=USE_GPU, n_rounds=n_rounds)
    bench.append(res)
    for f in FAMILIES: fam_models[f][tg] = models[f]
    if "ridge" in models and "lightgbm_buckets" in models:
        fam_models["lightgbm_buckets"][tg + "_ridge"] = models["ridge"]     # k>=12 sense arbres
bench = pd.concat(bench, ignore_index=True)
display(bench.pivot(index="model", columns="target", values="MAE").loc[FAMILIES, list(T.TARGETS)].round(4))
display(bench.round(4))
""")
md("## 6. Comparació de famílies — puntuació oficial en Mode Difícil (el que compta)\nMateix control i mateix simulador per a tothom; només canvia qui fa les prediccions. `blend` = LightGBM per grups d'horitzó + Ridge (el que s'exporta per defecte). Trieu a `CHOSEN` quin voleu exportar (`blend`, `lgb` o `ridge` són els desplegables dins del límit de latència).")
code("""
models_val = T.train_final_models(arr_tr, cs_tr, cap_qs=(0.1, 0.2, 0.3))
sigma = T.residual_sigma(models_val, arr, cs_tr, blocks)       # error fora de mostra per horitzo -> DP estocastic
art_val = T.make_artifact(arr_tr, cs_tr, models_val, env_tr, sigma=sigma)
print("sigma consum (kW) k=0..5:", np.round(sigma["load"][:6]), "| sigma solar (fraccio de cel clar):", np.round(sigma["pv_frac"][:6], 2))
rows = {}
for fam in T.tqdm(FAMILIES, desc="families (simulacio Mode Dificil)"):
    rows[fam] = T.simulate_family(art_val, val_df, fam, fam_models[fam])
for pref in T.tqdm(("blend", "lgb", "ridge"), desc="desplegables"):
    rows[f"DESPLEGABLE: {pref}"] = T.simulate(art_val, val_df, {"prefer": pref}, desc=f"sim {pref}")
cmp_ctrl = pd.DataFrame(rows).T[["score", "co2_estalviat_pct", "cost_estalviat_pct", "infraccions", "ms_per_step_mean"]]
cmp_ctrl["% de l'oracle"] = (100 * cmp_ctrl["score"].astype(float) / ref[ORACLE]["score"]).round(1)
display(cmp_ctrl.sort_values("score", ascending=False))
print("Model escollit per exportar: CHOSEN =", CHOSEN)
""")
md("""
## 7. Ajust d'hiperparàmetres amb Optuna

**7.1 LightGBM** (per objectiu): cerca bayesiana TPE sobre `learning_rate`, `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l2` i nombre d'arbres (limitat a 300 per la latència), minimitzant el MAE fora de mostra. Com que menys error no sempre vol dir més punts (*decision-focused learning*, Depoortere et al. 2026), els models ajustats només s'adopten si **no empitjoren la puntuació oficial**.
""")
code("""
lgb_best, lgb_log = {}, []
if N_TRIALS_LGB > 0:
    for tg in T.TARGETS:
        bp, study = T.optuna_tune_lgb(tg, arr_tr, arr, cs_tr, blocks, n_trials=(5 if FAST else N_TRIALS_LGB))
        lgb_best[tg] = bp
        lgb_log.append({"objectiu": tg, "MAE per defecte": study.trials[0].value, "MAE Optuna": study.best_value, **bp})
    display(pd.DataFrame(lgb_log).round(4))
    models_tuned = T.train_final_models(arr_tr, cs_tr, lgb_params=lgb_best, cap_qs=(0.1, 0.2, 0.3))
    sigma_tuned = T.residual_sigma(models_tuned, arr, cs_tr, blocks)
    art_tuned = T.make_artifact(arr_tr, cs_tr, models_tuned, env_tr, sigma=sigma_tuned)
    r0 = T.simulate(art_val, val_df, {"prefer": CHOSEN}); r1 = T.simulate(art_tuned, val_df, {"prefer": CHOSEN})
    print(f"Puntuacio Mode Dificil -> LightGBM per defecte: {r0['score']} ({r0['ms_per_step_mean']} ms/pas) | Optuna: {r1['score']} ({r1['ms_per_step_mean']} ms/pas)")
    USE_TUNED_LGB = r1["score"] >= r0["score"] and r1["infraccions"] == 0
    if USE_TUNED_LGB: art_val, sigma = art_tuned, sigma_tuned
    else: lgb_best = {}
    print("S'adopten els hiperparametres d'Optuna:", USE_TUNED_LGB)
""")
md("""
**7.2 Control** (aproximació paramètrica de la funció de cost ajustada sobre el simulador; Ghadimi & Powell 2023): `sigma_scale` (pes de l'error de predicció al DP estocàstic), `cap_q` (quantil del sostre de descàrrega), `buf_kw`, `gamma`, `w_co2`, `net_bias_kw` i els pesos de la mescla LightGBM/Ridge per objectiu. Objectiu = puntuació oficial sobre els 12 blocs.

**Antisobreajust:** 12 setmanes són poques per a 10 paràmetres. Una configuració només s'accepta si millora la de per defecte a **les dues meitats** dels blocs (parells i senars) i guanya ≥ 0,05 punts en total; si no, es mantenen els valors per defecte.
""")
code("""
best = {"prefer": CHOSEN}
if N_TRIALS_CTRL > 0:
    best, study_ctrl, tab_ctrl = T.optuna_tune_control(art_val, val_df, blocks, n_trials=(8 if FAST else N_TRIALS_CTRL), base={"prefer": CHOSEN})
    print("Assaig escollit:", tab_ctrl.attrs["chosen_trial"], "(0 = valors per defecte)")
    display(tab_ctrl.head(10).round(3))
    try:
        import optuna
        display(pd.Series(optuna.importance.get_param_importances(study_ctrl)).round(3).to_frame("importancia").T)
    except Exception as ex:
        print("sense importancies:", ex)
print("Parametres escollits:", best)
""")
md("## 8. Entrenament final amb totes les dades i exportació")
code("""
cq = best.get("cap_q", A.DEFAULT_PARAMS["cap_q"])
models_full = T.train_final_models(arr, cs_full, lgb_params=lgb_best, cap_qs=(cq,) if cq else ())
art_final = T.make_artifact(arr, cs_full, models_full, env_df, params=best, sigma=sigma)   # sigma mesurada fora de mostra
joblib.dump(art_final, "hm_build/model_bateria.pkl", compress=3)
with zipfile.ZipFile("hm_build/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("hm_build/agent.py", "agent.py"); z.write("hm_build/model_bateria.pkl", "model_bateria.pkl")
for f in ("model_bateria.pkl", "submission.zip"):
    Path(f).write_bytes(Path("hm_build", f).read_bytes())            # copia al costat del notebook
print({f: f"{Path('hm_build', f).stat().st_size / 1e6:.2f} MB" for f in ("model_bateria.pkl", "submission.zip")})
print("params:", art_final["params"])
""")
md("""
## 9. Validació final en Mode Difícil

* **Validació honesta**: 12 blocs amb l'artefacte entrenat *sense* aquests blocs.
* **Train complet** amb l'artefacte final (in-sample: serveix per comprovar infraccions i robustesa, no com a estimació).
* **Entorn oficial sense modificar** (`CampusBatteryEnv(df_hard, spec=spec_hard)`), que no retalla l'horitzó ni dona meteo: comprova que l'agent funciona igualment.
* **Novembre** sol (arrencada en fred) i latència per pas.
""")
code("""
final = {}
final["validacio 12 blocs (honesta)"] = T.simulate(art_val, val_df, best)
final["train complet (in-sample)"] = T.simulate(art_final, env_df)
ag = A.BatteryAgent(artifact=art_final)
r = run_agent_simulation(ag, env_df.drop(columns=list(HE.METEO_KEYS)), spec=SPEC_HARD); r["fallbacks"] = ag.n_fallback
final["entorn oficial sense modificar (train)"] = r
s, e = [b for b in blocks if b[0].month == 11][0]
final["nomes novembre (cold start)"] = T.simulate(art_val, val_df.loc[s:e], best)
ag = A.BatteryAgent(); ag_ok = ag.ready                      # carrega hm_build/model_bateria.pkl des de disc
final["submission (agent.py + pkl des de disc, in-sample)"] = {**T.run_hard(ag, val_df), "fallbacks": ag.n_fallback}
fin = pd.DataFrame(final).T
display(fin[[c for c in ["score", "co2_estalviat_pct", "cost_estalviat_pct", "infraccions", "fallbacks", "ms_per_step_mean", "ms_per_step_max"] if c in fin.columns]])
assert ag_ok and (fin["infraccions"] == 0).all() and (fin["fallbacks"] == 0).all()
print("OK: zero infraccions i zero passos de reserva en totes les simulacions.")
""")
code("""
# Puntuacio per bloc de validacio (mes a mes)
res, env = T.simulate(art_val, val_df, best, return_env=True)
ts = pd.DatetimeIndex(env.df["Fecha"]); rows = []
for s, e in blocks:
    m = np.asarray((ts >= s) & (ts <= e)); r = env.evaluate_results(mask=m)
    rows.append({"bloc": s.strftime("%Y-%m"), "score": r["score"], "co2_%": r["co2_estalviat_pct"], "cost_%": r["cost_estalviat_pct"], "infraccions": r["infraccions"]})
display(pd.DataFrame(rows).set_index("bloc").T)
try:
    import matplotlib.pyplot as plt
    i0 = int(np.flatnonzero(ts >= blocks[7][0])[0]); sl = slice(i0, i0 + 168)
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax[0].plot(ts[sl], env.df["load_kw"].values[sl] - env.df["pv_kw"].values[sl], label="consum net (kW)")
    ax[0].plot(ts[sl], np.array(env.actions)[sl], label="consigna bateria (kW)"); ax[0].axhline(0, c="k", lw=.5); ax[0].legend()
    ax[1].plot(ts[sl], np.array(env.soc_series)[sl], c="tab:green", label="SOC (kWh)"); ax2 = ax[1].twinx()
    ax2.plot(ts[sl], env.df["co2_kg_per_kwh"].values[sl], c="tab:red", alpha=.5, label="CO2"); ax[1].legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout(); plt.show()
except Exception as ex:
    print("sense grafic:", ex)
""")
md("## 10. Verificació del lliurament\nExecuta `check_submission.py` oficial si es troba (repo local); a Kaggle es fa una comprovació equivalent mínima.")
code("""
chk = [p for p in (Path("../starter_kit/check_submission.py"), Path("starter_kit/check_submission.py")) if p.exists()]
if chk:
    out = subprocess.run([sys.executable, str(chk[0].resolve()), str(Path("hm_build/submission.zip").resolve())], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(out.stdout[-1500:], out.stderr[-500:])
else:
    import ast
    bad = {"os", "sys", "shutil", "subprocess", "socket", "urllib", "requests", "http", "importlib", "inspect", "threading", "multiprocessing", "ctypes"}
    tree = ast.parse(Path("hm_build/agent.py").read_text(encoding="utf-8"))
    imps = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    print("imports:", sorted(imps), "| prohibits:", sorted(imps & bad))
print("Fitxers a lliurar: hm_build/submission.zip  (agent.py + model_bateria.pkl)")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = SRC / "train_pipeline.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("escrit", out, f"({len(cells)} cel.les)")

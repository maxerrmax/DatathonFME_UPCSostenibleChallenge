"""
agent.py — Repte UPC Sostenible (Datathon FME 2026) — MODE DIFICIL
==================================================================
Bateria 6.000 kWh / +-1.500 kW / eta 0.90, solar x12, nomes 3 hores de preu i CO2.

Pipeline a cada pas t (tot offline, sense xarxa):
  1. `self.historial`: buffer incremental indexat per hora absoluta amb load, pv(x12),
     irradiancia, temperatures, preu i CO2. Es pre-carrega amb l'historic de train
     (les setmanes de test son la continuacio immediata de cada mes de train) i amb
     priors Oct/Des per al novembre 2025 (sense telemetria a train).
  2. Prediccio a 24 h de consum, solar, preu i CO2 (LightGBM multi-horitzo; Ridge de reserva).
     Preu/CO2: valors exactes per a k=0..2, model per a k=3..23.
  3. Planificacio de 24 passos amb importacio G_k = max(0, net_k + P_k) >= 0: absorbeix
     l'excedent solar gratuit i mai descarrega per sobre del consum net (export = 0 EUR).
     Dos solvers equivalents: LP (scipy.linprog / HiGHS) i DP sobre graella de SOC (numpy,
     el doble de rapid; per defecte). Es tria amb params["solver"].
  4. Clamp fisic exacte -> zero infraccions.

Tambe conte els constructors de features que fa servir train_pipeline.ipynb, de manera
que les features offline i online son identiques per construccio.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None
try:
    from scipy.optimize import linprog
except Exception:  # pragma: no cover
    linprog = None
try:
    from scipy.special import ndtr
except Exception:  # pragma: no cover
    ndtr = None

_INV_SQRT_2PI = 0.3989422804014327


def expected_import(x, sigma):
    """E[max(x + e, 0)] amb e ~ N(0, sigma): importacio esperada quan el consum net previst
    es x i l'error de prediccio te desviacio sigma. Suavitza el colze de max(x, 0)."""
    z = x / sigma
    return x * ndtr(z) + sigma * _INV_SQRT_2PI * np.exp(-0.5 * z * z)


ARTIFACT_NAME = "model_bateria.pkl"

# ----------------------------------------------------------------------------------
# Eix temporal absolut (hores des de EPOCH)
# ----------------------------------------------------------------------------------
EPOCH = pd.Timestamp("2025-08-01 00:00:00")
END = pd.Timestamp("2027-01-01 00:00:00")
_EPOCH_H = EPOCH.value // 3_600_000_000_000
N_HOURS = int((END - EPOCH) // pd.Timedelta(hours=1))
SERIES = ("load", "pv", "irr", "tamb", "tpan", "price", "co2")
TELEMETRY = ("load", "pv", "irr", "tamb", "tpan")
HORIZON = 24
NAN = float("nan")


def hour_index(ts) -> int:
    """Hora absoluta (enter) d'un timestamp respecte EPOCH."""
    return int(pd.Timestamp(ts).value // 3_600_000_000_000 - _EPOCH_H)


def full_index() -> pd.DatetimeIndex:
    return pd.date_range(EPOCH, periods=N_HOURS, freq="1h")


# ----------------------------------------------------------------------------------
# Calendari Barcelona / UPC (nomes depen del timestamp)
# ----------------------------------------------------------------------------------
BCN_HOLIDAYS = [
    "2025-08-15", "2025-09-11", "2025-09-24", "2025-10-12", "2025-11-01", "2025-12-06",
    "2025-12-08", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-01-06", "2026-04-03", "2026-04-06", "2026-05-01", "2026-05-25",
    "2026-06-24", "2026-08-15", "2026-09-11", "2026-09-24", "2026-10-12", "2026-11-01",
    "2026-12-06", "2026-12-08", "2026-12-25", "2026-12-26",
]
TERM_BREAKS = [("2025-12-24", "2026-01-06"), ("2026-03-30", "2026-04-06"),
               ("2026-12-24", "2027-01-06")]
CAL_NAMES = ["hour", "dow", "h_sin", "h_cos", "d_sin", "d_cos", "y_sin", "y_cos",
             "is_weekend", "is_bcn_holiday", "is_august_recess", "is_exam_period",
             "is_datathon_weekend", "is_term_break", "is_offday"]
C_HSIN, C_HCOS = 2, 3


def calendar_matrix(index: pd.DatetimeIndex) -> np.ndarray:
    """Features ciclics + flags is_x, calculables nomes amb el timestamp."""
    hour = index.hour.values.astype(float)
    dow = index.dayofweek.values.astype(float)
    doy = index.dayofyear.values.astype(float)
    day = index.normalize()
    month = index.month.values
    dom = index.day.values
    out = np.zeros((len(index), len(CAL_NAMES)))
    out[:, 0] = hour
    out[:, 1] = dow
    out[:, 2] = np.sin(2 * np.pi * hour / 24.0)
    out[:, 3] = np.cos(2 * np.pi * hour / 24.0)
    out[:, 4] = np.sin(2 * np.pi * dow / 7.0)
    out[:, 5] = np.cos(2 * np.pi * dow / 7.0)
    out[:, 6] = np.sin(2 * np.pi * doy / 365.25)
    out[:, 7] = np.cos(2 * np.pi * doy / 365.25)
    out[:, 8] = dow >= 5
    out[:, 9] = day.isin(pd.to_datetime(BCN_HOLIDAYS))
    out[:, 10] = month == 8
    exam = ((month == 1) & (dom >= 8) & (dom <= 23)) | ((month == 5) & (dom >= 25)) | \
           ((month == 6) & (dom <= 20))
    out[:, 11] = exam
    out[:, 12] = (index.year.values == 2026) & (month == 11) & ((dom == 14) | (dom == 15))
    brk = np.zeros(len(index), dtype=bool)
    for a, b in TERM_BREAKS:
        brk |= (day >= pd.Timestamp(a)) & (day <= pd.Timestamp(b))
    out[:, 13] = brk
    out[:, 14] = (out[:, 8] > 0) | (out[:, 9] > 0) | brk
    return out


# ----------------------------------------------------------------------------------
# Constructors de features (compartits offline / online)
# ----------------------------------------------------------------------------------
def _stats(x):
    """mean, std, min, max ignorant NaN (sense warnings)."""
    v = x[x == x]
    if v.size == 0:
        return NAN, NAN, NAN, NAN
    return float(v.mean()), float(v.std()), float(v.min()), float(v.max())


def _mean(x):
    v = x[x == x]
    return float(v.mean()) if v.size else NAN


_K = np.arange(HORIZON)
_CALN = len(CAL_NAMES)

LOAD_FEATS = (["k"] + ["cal_" + c for c in CAL_NAMES] + ["t_hsin", "t_hcos"] +
              ["l1", "l2", "l3", "l_h24", "l_h48", "l_h168", "l_t25",
               "l_m24", "l_s24", "l_min24", "l_max24", "l_m168", "tamb1", "irr1", "pv1", "l_adj"])
PV_FEATS = (["k"] + ["cal_" + c for c in CAL_NAMES] + ["t_hsin", "t_hcos"] +
            ["pv1", "pv2", "pv3", "irr1", "irr2", "tpan1", "tamb1", "pv_h24", "pv_h48",
             "pv_h168", "irr_h24", "cs_h", "cs_t1", "kt1", "kt_day", "kt_h24",
             "pv_m24", "pv_max24", "pv_pers", "pv_day"])
MKT_FEATS = (["k"] + ["cal_" + c for c in CAL_NAMES] + ["t_hsin", "t_hcos"] +
             ["y0", "y1", "y2", "ym1", "y_h24", "y_h48", "y_h168", "d0", "d2",
              "y_m24", "y_s24", "y_min24", "y_max24", "y_m168",
              "o0", "o2", "o_h24", "irr1", "kt_day", "y_adj", "ym3", "ym6", "ym12", "slope",
              "o_slope"])


class Features:
    """Construeix les matrius (24, F) per a les hores objectiu t..t+23.

    Regla anti-leakage (verificada per tests): la telemetria (load, pv, meteo) nomes
    s'indexa fins a t-1 i el mercat (preu, CO2) nomes fins a t+2.
    """

    def __init__(self, hist: dict, cal: np.ndarray, cs: np.ndarray):
        self.h = hist      # dict nom -> array (N_HOURS,)
        self.cal = cal     # (N_HOURS, len(CAL_NAMES))
        self.cs = cs       # (N_HOURS,) envolupant solar de cel clar (kW, x12)

    def _base(self, t, n_extra):
        X = np.empty((HORIZON, 1 + _CALN + 2 + n_extra))
        X[:, 0] = _K
        X[:, 1:1 + _CALN] = self.cal[t:t + HORIZON]
        X[:, 1 + _CALN] = self.cal[t, C_HSIN]
        X[:, 2 + _CALN] = self.cal[t, C_HCOS]
        return X, 3 + _CALN

    def _kt_day(self, t):
        pv = self.h["pv"][t - 24:t]
        cs = self.cs[t - 24:t]
        m = pv == pv
        den = float(cs[m].sum()) if m.any() else 0.0
        return float(pv[m].sum()) / den if den > 100.0 else NAN

    def load(self, t):
        L = self.h["load"]
        X, j = self._base(t, 16)
        X[:, j] = L[t - 1]; X[:, j + 1] = L[t - 2]; X[:, j + 2] = L[t - 3]
        X[:, j + 3] = L[t - 24:t]
        X[:, j + 4] = L[t - 48:t - 24]
        X[:, j + 5] = L[t - 168:t - 144]
        X[:, j + 6] = L[t - 25]
        X[:, j + 7:j + 11] = _stats(L[t - 24:t])
        X[:, j + 11] = _mean(L[t - 168:t])
        X[:, j + 12] = self.h["tamb"][t - 1]
        X[:, j + 13] = self.h["irr"][t - 1]
        X[:, j + 14] = self.h["pv"][t - 1]
        X[:, j + 15] = L[t - 1] + L[t - 24:t] - L[t - 25]      # persistencia + forma d'ahir
        return X

    def pv(self, t):
        P = self.h["pv"]; I = self.h["irr"]; cs = self.cs
        X, j = self._base(t, 20)
        X[:, j] = P[t - 1]; X[:, j + 1] = P[t - 2]; X[:, j + 2] = P[t - 3]
        X[:, j + 3] = I[t - 1]; X[:, j + 4] = I[t - 2]
        X[:, j + 5] = self.h["tpan"][t - 1]
        X[:, j + 6] = self.h["tamb"][t - 1]
        X[:, j + 7] = P[t - 24:t]
        X[:, j + 8] = P[t - 48:t - 24]
        X[:, j + 9] = P[t - 168:t - 144]
        X[:, j + 10] = I[t - 24:t]
        X[:, j + 11] = cs[t:t + HORIZON]
        X[:, j + 12] = cs[t - 1]
        kt1 = P[t - 1] / cs[t - 1] if cs[t - 1] > 30.0 else NAN
        ktd = self._kt_day(t)
        X[:, j + 13] = kt1
        X[:, j + 14] = ktd
        X[:, j + 18] = (kt1 if kt1 == kt1 else ktd) * cs[t:t + HORIZON]   # persistencia de claredat
        X[:, j + 19] = ktd * cs[t:t + HORIZON]
        csy = cs[t - 24:t]
        with np.errstate(divide="ignore", invalid="ignore"):
            X[:, j + 15] = np.where(csy > 30.0, P[t - 24:t] / csy, NAN)
        st = _stats(P[t - 24:t])
        X[:, j + 16] = st[0]; X[:, j + 17] = st[3]
        return X

    def market(self, t, own, other):
        Y = self.h[own]; O = self.h[other]
        X, j = self._base(t, 25)
        X[:, j] = Y[t]; X[:, j + 1] = Y[t + 1]; X[:, j + 2] = Y[t + 2]; X[:, j + 3] = Y[t - 1]
        X[:, j + 4] = Y[t - 24:t]
        X[:, j + 5] = Y[t - 48:t - 24]
        X[:, j + 6] = Y[t - 168:t - 144]
        X[:, j + 7] = Y[t] - Y[t - 24]
        X[:, j + 8] = Y[t + 2] - Y[t - 22]
        X[:, j + 9:j + 13] = _stats(Y[t - 21:t + 3])
        X[:, j + 13] = _mean(Y[t - 165:t + 3])
        X[:, j + 14] = O[t]; X[:, j + 15] = O[t + 2]
        X[:, j + 16] = O[t - 24:t]
        X[:, j + 17] = self.h["irr"][t - 1]
        X[:, j + 18] = self._kt_day(t)
        X[:, j + 19] = Y[t + 2] + Y[t - 24:t] - Y[t - 22]       # ancoratge a l'ultim valor exacte
        X[:, j + 20] = Y[t - 3]; X[:, j + 21] = Y[t - 6]; X[:, j + 22] = Y[t - 12]
        X[:, j + 23] = Y[t + 2] - Y[t]
        X[:, j + 24] = O[t + 2] - O[t]
        return X

    def build(self, target, t):
        if target == "load":
            return self.load(t)
        if target == "pv":
            return self.pv(t)
        if target == "price":
            return self.market(t, "price", "co2")
        return self.market(t, "co2", "price")


FEATURE_NAMES = {"load": LOAD_FEATS, "pv": PV_FEATS, "price": MKT_FEATS, "co2": MKT_FEATS}


def clear_sky_series(cs_table: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    """cs_table (367, 24) -> serie horaria sobre `index`."""
    return cs_table[index.dayofyear.values, index.hour.values]


def seed_gap_priors(hist: dict, cal: np.ndarray):
    """Priors de cold-start a `self.historial`:
    * Novembre 2025 (sense comptador / inversor / meteo a train): perfil horari per tipus
      de dia (lectiu / no lectiu) del periode academic Oct + 1-21 Des 2025.
    * Solar de finals d'agost 2026 (l'inversor s'atura el 16/08): perfil horari d'agost.
    Nomes s'omplen hores sense dada; les mesures reals sobreescriuen el prior."""
    i = hour_index
    nov0, nov1 = i("2025-11-01"), i("2025-12-01")
    src = np.zeros(N_HOURS, dtype=bool)
    src[i("2025-10-01"):i("2025-11-01")] = True
    src[i("2025-12-01"):i("2025-12-22")] = True
    hour = cal[:, 0].astype(int)
    off = cal[:, 14] > 0
    for name in TELEMETRY:
        a = hist[name]
        if not np.isnan(a[nov0:nov1]).any():
            continue
        for o in (False, True):
            for h in range(24):
                m = src & (hour == h) & (off == o) & ~np.isnan(a)
                if not m.any():
                    m = src & (hour == h) & ~np.isnan(a)
                if not m.any():
                    continue
                tgt = np.zeros(N_HOURS, dtype=bool)
                tgt[nov0:nov1] = True
                tgt &= (hour == h) & (off == o) & np.isnan(a)
                a[tgt] = a[m].mean()
    a0, a1 = i("2026-08-01"), i("2026-09-01")
    for name in ("pv",):
        a = hist[name]
        for h in range(24):
            m = np.zeros(N_HOURS, dtype=bool)
            m[a0:a1] = True
            m &= (hour == h)
            known = m & ~np.isnan(a)
            if known.any():
                a[m & np.isnan(a)] = a[known].mean()


# ----------------------------------------------------------------------------------
# Models de prediccio
# ----------------------------------------------------------------------------------
# Grups d'horitzo k amb LightGBM. Mes enlla de k=11 nomes Ridge (mateixa precisio, 5x mes rapid).
BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 5), (6, 11))


class Forecaster:
    """Un objectiu = LightGBM per grup d'horitzo (k <= 11) + Ridge directe per horitzo.
    prefer="blend": mitjana LightGBM/Ridge on hi ha tots dos; "lgb": LightGBM i Ridge nomes
    on no hi ha arbres; "ridge": nomes Ridge (numpy pur, sense dependencies)."""

    def __init__(self, spec: dict, prefer: str = "blend", blend_w: float = 0.5):
        self.boosters = []
        self.has_ridge = False
        self.blend = float(blend_w) if prefer == "blend" else 0.0
        if prefer in ("lgb", "blend") and lgb is not None and spec.get("lgb"):
            try:
                for (k0, k1, txt) in spec["lgb"]:
                    self.boosters.append((int(k0), int(k1) + 1, lgb.Booster(model_str=txt)))
            except Exception:
                self.boosters = []
        if spec.get("ridge") is not None:
            r = spec["ridge"]
            self.W = np.asarray(r["W"], dtype=float)     # (24, F)
            self.b = np.asarray(r["b"], dtype=float)     # (24,)
            self.mu = np.asarray(r["mu"], dtype=float)
            self.sd = np.asarray(r["sd"], dtype=float)
            self.has_r = np.abs(self.W).sum(axis=1) > 0
            self.has_ridge = True
        self.kind = "lgb" if self.boosters else ("ridge" if self.has_ridge else None)

    def predict(self, X):
        """X: (24, F) -> prediccio (24,). Les k sense cap model queden a NaN."""
        out = np.full(X.shape[0], NAN)
        for k0, k1, bst in self.boosters:
            out[k0:k1] = bst.predict(X[k0:k1], num_threads=1)
        if self.has_ridge:
            Z = (np.where(X == X, X, self.mu) - self.mu) / self.sd
            r = np.where(self.has_r, (Z * self.W).sum(axis=1) + self.b, NAN)
            mix = np.where(r == r, (1.0 - self.blend) * out + self.blend * r, out)
            out = np.where(out == out, mix, r)
        return out


DEFAULT_PARAMS = {
    "lam_term": 0.95,          # V_term = lam * mean(w previst) per kWh emmagatzemat al final
    "lam_weekend": 1.0,        # factor multiplicatiu si l'hora terminal cau en dia no lectiu
    "lam_month": [1.0] * 12,   # factor per mes (regim estacional)
    "cap_q": 0.2,              # quantil (consum baix / solar alta) per al sostre de descarrega; 0 = off
    "buf_kw": 0.0,             # marge de seguretat d'exportacio a k=0 (kW)
    "buf_frac": 0.0,           # marge relatiu sobre el consum net previst
    "gamma": 1.0,              # descompte d'incertesa per a k>=3 (1 = sense descompte)
    "eps_thr": 1e-5,           # penalitzacio minima de throughput (evita carrega+descarrega)
    "co2_ref": 0.085,          # CO2 mitja ponderada del baseline (kg/kWh)
    "price_ref": 0.139,        # preu mitja ponderat del baseline (EUR/kWh)
    "sim_end": "2026-08-31 23:00:00",   # ultima hora del test: mes enlla el SOC no val res
    "horizon": 24,             # hores planificades (24..48); k>=24 = prediccio de k-24 esmorteida
    "gamma_far": 0.7,          # esmorteiment cap a la mitjana dels pesos w per a k>=24
    "w_co2": 0.7,              # pes del CO2 a l'objectiu del planificador (metrica oficial: 0.7)
    "sigma_steps": 6,          # nombre de passos (des de k=0) amb cost estocastic
    "sigma_scale": 0.6,        # >0: DP estocastic (multiplica la sigma d'error del consum net)
    "net_bias_kw": 0.0,        # biaix afegit al consum net previst per a k>=1 (kW)
    "solver": "dp",            # "dp" (graella de SOC, numpy) o "lp" (scipy.linprog / HiGHS)
    "dp_grid": 81,
    "prefer": "blend",         # "blend" (LightGBM+Ridge), "lgb" o "ridge"
}


class BatteryAgent:
    def __init__(self, artifact=None):
        """`artifact`: dict (per a validacio) o None -> carrega model_bateria.pkl del costat."""
        self.ready = False
        self.params = dict(DEFAULT_PARAMS)
        self.prev_idx = None
        self.cap = 6000.0
        self.pmax = 1500.0
        self.eta = 0.90 ** 0.5
        self.last = {}
        self.dropout = False
        self.net_lo = None
        self.n_fallback = 0        # passos resolts amb l'heuristica de reserva (diagnostic)
        try:
            if artifact is None:
                artifact = joblib.load(Path(__file__).parent / ARTIFACT_NAME)
            self._setup(artifact)
            self.ready = True
        except Exception:
            self.ready = False

    # ------------------------------------------------------------------ init
    def _setup(self, art):
        self.params.update(art.get("params", {}))
        idx = full_index()
        self.cal = calendar_matrix(idx)
        # Buffer d'estat incremental: historial[serie][hora_absoluta]
        self.historial = {}
        for name in SERIES:
            a = np.full(N_HOURS, NAN)
            src = np.asarray(art["hist"][name], dtype=float)
            a[:len(src)] = src
            self.historial[name] = a
        seed_gap_priors(self.historial, self.cal)
        self.cs = clear_sky_series(np.asarray(art["cs_table"], dtype=float), idx)
        self.feat = Features(self.historial, self.cal, self.cs)
        bw = self.params.get("blend_w", 0.5)
        if not isinstance(bw, dict):
            bw = {k: bw for k in ("load", "pv", "price", "co2")}
        self.models = {k: Forecaster(art["models"][k], self.params.get("prefer", "blend"),
                                     bw.get(k, 0.5))
                       for k in ("load", "pv", "price", "co2")}
        # Models quantil de k=0 per al sostre de descarrega (evita exportar energia a 0 EUR)
        self.cap_models = None
        try:
            cm = (art["models"].get("cap") or {}).get(float(self.params.get("cap_q", 0.0)))
            if cm and lgb is not None and self.params.get("prefer", "lgb") != "ridge":
                self.cap_models = (lgb.Booster(model_str=cm[0]), lgb.Booster(model_str=cm[1]))
        except Exception:
            self.cap_models = None
        self.irr_per_pv = float(art.get("irr_per_pv", 0.6))
        sg = art.get("sigma") or {}
        self.sig_load = np.asarray(sg["load"], dtype=float) if "load" in sg else None
        self.sig_pv = np.asarray(sg["pv_frac"], dtype=float) if "pv_frac" in sg else None
        if self.sig_pv is None:
            self.sig_load = None
        self.sim_end_idx = hour_index(self.params["sim_end"])
        self._lp_cache = {}

    # ------------------------------------------------------------------ estat
    def _update_history(self, t, obs):
        H = self.historial
        prices = [float(v) for v in obs.get("preus_properes_24h", [])][:HORIZON]
        co2s = [float(v) for v in obs.get("co2_properes_24h", [])][:HORIZON]
        if not prices:
            prices = [float(obs["preu_actual"])]
        if not co2s:
            co2s = [float(obs["co2_actual"])]
        prices[0] = float(obs.get("preu_actual", prices[0]))
        co2s[0] = float(obs.get("co2_actual", co2s[0]))
        m = min(len(prices), len(co2s))
        prices, co2s = prices[:m], co2s[:m]
        # No desem a l'historial valors mes enlla d'una frontera de bloc (canvi de mes o
        # inici dels darrers 7 dies): la fila seguent de l'entorn pot ser d'una altra data.
        ts = EPOCH + pd.Timedelta(hours=t)
        for k in range(m):
            tk = ts + pd.Timedelta(hours=k)
            if k > 0 and (tk.month != ts.month or
                          (tk.day > tk.days_in_month - 7) != (ts.day > ts.days_in_month - 7)):
                break
            if prices[k] == prices[k]:
                H["price"][t + k] = prices[k]
            if co2s[k] == co2s[k]:
                H["co2"][t + k] = co2s[k]

        # Telemetria de la darrera hora: correspon a la fila anterior de l'entorn (prev_idx).
        nd = obs.get("noves_dades_5min") or {}
        p = self.prev_idx
        self.dropout = False
        if p is not None and p != t and 0 <= p < N_HOURS:
            def val(key):
                v = nd.get(key)
                try:
                    v = float(v)
                except Exception:
                    return NAN
                return v
            load = val("ultim_consum_kw")
            pv = val("ultima_solar_kw")
            if load == load:
                if load > 1.0:
                    H["load"][p] = load
                else:
                    self.dropout = True          # 0.0 = caiguda del sensor del comptador
            if pv == pv:
                H["pv"][p] = max(0.0, pv)
            irr = val("irradiancia_wm2")
            if irr != irr and pv == pv:
                irr = max(0.0, pv) * self.irr_per_pv    # r(pv, irradiancia) = 0.997
            if irr == irr:
                H["irr"][p] = max(0.0, irr)
            ta = val("temp_ambient_c")
            if ta == ta:
                H["tamb"][p] = ta
            elif H["tamb"][p] != H["tamb"][p]:
                H["tamb"][p] = H["tamb"][p - 1]
            tp = val("temp_panel_c")
            if tp == tp:
                H["tpan"][p] = tp
            elif H["tpan"][p] != H["tpan"][p]:
                H["tpan"][p] = H["tpan"][p - 1]
        return prices, co2s

    # ------------------------------------------------------------------ prediccio
    def forecast(self, t, prices, co2s):
        f = self.feat
        Xl = f.load(t)
        Xp = f.pv(t)
        load, pv, price, co2 = self._predict_all(t, Xl, Xp)
        cs = self.cs[t:t + HORIZON]
        self.net_lo = None
        if self.cap_models is not None:
            lo = float(self.cap_models[0].predict(Xl[:1], num_threads=1)[0])
            hi = float(self.cap_models[1].predict(Xp[:1], num_threads=1)[0])
            hi = min(max(hi, 0.0), 1.2 * cs[0] + 20.0) if cs[0] > 1.0 else 0.0
            self.net_lo = lo - hi
        pv = np.where(cs > 1.0, np.clip(pv, 0.0, 1.2 * cs + 20.0), 0.0)
        load = np.maximum(load, 0.0)
        m = len(prices)
        price[:m] = prices
        co2[:m] = co2s
        if price[-1] != price[-1] or co2[-1] != co2[-1]:      # reserva: persistencia
            price = np.where(price == price, price, prices[-1])
            co2 = np.where(co2 == co2, co2, co2s[-1])
        price = np.maximum(price, 0.0)
        co2 = np.maximum(co2, 0.0)
        return load, pv, price, co2

    def _predict_all(self, t, Xl, Xp):
        """Prediccions crues (24,) de consum, solar, preu i CO2."""
        f = self.feat
        return (self.models["load"].predict(Xl), self.models["pv"].predict(Xp),
                self.models["price"].predict(f.market(t, "price", "co2")),
                self.models["co2"].predict(f.market(t, "co2", "price")))

    # ------------------------------------------------------------------ LP
    def _lp_matrices(self, n):
        if n in self._lp_cache:
            return self._lp_cache[n]
        I = np.eye(n)
        T = np.tril(np.ones((n, n)))
        A = np.zeros((3 * n, 3 * n))
        A[:n, :n] = I; A[:n, n:2 * n] = -I; A[:n, 2 * n:] = -I          # c - d - g <= -net
        A[n:2 * n, :n] = T * self.eta; A[n:2 * n, n:2 * n] = -T / self.eta     # soc <= cap
        A[2 * n:, :n] = -T * self.eta; A[2 * n:, n:2 * n] = T / self.eta       # soc >= 0
        self._lp_cache[n] = A
        return A

    def plan(self, soc, net, w, v_term, d_cap0):
        """LP de n passos. Variables: carrega c_k, descarrega d_k, importacio g_k >= 0."""
        n = len(net)
        A = self._lp_matrices(n)
        P = self.params
        cost = np.empty(3 * n)
        cost[:n] = P["eps_thr"] - v_term * self.eta
        cost[n:2 * n] = P["eps_thr"] + v_term / self.eta
        cost[2 * n:] = w
        b = np.empty(3 * n)
        b[:n] = -net
        b[n:2 * n] = self.cap - soc
        b[2 * n:] = soc
        ub = np.full(3 * n, np.inf)
        ub[:2 * n] = self.pmax
        ub[n] = min(self.pmax, d_cap0)
        res = linprog(cost, A_ub=A, b_ub=b, bounds=np.column_stack((np.zeros(3 * n), ub)),
                      method="highs")
        if res.status != 0:
            return None
        return float(res.x[0] - res.x[n])

    def plan_dp(self, soc, net, w, v_term, d_cap0, sigma=None):
        """Mateix problema que `plan`, resolt per programacio dinamica sobre una graella de
        SOC (numpy). Si es dona `sigma` (desviacio de l'error del consum net per pas), el cost
        de cada pas es la importacio ESPERADA w_k * E[max(net_k + e_k + P, 0)]: control
        estocastic conscient de l'error de prediccio (decideix quant arriscar-se a exportar
        en funcio del valor de l'energia a cada hora)."""
        n = len(net)
        stoch = sigma is not None and ndtr is not None
        G = int(self.params.get("dp_grid", 81))
        key = ("dp", G, self.cap, self.pmax)
        if key not in self._lp_cache:
            grid = np.linspace(0.0, self.cap, G)
            D = grid[None, :] - grid[:, None]                    # s -> s'
            Pm = np.where(D >= 0, D / self.eta, D * self.eta)    # potencia necessaria (kW)
            pen = np.where(np.abs(Pm) <= self.pmax + 1e-9, 0.0, np.inf)
            self._lp_cache[key] = (grid, Pm, pen)
        grid, Pm, pen = self._lp_cache[key]
        eps = self.params["eps_thr"]
        V = -v_term * grid
        for k in range(n - 1, 0, -1):
            if stoch and sigma[k] > 1.0:
                C = expected_import(Pm + net[k], sigma[k])
            else:
                C = np.maximum(Pm + net[k], 0.0)
            C *= w[k]
            C += pen
            C += V[None, :]
            V = C.min(axis=1)
        lo = -min(self.pmax, soc * self.eta, d_cap0)
        hi = min(self.pmax, (self.cap - soc) / self.eta)
        ds = grid - soc
        cand = np.concatenate((np.where(ds >= 0, ds / self.eta, ds * self.eta),
                               (0.0, -net[0], lo, hi)))
        if stoch and sigma[0] > 1.0:
            cand = np.concatenate((cand, np.linspace(lo, hi, 61)))
        cand = np.clip(cand, lo, hi)
        s1 = soc + np.where(cand >= 0, cand * self.eta, cand / self.eta)
        if stoch and sigma[0] > 1.0:
            imp0 = expected_import(net[0] + cand, sigma[0])
        else:
            imp0 = np.maximum(net[0] + cand, 0.0)
        val = w[0] * imp0 + np.interp(s1, grid, V) + eps * np.abs(cand)
        return float(cand[int(np.argmin(val))])

    def decide(self, t, soc, prices, co2s):
        P = self.params
        load, pv, price, co2 = self.forecast(t, prices, co2s)
        if self.dropout:
            load[0] = 0.0
        net = load - pv
        if P["net_bias_kw"] != 0.0:
            net[1:] += P["net_bias_kw"]
        a_co2 = float(P["w_co2"])
        w = a_co2 * co2 / P["co2_ref"] + (1.0 - a_co2) * price / P["price_ref"]
        m = len(prices)
        if P["gamma"] != 1.0 and m < HORIZON:
            wbar = float(w.mean())
            w[m:] = wbar + P["gamma"] * (w[m:] - wbar)
        # Horitzo ampliat (fins a 48 h): les hores k>=24 repeteixen la prediccio de k-24
        # (mateixa hora de l'endema) amb els pesos esmorteits cap a la mitjana.
        Hn = int(min(max(P.get("horizon", HORIZON), HORIZON), 2 * HORIZON))
        if Hn > HORIZON:
            net, w = self._extend_horizon(t, net, w, Hn)
        # Horitzo retallat al final conegut de la simulacio.
        n = Hn
        remaining = self.sim_end_idx - t + 1
        v_scale = 1.0
        if 0 < remaining <= Hn:
            n = max(1, remaining)
            v_scale = 0.0
        if m < 3:                      # l'entorn s'esta acabant (llistes retallades)
            n = max(1, m)
            v_scale = 0.0
        te = min(t + min(n, HORIZON), N_HOURS - 1)
        lam = P["lam_term"] * P["lam_month"][int(self.month[te]) - 1]
        if self.cal[te, 14] > 0:
            lam *= P["lam_weekend"]
        v_term = v_scale * lam * float(w[:HORIZON].mean())
        net_lo = net[0]
        if self.net_lo is not None and not self.dropout:
            net_lo = min(net_lo, self.net_lo)
        d_cap0 = max(0.0, net_lo * (1.0 - P["buf_frac"]) - P["buf_kw"])
        self.last = {"load": load, "pv": pv, "price": price, "co2": co2, "w": w}
        a = None
        if self.params.get("solver", "dp") == "lp" and linprog is not None:
            try:
                a = self.plan(soc, net[:n], w[:n], v_term, d_cap0)
            except Exception:
                a = None
        if a is None or a != a:
            a = self.plan_dp(soc, net[:n], w[:n], v_term, d_cap0, self._sigma(t, n))
        return a

    def _sigma(self, t, n):
        """Desviacio de l'error del consum net per pas: sqrt(sig_load^2 + (sig_pv * cs)^2)."""
        sc = float(self.params.get("sigma_scale", 0.0))
        if sc <= 0.0 or self.sig_load is None:
            return None
        k = np.minimum(np.arange(n), HORIZON - 1)
        cs = self.cs[t:t + n]
        if len(cs) < n:
            return None
        s = np.sqrt(self.sig_load[k] ** 2 + (self.sig_pv[k] * cs) ** 2) * sc
        s[int(self.params.get("sigma_steps", 6)):] = 0.0     # mes enlla: cost determinista
        if self.dropout:
            s[0] = 0.0
        return s

    def _extend_horizon(self, t, net, w, Hn):
        extra = Hn - HORIZON
        wbar = float(w.mean())
        w2 = wbar + self.params["gamma_far"] * (w[:extra] - wbar)
        return np.concatenate((net, net[:extra])), np.concatenate((w, w2))

    @property
    def month(self):
        mth = getattr(self, "_month", None)
        if mth is None:
            mth = full_index().month.values
            self._month = mth
        return mth

    # ------------------------------------------------------------------ fallback
    @staticmethod
    def _heuristic(obs):
        hora = int(obs.get("hour", 0))
        nd = obs.get("noves_dades_5min") or {}
        try:
            net = float(nd.get("ultim_consum_kw", 0.0)) - float(nd.get("ultima_solar_kw", 0.0))
        except Exception:
            net = 0.0
        if 10 <= hora <= 15:
            return 1000.0
        if 18 <= hora <= 22:
            return -max(0.0, 0.8 * net)
        return 0.0

    # ------------------------------------------------------------------ act
    def act(self, obs: dict) -> float:
        consigna_desitjada = 0.0
        try:
            soc_kwh = float(obs["soc_kwh"])
            cap = obs.get("cap_max_kwh")
            if cap is None:
                # L'entorn oficial no informa la capacitat: es dedueix de soc_kwh / soc_pct i
                # s'arrodoneix (l'error de coma flotant donaria p_max = 1500.0000000000002).
                pct = float(obs.get("soc_pct", 0.0) or 0.0)
                cap = round(soc_kwh / pct) if (pct > 1e-9 and soc_kwh > 1e-6) else self.cap
                if not (100.0 <= cap <= 1e6):
                    cap = self.cap
            self.cap = float(cap)
            self.pmax = float(obs.get("power_max_kw", self.cap / 4.0))
        except Exception:
            soc_kwh = float(obs.get("soc_kwh", 0.0))
        try:
            if not self.ready:
                raise RuntimeError("sense artefacte")
            t = hour_index(obs["timestamp"])
            if t < 200 or t + HORIZON + 2 >= N_HOURS:
                raise RuntimeError("timestamp fora de rang")
            prices, co2s = self._update_history(t, obs)
            self.prev_idx = t
            a = self.decide(t, soc_kwh, prices, co2s)
            if a is None or a != a:
                raise RuntimeError("LP sense solucio")
            consigna_desitjada = a
        except Exception:
            self.n_fallback += 1
            try:
                consigna_desitjada = self._heuristic(obs)
            except Exception:
                consigna_desitjada = 0.0

        # Clamp fisic exacte (zero infraccions)
        cap_max = float(self.cap)
        p_max = float(self.pmax)
        eta = 0.90 ** 0.5
        eps = 1e-4
        p_max = p_max - 1e-6                      # marge numeric sobre el limit de l'inversor
        max_carrega = max(0.0, min(p_max, (cap_max - soc_kwh) / eta - eps))
        max_descarrega = max(0.0, min(p_max, soc_kwh * eta - eps))
        consigna_segura = float(max(-max_descarrega, min(max_carrega, consigna_desitjada)))
        if consigna_segura != consigna_segura:
            consigna_segura = 0.0
        return consigna_segura

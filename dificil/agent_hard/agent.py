"""
BatteryAgent — Mode Difícil (6.000 kWh, ±1.500 kW, solar ×12, mercat a 3 h).

Cada hora:
  1. Actualitza l'historial (preu/CO2 de la finestra de 3 h; consum, solar i meteo de l'hora anterior).
  2. Preveu 24 h amb LightGBM: preu i CO2 (+3..+23 h; les 3 primeres són exactes),
     consum i solar (+0..+23 h). Consum/preu: lag24 + residu; solar: mescla model + persistència.
  3. Resol un LP: min sum w_t * imp_t  amb  imp_t >= net_t + c_t - d_t, imp_t >= 0
     (l'excedent solar és gratuït i el que es descarrega de més es perd), més valor terminal.
  4. Aplica la primera acció amb un marge de prudència a la descàrrega i el clamp físic.
Blindat: qualsevol error -> regla simple / repòs, sortida sempre dins dels límits.
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog

try:
    import joblib, lightgbm as lgb
except Exception:  # pragma: no cover
    joblib = lgb = None

from feats import Hist, mkt_features, mkt_features2, load_features2, meas_features, set_cs, cs_value

ETA = 0.9 ** 0.5
HERE = Path(__file__).parent


def _num(x, default=float("nan")):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _epoch_hour(ts, fallback):
    try:
        if hasattr(ts, "value"):                       # pandas.Timestamp
            return int(ts.value // 3_600_000_000_000)
        return int(np.datetime64(str(ts)[:19], "h").astype(np.int64))
    except Exception:
        return fallback


class BatteryAgent:
    def __init__(self, params=None):
        p = {"ratio": 0.6213, "q_term": 0.6, "term_scale": 1.0, "k_dis": 0.5, "pv_blend": 0.3,
             "horizon": 24, "k_future": 0.0, "sigma_mode": "global",
             "models": "models_all.pkl"}
        try:
            f = HERE / "parametres.json"
            if f.exists():
                p.update(json.loads(f.read_text()))
        except Exception:
            pass
        if params:
            p.update(params)
        self.p = p
        self.m = {}
        self.mk2 = {}
        self.ld2 = []
        self.n0 = None
        self.cbr = None
        self.mkg = {}
        try:
            raw = p.get("_models")
            if raw is None:
                raw = joblib.load(HERE / p["models"])
            # LightGBM es desa com a text; la resta (XGBoost, sklearn, ensembles) com a objectes amb .predict
            raw = dict(raw)
            set_cs(raw.pop("pv_cs", None))
            self._h0 = raw.pop("hist0", None)
            mk2 = raw.pop("mkt2", None) or {}
            self.mk2 = {k: [(lgb.Booster(model_str=ms), list(cols), base) for ms, cols, base in v] for k, v in mk2.items()}
            self.ld2 = [(lgb.Booster(model_str=ms), list(cols), base) for ms, cols, base in (raw.pop("load2", None) or [])]
            self.mkg = {k: [(a, b_, lgb.Booster(model_str=ms), list(cols), kind) for a, b_, ms, cols, kind in v]
                        for k, v in (raw.pop("mktg", None) or {}).items()}
            cb = raw.pop("cbr", None)                       # base de casos (CBR) del CO2
            self.cbr = (np.asarray(cb[0], np.float32), np.asarray(cb[1], np.float32)) if cb is not None else None
            n0 = raw.pop("net0", None)
            self.n0 = lgb.Booster(model_str=n0) if n0 else None
            self.m = {k: (lgb.Booster(model_str=v) if isinstance(v, str) else v) for k, v in raw.items()}
        except Exception:
            self.m = {}
        self.hist = Hist()
        # historial inicial = dades d'entrenament (el dia abans de cada setmana de test és al train):
        # així els lags de 24/48/168 h existeixen des de la primera hora de cada setmana
        try:
            h0 = p.get("_hist0", getattr(self, "_h0", None)) if p.get("prefill", True) else None
            if h0:
                for k, (hh, vv) in h0.items():
                    for e, v in zip(hh, vv):
                        self.hist.put(k, e, v)
        except Exception:
            pass
        self.t_prev = None
        self.err = []                                   # errors recents de previsió de net (t-1)
        self.pred_net0 = None
        self.hist_w = []
        self.cap, self.pmax = 6000.0, 1500.0
        self.max_len = 0                                # llargada màxima vista de la finestra de mercat
        self._co2_raw_forecasts = {}                     # prediccions CAUSALS de mercat anteriors


    # ------------------------------------------------------------------ previsions
    def _pred(self, key, X, kind):
        if self.p.get("no168"):
            X[:, 7] = np.nan                            # el lag setmanal generalitza malament
        lag = X[:, 5]
        if not self.m:
            out = lag.copy()
        else:
            d = self.m[key + "_dir"].predict(X, num_threads=1)
            r = lag + self.m[key + "_res"].predict(X, num_threads=1)
            out = np.where(np.isfinite(lag), r, d) if kind == "res" else d
        if key == "pv":
            last, last24 = X[:, 8], X[:, 8] - X[:, 9]
            ratio = np.clip(np.where(np.isfinite(last24) & (last24 > 50), last / np.maximum(last24, 1), 1.0), 0.3, 1.5)
            pers = lag * (1 + (ratio - 1) * np.exp(-X[:, 0] / 6.0))
            csp = X[:, -1]                                   # k_last * cel_clar(tau)
            if self.p.get("pv_cs", True):
                wcs = np.exp(-X[:, 0] / self.p.get("cs_tau", 3.0))
                pers = np.where(np.isfinite(csp), wcs * csp + (1 - wcs) * np.where(np.isfinite(pers), pers, csp), pers)
            a = self.p["pv_blend"]
            out = np.where(np.isfinite(pers), a * out + (1 - a) * pers, out)
            hod = X[:, 1]
            out = np.where((hod < 6) | (hod > 21), 0.0, np.maximum(out, 0.0))
        return np.where(np.isfinite(out), out, np.nan)

    def _pred_mkt2(self, key, t, hs):
        """Mitjana de models residuals (respecte a l'últim valor conegut o al perfil d'ahir desplaçat)."""
        X = mkt_features2(self.hist, key, t, hs)
        k2 = X[:, 7]; xs = np.where(np.isfinite(X[:, 13]), X[:, 13], k2)
        out = []
        xsl = np.where(np.isfinite(X[:, 13]) & (X[:, 13] > 0.3 * k2), X[:, 13], k2)
        for b, cols, base in self.mk2[key]:
            pr = b.predict(X[:, cols], num_threads=1)
            if base == "logk2":                          # model del quocient (escala lliure)
                out.append(np.where(k2 > 0, k2 * np.exp(pr), k2))
            elif base == "logxs":
                out.append(np.where(xsl > 0, xsl * np.exp(pr), xsl))
            else:
                out.append((xs if base == "xs" else k2) + pr)
        res = np.mean(out, axis=0)
        # models especialitzats per tram d'horitzó (+3..+5, +6..+11 h), barrejats amb el general
        gm = self.p.get("grp_mix", 0.0)
        if gm > 0 and key in self.p.get("grp_keys", ["price", "co2"]):
            hs_a = np.asarray(hs)
            for a, b_, bst, cols, kind in self.mkg.get(key, []):
                sel = (hs_a >= a) & (hs_a <= b_)
                if sel.any():
                    pr = bst.predict(X[sel][:, cols], num_threads=1)
                    v = np.where(k2[sel] > 0, k2[sel] * np.exp(pr), k2[sel]) if kind == "log" else k2[sel] + pr
                    res[sel] = np.where(np.isfinite(v), gm * v + (1 - gm) * res[sel], res[sel])
        return res

    def _pred_load2(self, t, hs):
        """Consum: perfil d'ahir desplaçat al nivell actual (hores properes) barrejat amb lag24 + residu (hores llunyanes)."""
        X = load_features2(self.hist, t, hs)
        def fb(*idx):
            b = X[:, idx[0]]
            for i in idx[1:]:
                b = np.where(np.isfinite(b), b, X[:, i])
            return np.where(np.isfinite(b), b, 1000.0)
        bases = {"lag24": fb(9, 5, 15), "xs": fb(13, 9, 5, 15)}
        pr = {base: bases[base] + b.predict(X[:, cols], num_threads=1) for b, cols, base in self.ld2}
        w = np.exp(-X[:, 0] / self.p.get("load_tau", 4.0))
        return w * pr["xs"] + (1 - w) * pr["lag24"]

    def _cbr_co2(self, t):
        """Raonament basat en casos: els k dies més semblants (forma de les últimes 24 h de CO2 i preu, hora, tipus de
        dia, estació i nivell) i la seva continuació, adaptada a l'últim valor conegut."""
        try:
            g = self.hist.get
            k2 = g("co2", t + 2); p2 = g("price", t + 2)
            if not (math.isfinite(k2) and k2 > 0 and math.isfinite(p2) and p2 > 0):
                return None
            c = np.array([g("co2", t + 2 - i) for i in range(27)]); p = np.array([g("price", t + 2 - i) for i in range(27)])
            if np.isnan(c).sum() > 3:
                return None
            c = np.nan_to_num(np.log(np.maximum(c, 1e-4) / k2)); p = np.nan_to_num(np.log(np.maximum(p, 1e-4) / p2))
            ts = pd.Timestamp(int(t) * 3600, unit="s"); h = ts.hour; wk = 1.0 if ts.dayofweek >= 5 else 0.0
            doy = ts.dayofyear / 365.25 * 2 * np.pi
            q = np.concatenate([c * 3, p, [np.sin(2 * np.pi * h / 24) * 3, np.cos(2 * np.pi * h / 24) * 3, wk * 2,
                                            np.sin(doy), np.cos(doy), np.log(k2) * 2]]).astype(np.float32)
            D, F = self.cbr; k = int(self.p.get("cbr_k", 15))
            d = ((D - q) ** 2).sum(1); nn = np.argpartition(d, k)[:k]; w = 1.0 / (np.sqrt(d[nn]) + 1e-3)
            return k2 * np.exp((F[nn] * w[:, None]).sum(0) / w.sum())
        except Exception:
            return None

    def _forecast(self, t, price3, co2_3):
        H = 24
        price = np.full(H, np.nan); co2 = np.full(H, np.nan)
        price[:len(price3)] = price3; co2[:len(co2_3)] = co2_3
        hs = list(range(len(price3), H))
        if hs:
            if self.mk2:
                price[hs] = self._pred_mkt2("price", t, hs)
                co2[hs] = self._pred_mkt2("co2", t, hs)
                wc = self.p.get("cbr_w", 0.0)
                if self.cbr is not None and wc > 0 and len(price3) >= 3:
                    pc = self._cbr_co2(t)
                    if pc is not None:
                        for h in hs:
                            if 3 <= h <= 23:
                                co2[h] = (1 - wc) * co2[h] + wc * pc[h - 3]
            else:
                price[hs] = self._pred("price", mkt_features(self.hist, "price", t, hs), "res")
                co2[hs] = self._pred("co2", mkt_features(self.hist, "co2", t, hs), "dir")
        hs_all = list(range(H))
        if self.ld2:
            load = self._pred_load2(t, hs_all)
        else:
            load = self._pred("load", meas_features(self.hist, "load", t, hs_all), "res")
        pv = self._pred("pv", meas_features(self.hist, "pv", t, hs_all), "dir")
        # hora actual: model directe del consum net (consum - solar), barrejat amb la previsió per separat
        mix = self.p.get("net0_mix", 0.5)
        if self.n0 is not None and mix > 0:
            try:
                x0 = np.concatenate([load_features2(self.hist, t, [0])[0], meas_features(self.hist, "pv", t, [0])[0]])
                nd = float(self.n0.predict(x0.reshape(1, -1), num_threads=1)[0])
                if np.isfinite(nd) and np.isfinite(load[0]) and np.isfinite(pv[0]):
                    load[0] = mix * nd + (1 - mix) * (load[0] - pv[0]) + pv[0]
            except Exception:
                pass
        def fill(a, default):
            a = np.asarray(a, float)
            if not np.isfinite(a).any():
                return np.full(len(a), default)
            idx = np.arange(len(a)); ok = np.isfinite(a)
            return np.interp(idx, idx[ok], a[ok])
        price = fill(price, price3[0] if len(price3) else 0.12)
        co2 = fill(co2, co2_3[0] if len(co2_3) else 0.10)
        load = fill(load, 1100.0); pv = fill(pv, 0.0)
        cheat = self.p.get("_cheat")                    # NOMÉS diagnòstic local (mai al lliurament)
        if cheat:
            tbl, keys = cheat
            for k, arr in (("price", price), ("co2", co2), ("load", load), ("pv", pv)):
                if k in keys:
                    for h in range(H):
                        v = tbl[k].get(t + h)
                        if v is not None:
                            arr[h] = v
        # Correcció causal en línia de CO2. A t, ja coneixem exactament t,t+1,t+2;
        # comparem aquests tres valors amb la previsió crua feta a t-3 (h=3..5).
        # Només es fan servir dades conegudes en l'instant t, mai dades del futur.
        try:
            old = self._co2_raw_forecasts.get(t - 3)
            raw_co2 = co2.copy()
            # Al final de la simulació la finestra coneguda pot tenir només 1 o 2 hores.
            n_exact = min(3, len(co2_3))
            if old is not None and n_exact > 0 and len(old) >= 3 + n_exact and self.p.get("co2_cal", True):
                e = np.asarray(co2[:n_exact] - old[3:3+n_exact], dtype=float)
                if np.isfinite(e).all():
                    bias = float(np.clip(np.median(e), -0.06, 0.06))
                    h = np.arange(3, len(co2))
                    co2[3:] = np.clip(co2[3:] + bias * np.exp(-(h - 3) / 8.0), 0.0, 1.0)
            self._co2_raw_forecasts[t] = raw_co2
            # memòria acotada i segura si el timestamp salta entre blocs
            for key in list(self._co2_raw_forecasts):
                if key < t - 6:
                    del self._co2_raw_forecasts[key]
        except Exception:
            pass
        return price, co2, load, pv

    # ------------------------------------------------------------------ optimització
    def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
        H = len(w); cap, pm = self.cap, self.pmax
        K = 0 if scen is None else len(scen)
        L = np.tril(np.ones((H, H)))
        Asoc = np.hstack([L * ETA, -L / ETA, np.zeros((H, H + K))])      # SOC_k - soc0
        # net + c - d - imp <= 0
        Aimp = np.hstack([np.eye(H), -np.eye(H), -np.eye(H), np.zeros((H, K))])
        A = np.vstack([Asoc, -Asoc, Aimp])
        b = np.concatenate([np.full(H, cap - 1e-3 - soc0), np.full(H, soc0 - 1e-3), -net])
        cost = np.concatenate([np.zeros(2 * H), w, np.zeros(K)]) - v_term * Asoc[-1]
        bounds = [(0, pm)] * H + [(0, pm)] * H + [(0, None)] * (H + K)
        bounds[H] = (0, max(0.0, min(pm, dmax0)))
        if K:
            # hora actual estocàstica: cost esperat de la importació sobre K escenaris del consum net
            # (el que es descarrega de més es perd; el que es carrega de més s'importa)
            S = np.zeros((K, 3 * H + K)); S[:, 0] = 1.0; S[:, H] = -1.0
            S[np.arange(K), 3 * H + np.arange(K)] = -1.0
            A = np.vstack([A, S]); b = np.concatenate([b, -(net[0] + np.asarray(scen, float))])
            cost[2 * H] = 0.0; cost[3 * H:] = w[0] / K
            b[2 * H] = 1e9                                   # la restricció determinista de l'hora 0 queda inactiva
        r = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if r.status != 0 or r.x is None:
            return None
        return r.x[:H] - r.x[H:2 * H]

    def _lp_saa(self, W, net, soc0, v_term, w0, scen0):
        """MPC estocàstic de dues etapes: acció de l'hora 0 comuna; hores 1..H-1 adaptades a cada escenari de mercat.
        W: K x H (pesos per escenari; la columna 0 no s'usa). scen0: escenaris d'error del consum net a l'hora 0."""
        from scipy.sparse import lil_matrix
        K, H = W.shape; cap, pm = self.cap, self.pmax; n1 = H - 1
        J = 1 if scen0 is None else len(scen0)
        e0 = np.zeros(1) if scen0 is None else np.asarray(scen0, float)
        nv = 2 + J + K * 3 * n1
        def ix(k, part, h):                                    # part 0=c 1=d 2=imp, h=1..H-1
            return 2 + J + k * 3 * n1 + part * n1 + (h - 1)
        rows = 2 * J + 2 + K * n1 * 3
        A = lil_matrix((rows, nv)); b = np.zeros(rows); r = 0
        for j in range(J):                                     # c0 - d0 - imp0_j <= -(net0 + e_j)
            A[r, 0] = 1; A[r, 1] = -1; A[r, 2 + j] = -1; b[r] = -(net[0] + e0[j]); r += 1
        A[r, 0] = ETA; A[r, 1] = -1 / ETA; b[r] = cap - 1e-3 - soc0; r += 1
        A[r, 0] = -ETA; A[r, 1] = 1 / ETA; b[r] = soc0 - 1e-3; r += 1
        cost = np.zeros(nv); cost[2:2 + J] = w0 / J
        for k in range(K):
            for h in range(1, H):
                # SOC_k(h) <= cap  i  >= 0
                A[r, 0] = ETA; A[r, 1] = -1 / ETA
                for i in range(1, h + 1):
                    A[r, ix(k, 0, i)] = ETA; A[r, ix(k, 1, i)] = -1 / ETA
                b[r] = cap - 1e-3 - soc0; r += 1
                A[r, 0] = -ETA; A[r, 1] = 1 / ETA
                for i in range(1, h + 1):
                    A[r, ix(k, 0, i)] = -ETA; A[r, ix(k, 1, i)] = 1 / ETA
                b[r] = soc0 - 1e-3; r += 1
                A[r, ix(k, 0, h)] = 1; A[r, ix(k, 1, h)] = -1; A[r, ix(k, 2, h)] = -1; b[r] = -net[h]; r += 1
                cost[ix(k, 2, h)] = W[k, h] / K
            # valor terminal de l'energia final de cada escenari
            cost[0] -= v_term * ETA / K; cost[1] += v_term / ETA / K
            for i in range(1, H):
                cost[ix(k, 0, i)] -= v_term * ETA / K; cost[ix(k, 1, i)] += v_term / ETA / K
        bounds = [(0, pm), (0, pm)] + [(0, None)] * J + ([(0, pm)] * n1 + [(0, pm)] * n1 + [(0, None)] * n1) * K
        res = linprog(cost, A_ub=A.tocsr(), b_ub=b, bounds=bounds, method="highs")
        if res.status != 0 or res.x is None:
            return None
        return np.array([res.x[0] - res.x[1]])

    def _clamp(self, soc, a):
        mc = max(0.0, min(self.pmax, (self.cap - soc) / ETA - 1e-3))
        md = max(0.0, min(self.pmax, soc * ETA - 1e-3))
        a = max(-md, min(mc, a))
        return 0.0 if (not math.isfinite(a) or abs(a) < 1e-6) else float(a)

    # ------------------------------------------------------------------ pas
    def _segment(self, t, obs):
        """(hores que queden de la setmana de test, primera hora de la setmana següent) o None.
        El simulador enganxa les setmanes de test (últims 7 dies de cada mes) en ordre cronològic."""
        try:
            st = self.p.get("_seg")
            if st is not None:
                return st.get(t)
            if not self.p.get("jump", False):
                return None
            ts = pd.Timestamp(obs.get("timestamp"))
            end = (ts + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(hours=23)
            if ts < end - pd.Timedelta(days=6, hours=23):
                return None
            rem = int(round((end - ts) / pd.Timedelta(hours=1))) + 1
            nxt = (end.normalize() + pd.Timedelta(days=1) + pd.offsets.MonthEnd(0)).normalize() - pd.Timedelta(days=6)
            tn = _epoch_hour(nxt, None)
            if tn is None or not math.isfinite(self.hist.get("price", tn - 1)):
                return None                              # no hi ha setmana següent coneguda
            return rem, tn
        except Exception:
            return None

    def _jump_fill(self, t, seg, price, co2, load, pvf, nk):
        """Si la setmana s'acaba dins l'horitzó, les hores de després són l'inici de la setmana següent:
        es preveuen des de l'hora anterior a aquella setmana (dades d'entrenament), no amb hores inexistents."""
        if seg is None:
            return price, co2, load, pvf
        rem, tn = seg
        if rem >= len(price):
            return price, co2, load, pvf
        try:
            tv = tn - 1
            kp, kc = [], []
            for i in range(3):
                a, b = self.hist.get("price", tv + i), self.hist.get("co2", tv + i)
                if not (math.isfinite(a) and math.isfinite(b)):
                    break
                kp.append(a); kc.append(b)
            if not kp:
                return price, co2, load, pvf
            saved = dict(self._co2_raw_forecasts) if hasattr(self, "_co2_raw_forecasts") else None
            vp, vc, vl, vv = self._forecast(tv, kp, kc)
            if saved is not None:
                self._co2_raw_forecasts = saved
            for h in range(max(rem, nk), len(price)):
                j = h - rem + 1
                if j < len(vp):
                    price[h], co2[h], load[h], pvf[h] = vp[j], vc[j], vl[j], vv[j]
        except Exception:
            pass
        return price, co2, load, pvf

    def act(self, obs):
        try:
            a = self._act(obs)
            im = self.p.get("idle_months")              # NOMÉS diagnòstic: no actuar en aquests mesos
            if im:
                try:
                    if pd.Timestamp(obs.get("timestamp")).month in im:
                        return 0.0
                except Exception:
                    pass
            return a
        except Exception:
            try:
                return self._clamp(_num((obs or {}).get("soc_kwh"), 0.0), 0.0)
            except Exception:
                return 0.0

    def _act(self, obs):
        # sense cap_max_kwh/power_max_kw a obs -> Mode Estàndard (2.000 kWh, ±500 kW)
        self.cap = _num(obs.get("cap_max_kwh"), 2000.0); self.pmax = _num(obs.get("power_max_kw"), 500.0)
        soc = min(self.cap, max(0.0, _num(obs.get("soc_kwh"), self.cap / 2)))
        t = _epoch_hour(obs.get("timestamp"), (self.t_prev + 1) if self.t_prev is not None else 0)
        tp = self.t_prev
        self.t_prev = t
        # després d'un salt entre setmanes, "l'última hora" de l'observació és l'última hora de la setmana anterior
        last = t - 1 if (tp is None or tp == t - 1 or not self.p.get("jump_fix", False)) else tp
        # historial
        pr = [_num(x) for x in (obs.get("preus_properes_24h") or [])]
        cr = [_num(x) for x in (obs.get("co2_properes_24h") or [])]
        if not pr or not math.isfinite(pr[0]):
            pr = [_num(obs.get("preu_actual"), self.hist.get("price", t - 1))] + pr[1:]
        if not cr or not math.isfinite(cr[0]):
            cr = [_num(obs.get("co2_actual"), self.hist.get("co2", t - 1))] + cr[1:]
        for i, v in enumerate(pr):
            self.hist.put("price", t + i, v)
        for i, v in enumerate(cr):
            self.hist.put("co2", t + i, v)
        nd = obs.get("noves_dades_5min") or {}
        ld, pv = _num(nd.get("ultim_consum_kw")), _num(nd.get("ultima_solar_kw"))
        self.hist.put("load", last, ld); self.hist.put("pv", last, pv)
        self.hist.put("irr", last, _num(nd.get("irradiancia_wm2"))); self.hist.put("temp", last, _num(nd.get("temp_ambient_c")))
        self.hist.put("vent", last, _num(nd.get("vent_ms")))
        if self.pred_net0 is not None and math.isfinite(ld) and math.isfinite(pv):
            self.err.append((last % 24, (ld - pv) - self.pred_net0)); self.err = self.err[-336:]
        seg = self._segment(t, obs)
        if seg is not None:                             # valors de mercat de la setmana següent: a la seva hora real
            rem, tn = seg
            for i in range(rem, len(pr)):
                self.hist.put("price", tn + i - rem, pr[i])
            for i in range(rem, len(cr)):
                self.hist.put("co2", tn + i - rem, cr[i])

        # tots els valors de mercat coneguts consecutius (3 h al Mode Difícil, 24 h a l'Estàndard)
        def known(v, d):
            out = []
            for x in v[:24]:
                if not math.isfinite(x):
                    break
                out.append(x)
            return out or [d]
        tbl = self.p.get("_fc_table")
        if tbl is not None and t in tbl:
            price, co2, load, pvf = [np.array(x, float) for x in tbl[t]]
            kp, kc = known(pr, 0.12), known(cr, 0.10)          # els valors coneguts són exactes
            price[:len(kp)] = kp; co2[:len(kc)] = kc
        else:
            price, co2, load, pvf = self._forecast(t, known(pr, 0.12), known(cr, 0.10))
        if tbl is None or t not in tbl:
            price, co2, load, pvf = self._jump_fill(t, seg, price, co2, load, pvf, len(known(pr, 0.12)))
        rec = self.p.get("_record")
        if rec is not None:
            rec[t] = (price.copy(), co2.copy(), load.copy(), pvf.copy())
        Hh = int(max(2, min(24, self.p.get("horizon", 24))))
        # final de la simulació: si la finestra de mercat s'escurça, només queden len(pr) hores
        n_mkt = len(pr); self.max_len = max(self.max_len, n_mkt)
        fi = 0 < n_mkt < self.max_len
        if fi:
            Hh = max(1, n_mkt)
        # final d'episodi conegut pel calendari (setmanes de test = darrers dies del mes, fins a les 23:00 de l'últim dia)
        rem = None
        rt = self.p.get("_rem")
        if rt is not None:
            rem = rt.get(t)
        elif self.p.get("ep_month_end", False):
            try:
                ts = pd.Timestamp(obs.get("timestamp"))
                if ts.day >= self.p.get("ep_day0", 20):
                    end = (ts + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(hours=23)
                    rem = int(round((end - ts) / pd.Timedelta(hours=1))) + 1
            except Exception:
                rem = None
        if rem is not None and 0 < rem < Hh:
            Hh = rem; fi = True
        price, co2, load, pvf = price[:Hh], co2[:Hh], load[:Hh], pvf[:Hh]
        self.pred_net0 = float(load[0] - pvf[0])        # error de previsió mesurat sense biaix
        # biaix deliberat: descarregar de més es perd del tot, descarregar de menys només ajorna
        load = np.maximum(load + self.p.get("load_bias", 0.0), 0.0)
        pvf = pvf * self.p.get("pv_bias", 1.0)
        net = load - pvf
        w = 0.7 * co2 + 0.3 * self.p["ratio"] * price
        hw = self.p.get("_half_w")                       # pes per hora segons la meitat (públic/privat) on cau
        if hw is not None:
            mc = np.array([hw.get(t + h, (1.0, 1.0))[0] for h in range(len(w))])
            mp = np.array([hw.get(t + h, (1.0, 1.0))[1] for h in range(len(w))])
            w = 0.7 * co2 * mc + 0.3 * self.p["ratio"] * price * mp
        self.hist_w.append(float(w[0]))
        ref = np.concatenate([w, self.hist_w[-72:]])
        v = 0.0 if fi else self.p["term_scale"] * ETA * float(np.quantile(ref, self.p["q_term"]))
        # prudència a la descàrrega de l'hora actual: net previst - k * error típic
        sig = float(np.mean([abs(e) for _, e in self.err])) if len(self.err) >= 12 else 80.0
        if self.p.get("sigma_mode") == "hod":
            same = [abs(e) for h_, e in self.err if h_ == t % 24]
            if len(same) >= 3:
                sig = float(np.mean(same))
        dmax0 = max(0.0, float(net[0]) - self.p["k_dis"] * sig)
        scen = None
        K = int(self.p.get("scen", 0))
        if K:
            h0 = t % 24
            pool = [e for h_, e in self.err if min((h_ - h0) % 24, (h0 - h_) % 24) <= self.p.get("scen_win", 1)]
            if len(pool) < 8:
                pool = [e for _, e in self.err]
            qs = (np.arange(K) + 0.5) / K
            if len(pool) >= 8:
                scen = np.quantile(pool, qs) * self.p.get("scen_scale", 1.0)
            else:
                from statistics import NormalDist
                scen = np.array([NormalDist().inv_cdf(q) for q in qs]) * self.p.get("scen_s0", 100.0)
            st = self.p.get("_scen_table")
            if st is not None and t in st:
                scen = np.asarray(st[t], float) - float(net[0])
            dmax0 = self.pmax
        net_plan = np.asarray(net, float).copy()
        if self.p.get("k_future", 0.0):
            net_plan[1:] = net_plan[1:] - self.p["k_future"] * sig * np.sqrt(np.arange(1, len(net_plan)))
        P = None
        ms = self.p.get("_mkt_scen")
        if ms is not None and t in ms and len(w) > 3:
            pS, cS = ms[t]
            n = len(w)
            pS = np.asarray(pS, float)[:, :n].copy(); cS = np.asarray(cS, float)[:, :n].copy()
            pS[:, :3] = price[:3]; cS[:, :3] = co2[:3]
            W = 0.7 * cS + 0.3 * self.p["ratio"] * pS
            P = self._lp_saa(W, net_plan, soc, v, float(w[0]), scen)
        if P is None:
            P = self._lp(w, net_plan, soc, v, dmax0, scen)
        if P is None:
            a = 0.0
        else:
            a = float(P[0])
        return self._clamp(soc, a)

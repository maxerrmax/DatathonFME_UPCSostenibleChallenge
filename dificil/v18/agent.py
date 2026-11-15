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
from scipy.optimize import linprog

try:
    import joblib, lightgbm as lgb
except Exception:  # pragma: no cover
    joblib = lgb = None

from feats import Hist, mkt_features, mkt_features2, load_features2, meas_features, set_cs, cs_value

ETA = 0.9 ** 0.5
HERE = Path(__file__).parent
END_EPOCH = int(np.datetime64("2026-08-31T23", "h").astype(np.int64))   # última hora del test (hores des d'epoch)
FIRST_WEEK = int(np.datetime64("2025-09-24T00", "h").astype(np.int64))  # primera hora de la primera setmana de test


def month_end(t):
    """Última hora del mes de l'hora t (hores des d'epoch). Les setmanes de test són els 7 últims dies de cada mes."""
    m = np.datetime64(int(t), "h").astype("datetime64[M]")
    return int((m + 1).astype("datetime64[h]").astype(np.int64)) - 1


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
        # v18: règim d'avaluació. 'reset' = cada setmana és un episodi (SOC al 50 % a l'inici de cada període, segons
        # l'enunciat); 'cont' = un sol episodi continu. None = encara no se sap (es juga a 'reset').
        self.regime = None
        self._exp_soc = None
        self._first = True
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
        return np.mean(out, axis=0)

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

    def _forecast(self, t, price3, co2_3):
        H = 24
        price = np.full(H, np.nan); co2 = np.full(H, np.nan)
        price[:len(price3)] = price3; co2[:len(co2_3)] = co2_3
        hs = list(range(len(price3), H))
        if hs:
            if self.mk2:
                price[hs] = self._pred_mkt2("price", t, hs)
                co2[hs] = self._pred_mkt2("co2", t, hs)
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
            if old is not None and n_exact > 0 and len(old) >= 3 + n_exact:
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

    def _clamp(self, soc, a):
        mc = max(0.0, min(self.pmax, (self.cap - soc) / ETA - 1e-3))
        md = max(0.0, min(self.pmax, soc * ETA - 1e-3))
        a = max(-md, min(mc, a))
        return 0.0 if (not math.isfinite(a) or abs(a) < 1e-6) else float(a)

    # ------------------------------------------------------------------ pas
    def act(self, obs):
        try:
            return self._act(obs)
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
        try:                                            # v18: detecció del règim
            if self._first:
                self._first = False
                if t != FIRST_WEEK and (t % 24) == 0 and month_end(t) - t + 1 == 7 * 24:
                    self.regime = "reset"               # agent creat de nou a l'inici d'una setmana posterior
            elif self.t_prev is not None and t - self.t_prev > 1 and self._exp_soc is not None and self.regime is None:
                half = self.cap * 0.5
                if abs(soc - self._exp_soc) > 1.0 and abs(soc - half) < 1.0:
                    self.regime = "reset"               # el SOC s'ha reiniciat en canviar de setmana
                elif abs(soc - self._exp_soc) <= 1.0 and abs(self._exp_soc - half) > 1.0:
                    self.regime = "cont"                # el SOC continua: un sol episodi
        except Exception:
            pass
        self.t_prev = t
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
        self.hist.put("load", t - 1, ld); self.hist.put("pv", t - 1, pv)
        self.hist.put("irr", t - 1, _num(nd.get("irradiancia_wm2"))); self.hist.put("temp", t - 1, _num(nd.get("temp_ambient_c")))
        self.hist.put("vent", t - 1, _num(nd.get("vent_ms")))
        if self.pred_net0 is not None and math.isfinite(ld) and math.isfinite(pv):
            self.err.append(((t - 1) % 24, (ld - pv) - self.pred_net0)); self.err = self.err[-336:]

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
        rec = self.p.get("_record")
        if rec is not None:
            rec[t] = (price.copy(), co2.copy(), load.copy(), pvf.copy())
        Hh = int(max(2, min(24, self.p.get("horizon", 24))))
        # final de la simulació: si la finestra de mercat s'escurça, només queden len(pr) hores
        n_mkt = len(pr); self.max_len = max(self.max_len, n_mkt)
        fi = 0 < n_mkt < self.max_len
        if fi:
            Hh = max(1, n_mkt)
        # v18: final conegut de l'episodi. Si cada setmana és un episodi (o encara no se sap), el final és l'última hora
        # del mes (fi de la setmana de test); si és un sol episodi continu, només el 31/8/2026 23:00. Quan l'horitzó hi
        # arriba es planifica fins al final i l'energia restant val 0 (la finestra de mercat només avisa 2 h abans).
        try:
            mode = self.p.get("week_mode", "auto")
            wl = self.p.get("_week_ends")                # NOMÉS proves locals: finals de setmana injectats
            if wl:
                wk_end = next((int(e) for e in wl if e >= t), None)
            else:
                wk_end = month_end(t)
            per_week = mode == "always" or (mode == "auto" and self.regime != "cont")
            ends = [int(self.p.get("end_epoch", END_EPOCH))]
            if per_week and wk_end is not None:
                ends.append(wk_end)
            R = min([e - t + 1 for e in ends if e - t + 1 > 0] or [0])
            if 0 < R < Hh:
                Hh = R; fi = True
        except Exception:
            pass
        price, co2, load, pvf = price[:Hh], co2[:Hh], load[:Hh], pvf[:Hh]
        self.pred_net0 = float(load[0] - pvf[0])        # error de previsió mesurat sense biaix
        # biaix deliberat: descarregar de més es perd del tot, descarregar de menys només ajorna
        load = np.maximum(load + self.p.get("load_bias", 0.0), 0.0)
        pvf = pvf * self.p.get("pv_bias", 1.0)
        net = load - pvf
        w = 0.7 * co2 + 0.3 * self.p["ratio"] * price
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
        P = self._lp(w, net_plan, soc, v, dmax0, scen)
        if P is None:
            a = 0.0
        else:
            a = float(P[0])
        a = self._clamp(soc, a)
        self._exp_soc = soc + (a * ETA if a > 0 else a / ETA)
        return a

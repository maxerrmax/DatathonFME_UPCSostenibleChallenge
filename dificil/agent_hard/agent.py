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

from feats import Hist, mkt_features, meas_features

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
        try:
            raw = p.get("_models")
            if raw is None:
                raw = joblib.load(HERE / p["models"])
            # LightGBM es desa com a text; la resta (XGBoost, sklearn, ensembles) com a objectes amb .predict
            self.m = {k: (lgb.Booster(model_str=v) if isinstance(v, str) else v) for k, v in raw.items()}
        except Exception:
            self.m = {}
        self.hist = Hist()
        self.t_prev = None
        self.err = []                                   # errors recents de previsió de net (t-1)
        self.pred_net0 = None
        self.hist_w = []
        self.cap, self.pmax = 6000.0, 1500.0
        self.max_len = 0                                # llargada màxima vista de la finestra de mercat

    # ------------------------------------------------------------------ previsions
    def _pred(self, key, X, kind):
        lag = X[:, 5]
        if not self.m:
            out = lag.copy()
        else:
            d = self.m[key + "_dir"].predict(X)
            r = lag + self.m[key + "_res"].predict(X)
            out = np.where(np.isfinite(lag), r, d) if kind == "res" else d
        if key == "pv":
            last, last24 = X[:, 8], X[:, 8] - X[:, 9]
            ratio = np.clip(np.where(np.isfinite(last24) & (last24 > 50), last / np.maximum(last24, 1), 1.0), 0.3, 1.5)
            pers = lag * (1 + (ratio - 1) * np.exp(-X[:, 0] / 6.0))
            a = self.p["pv_blend"]
            out = np.where(np.isfinite(lag), a * out + (1 - a) * pers, out)
            hod = X[:, 1]
            out = np.where((hod < 6) | (hod > 21), 0.0, np.maximum(out, 0.0))
        return np.where(np.isfinite(out), out, np.nan)

    def _forecast(self, t, price3, co2_3):
        H = 24
        price = np.full(H, np.nan); co2 = np.full(H, np.nan)
        price[:len(price3)] = price3; co2[:len(co2_3)] = co2_3
        hs = list(range(len(price3), H))
        if hs:
            price[hs] = self._pred("price", mkt_features(self.hist, "price", t, hs), "res")
            co2[hs] = self._pred("co2", mkt_features(self.hist, "co2", t, hs), "dir")
        hs_all = list(range(H))
        load = self._pred("load", meas_features(self.hist, "load", t, hs_all), "res")
        pv = self._pred("pv", meas_features(self.hist, "pv", t, hs_all), "dir")
        # farciment de forats: últim valor conegut / mitjana
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
        return price, co2, load, pv

    # ------------------------------------------------------------------ optimització
    def _lp(self, w, net, soc0, v_term, dmax0):
        H = len(w); cap, pm = self.cap, self.pmax
        L = np.tril(np.ones((H, H)))
        Asoc = np.hstack([L * ETA, -L / ETA, np.zeros((H, H))])          # SOC_k - soc0
        # net + c - d - imp <= 0
        Aimp = np.hstack([np.eye(H), -np.eye(H), -np.eye(H)])
        A = np.vstack([Asoc, -Asoc, Aimp])
        b = np.concatenate([np.full(H, cap - 1e-3 - soc0), np.full(H, soc0 - 1e-3), -net])
        cost = np.concatenate([np.zeros(2 * H), w]) - v_term * Asoc[-1]
        bounds = [(0, pm)] * H + [(0, pm)] * H + [(0, None)] * H
        bounds[H] = (0, max(0.0, min(pm, dmax0)))
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
        price, co2, load, pvf = price[:Hh], co2[:Hh], load[:Hh], pvf[:Hh]
        net = load - pvf
        self.pred_net0 = float(net[0])
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
        net_plan = np.asarray(net, float).copy()
        if self.p.get("k_future", 0.0):
            net_plan[1:] = net_plan[1:] - self.p["k_future"] * sig * np.sqrt(np.arange(1, len(net_plan)))
        P = self._lp(w, net_plan, soc, v, dmax0)
        if P is None:
            a = 0.0
        else:
            a = float(P[0])
        return self._clamp(soc, a)

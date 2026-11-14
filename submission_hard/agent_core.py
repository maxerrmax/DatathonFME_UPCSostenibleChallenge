# ---------------------------------------------------------------------------
# BatteryAgent — Mode Difícil (6.000 kWh, ±1.500 kW, solar x12, mercat a 3 h)
#
# Cada hora:
#   1. Guarda a la memòria el que sap (preu/CO2 de 3 h, consum/solar/meteo de l'hora passada).
#   2. Preveu 24 h de preu, CO2, consum i solar amb models LightGBM (fitxers model_*.txt).
#   3. Resol un programa lineal de 24 h (minimitza 0.7·CO2 + 0.3·cost importats de la xarxa,
#      aprofitant l'excedent solar gratuït) i aplica només la primera hora (MPC).
#   4. Blindatge: dades absents/NaN, descàrrega limitada al consum previst, pla B si falla,
#      i clamp físic per tenir 0 infraccions.
# ---------------------------------------------------------------------------
import json
from pathlib import Path

import numpy as np

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

HERE = Path(__file__).resolve().parent
ETA = 0.9 ** 0.5

DEFAULTS = {
    "KC": 78.09690663686939,     # mitjana(consum_net_base · CO2) al train (solar x12)
    "KP": 127.89844467070672,    # mitjana(consum_net_base · preu)
    "alpha": 0.7, "beta": 0.3,
    "horizon": 24,
    "pv_scale_hard": 12.0,
    "v_term_q": 0.0,             # valor terminal de l'energia (quantil dels pesos de l'horitzó)
    "v_term_f": 0.0,
    "dis_margin_kw": 0.0,        # marge de seguretat en descarregar respecte al consum net previst
    "load_bias": 0.0,
    "pv_bias": 1.0,
    "outage_kw": 100.0,
    "pc_contrast": 1.0,          # amplifica el contrast entre hores previstes de preu/CO2
    "solar_file": "solar.json",
    "pv_mix": 0.0,               # pes del model físic de solar (cel clar × claredat); 0 = només LightGBM
    "res_w": 0.0,                # pes del model residual de preu/CO2 (0 = només directe)          # consum mesurat per sota d'això = avaria / campus aturat
}


def _f(x, default=float("nan")):
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else default
    except Exception:
        return default


class BatteryAgent:
    def __init__(self, model_dir=None, params=None):
        d = Path(model_dir) if model_dir else HERE
        self.p = dict(DEFAULTS)
        try:
            pf = d / "params.json"
            if pf.exists():
                self.p.update(json.loads(pf.read_text()))
        except Exception:
            pass
        if params:
            self.p.update(params)
        self.models = {}
        if lgb is not None:
            for tg in ("p", "c", "load", "pv", "p_res", "c_res"):
                f = d / f"model_{tg}.txt"
                if f.exists():
                    try:
                        self.models[tg] = lgb.Booster(model_file=str(f))
                    except Exception:
                        pass
        self.solar = None
        try:
            sf = d / self.p["solar_file"]
            if sf.exists():
                js = json.loads(sf.read_text())
                self.solar = {k: np.asarray(v, float) for k, v in js.items()}
        except Exception:
            self.solar = None
        self.store = {k: {} for k in ("p", "c", "load", "pv", "irr", "temp")}
        self.last_load = 900.0
        self.last_price, self.last_co2 = 0.13, 0.08
        self.last_t0 = None
        self.repeat_guard = True   # si el timestamp no avança, avancem el rellotge intern

    # ------------------------------------------------------------------ memòria
    def _get(self, col, h):
        return self.store[col].get(h, float("nan"))

    def _remember(self, obs, t0, pv_scale):
        prices = list(obs.get("preus_properes_24h") or [])
        co2s = list(obs.get("co2_properes_24h") or [])
        p_now = _f(obs.get("preu_actual"))
        c_now = _f(obs.get("co2_actual"))
        if not prices:
            prices = [p_now]
        if not co2s:
            co2s = [c_now]
        for i, v in enumerate(prices):
            v = _f(v)
            if v == v:
                self.store["p"][t0 + i] = v
        for i, v in enumerate(co2s):
            v = _f(v)
            if v == v:
                self.store["c"][t0 + i] = v
        n5 = obs.get("noves_dades_5min") or {}
        load = _f(n5.get("ultim_consum_kw"))
        pv = _f(n5.get("ultima_solar_kw"))
        if load == load and t0 > 0:
            self.store["load"][t0 - 1] = load
        if pv == pv:
            self.store["pv"][t0 - 1] = pv / pv_scale
        irr = _f(n5.get("irradiancia_wm2"))
        tmp = _f(n5.get("temp_ambient_c"))
        if irr == irr:
            self.store["irr"][t0 - 1] = irr
        if tmp == tmp:
            self.store["temp"][t0 - 1] = tmp
        # neteja de memòria antiga (> 3 dies)
        if t0 % 168 == 0:
            for col in self.store:
                self.store[col] = {h: v for h, v in self.store[col].items() if h >= t0 - 72}

    # ------------------------------------------------------------------ previsions
    def _cs(self, h):
        d = hour_to_dt(h)
        return float(self.solar["cs"][d.timetuple().tm_yday - 1, d.hour])

    def _pv_physical(self, t0, H):
        """Solar (kW reals) = cel clar × claredat prevista (mescla de l'última hora, les darreres 24 h i la mitjana)."""
        sol = self.solar
        cs1 = self._cs(t0 - 1)
        p1 = self._get("pv", t0 - 1)
        kn = min(max(p1 / cs1, 0.0), 1.3) if (cs1 > 8.0 and p1 == p1) else float("nan")
        num = den = 0.0
        for i in range(1, 25):
            c = self._cs(t0 - i)
            v = self._get("pv", t0 - i)
            if c > 8.0 and v == v:
                num += v; den += c
        kd = num / den if den > 0 else float("nan")
        out = np.zeros(H)
        for k in range(H):
            T = t0 + k
            dt = hour_to_dt(T)
            c = self._cs(T)
            if c <= 0.5:
                continue
            kc = float(sol["ktc"][dt.month - 1, dt.hour])
            kk = min(k, 23)
            wn, wd = float(sol["w_now"][kk]), float(sol["w_d1"][kk])
            kd_ = kd if kd == kd else kc
            kn_ = kn if kn == kn else kd_
            out[k] = c * (wn * kn_ + wd * kd_ + (1 - wn - wd) * kc)
        return out

    def _forecast(self, t0, H, pv_scale):
        get = self._get
        price = np.full(H, np.nan)
        co2 = np.full(H, np.nan)
        for k in range(H):
            price[k] = get("p", t0 + k)
            co2[k] = get("c", t0 + k)
        unk = [k for k in range(H) if price[k] != price[k] or co2[k] != co2[k]]
        if unk:
            for tg, arr in (("p", price), ("c", co2)):
                m = self.models.get(tg)
                ks = [k for k in unk if arr[k] != arr[k]]
                if m is not None and ks:
                    X = np.array([price_features(get, t0, k, tg) for k in ks], float)
                    pred = m.predict(X)
                    mr = self.models.get(tg + "_res")
                    rw = float(self.p["res_w"])
                    if mr is not None and rw > 0:
                        # model residual: diferència respecte a l'últim valor conegut (t0+2)
                        base = X[:, PRICE_FEATS.index("x_cur")]
                        known = next((v for v in (get(tg, t0 + i) for i in (2, 1, 0)) if v == v), float("nan"))
                        pr = np.where(np.isnan(base), known, base) + mr.predict(X)
                        pred = np.where(np.isnan(pr), pred, (1 - rw) * pred + rw * pr)
                    g = float(self.p["pc_contrast"])
                    if g != 1.0 and len(ks) > 1:
                        mu = float(np.mean(pred))
                        pred = mu + g * (pred - mu)
                    arr[ks] = pred
        # forats que quedin: últim conegut
        lp = next((v for v in (get("p", t0 + i) for i in (2, 1, 0)) if v == v), self.last_price)
        lc = next((v for v in (get("c", t0 + i) for i in (2, 1, 0)) if v == v), self.last_co2)
        price = np.where(np.isnan(price), lp, price)
        co2 = np.where(np.isnan(co2), lc, co2)
        self.last_price, self.last_co2 = lp, lc

        ks = list(range(H))
        m = self.models.get("load")
        if m is not None:
            load = m.predict(np.array([load_features(get, t0, k) for k in ks], float))
        else:
            ll = get("load", t0 - 1)
            load = np.full(H, ll if ll == ll else self.last_load)
        m = self.models.get("pv")
        if m is not None:
            pv = m.predict(np.array([pv_features(get, t0, k) for k in ks], float))
        else:
            pv = np.zeros(H)
        mix = float(self.p["pv_mix"])
        if self.solar is not None and mix > 0:
            try:
                pv = mix * self._pv_physical(t0, H) + (1 - mix) * np.clip(pv, 0, None)
            except Exception:
                pass
        pv = np.clip(pv, 0, None) * pv_scale * self.p["pv_bias"]
        load = np.clip(load, 0, None) + self.p["load_bias"]
        # Blindatge: si el comptador marca ~0 (avaria del sensor o campus aturat), suposem que continua
        ll = get("load", t0 - 1)
        if ll == ll and ll < self.p["outage_kw"]:
            load = np.full(H, max(ll, 0.0))
        return price, co2, np.clip(load, 0, None), pv
    # ------------------------------------------------------------------ decisió
    def act(self, obs):
        try:
            return self._act(obs)
        except Exception:
            try:
                return self._fallback(obs)
            except Exception:
                return 0.0

    def _limits(self, obs):
        cap = _f(obs.get("cap_max_kwh"), 6000.0)
        pmax = _f(obs.get("power_max_kw"), 1500.0)
        soc = min(max(_f(obs.get("soc_kwh"), cap / 2), 0.0), cap)
        return cap, pmax, soc

    def _clamp(self, p, obs):
        cap, pmax, soc = self._limits(obs)
        hi = max(0.0, min(pmax, (cap - soc) / ETA - 1e-4))
        lo = max(0.0, min(pmax, soc * ETA - 1e-4))
        p = _f(p, 0.0)
        return float(max(-lo, min(hi, p)))

    def _act(self, obs):
        cap, pmax, soc = self._limits(obs)
        hard = cap > 2500
        pv_scale = self.p["pv_scale_hard"] if hard else 1.0
        t0 = ts_to_hour(obs.get("timestamp"))
        if t0 is None or (self.last_t0 is not None and t0 <= self.last_t0 and self.repeat_guard):
            # sense data fiable: rellotge intern (1 h per crida)
            t0 = (self.last_t0 + 1) if self.last_t0 is not None else 0
        self.last_t0 = t0
        self._remember(obs, t0, pv_scale)
        H = int(self.p["horizon"])
        price, co2, load, pv = self._forecast(t0, H, pv_scale)
        net = load - pv
        w = self.p["alpha"] * co2 / self.p["KC"] + self.p["beta"] * price / self.p["KP"]
        v_term = 0.0
        if self.p["v_term_f"]:
            v_term = self.p["v_term_f"] * float(np.quantile(w, self.p["v_term_q"]))
        plan = solve_dispatch(w, net, soc, cap, pmax, v_term=v_term)
        p0 = float(plan[0])
        if p0 < 0:
            # blindatge: no descarregar per sobre del consum net previst (l'excedent no compta)
            p0 = -min(-p0, max(0.0, net[0] - self.p["dis_margin_kw"]))
        return self._clamp(p0, obs)

    def _fallback(self, obs):
        """Pla B: carregar en hores barates/netes, descarregar en hores cares/brutes."""
        cap, pmax, soc = self._limits(obs)
        pr = _f(obs.get("preu_actual"), 0.13)
        c = _f(obs.get("co2_actual"), 0.08)
        s = 0.7 * c / 0.085 + 0.3 * pr / 0.138
        n5 = obs.get("noves_dades_5min") or {}
        net = _f(n5.get("ultim_consum_kw"), 0.0) - _f(n5.get("ultima_solar_kw"), 0.0)
        if net < 0:
            p = min(pmax, -net)
        elif s < 0.85:
            p = pmax * 0.6
        elif s > 1.15:
            p = -min(pmax, max(0.0, net))
        else:
            p = 0.0
        return self._clamp(p, obs)

"""stoch_mpc: MPC estocastic de dues etapes sobre tot l'horitzo (sobre el v12).

Idea: en lloc d'un unic LP determinista, es construeixen S escenaris del futur (consum net, CO2 i preu = previsio
+ errors mostrejats amb dispersio creixent amb l'horitzo, autocorrelacionats). Decisio de la 1a hora COMPARTIDA
(c0,d0); de l'hora 1 en endavant cada escenari te la seva recurs (c,d,soc,import). Es minimitza el cost mitja.
L'error de CO2 s'estima en linia (error relatiu a h=3..5 de les previsions pro'pies); el de consum net a partir
dels errors de l'hora actual del v12. L'hora 0 usa els quantils d'error de v12 (`scen`) si hi son.
Qualsevol fallada -> LP determinista del v12.
Params (self.p): S, co2_rel, price_rel, net_sig, corr, adapt, seed, k_future (per defecte 0: l'spread ja cobreix).
"""
import sys
from pathlib import Path
from statistics import NormalDist
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

_B = load_base()
Base = _B.BatteryAgent
ETA = _B.ETA


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__(params)
        d = {"S": 8, "co2_rel": 0.05, "price_rel": 0.08, "net_sig": 60.0, "corr": 0.8, "k_future": 0.0,
             "adapt": True, "seed": 7}
        for k, v in d.items():
            if not (params and k in params):
                self.p[k] = v
        self._fc = None
        self._fc_hist = {}
        self._cstat = 1.0                       # factor d'escala online de l'error relatiu de CO2
        self._lpcache = {}
        self._mkscen()

    # -------------------------------------------------------------- escenaris
    def _mkscen(self):
        S = int(self.p["S"]); H = 24
        rng = np.random.default_rng(int(self.p["seed"]))

        def strat():
            q = (np.arange(S) + 0.5) / S
            z = np.array([NormalDist().inv_cdf(x) for x in q]); rng.shuffle(z)
            return z

        def field():
            a = self.p["corr"]
            u = rng.standard_normal((S // 2 + 1, H)); u = np.vstack([u, -u])[:S]
            return a * strat()[:, None] + np.sqrt(1 - a * a) * u
        self.Zc, self.Zp, self.Zn = field(), field(), field()

    # -------------------------------------------------------------- previsio (guarda l'ultima i l'error de CO2)
    def _forecast(self, t, price3, co2_3):
        out = super()._forecast(t, price3, co2_3)
        try:
            self._fc = out
            co2 = out[1]
            self._fc_hist[t] = co2.copy()
            old = self._fc_hist.get(t - 3)
            n = min(3, len(co2_3))
            if old is not None and n > 0 and self.p.get("adapt", True):
                h = np.arange(3, 3 + n)
                rel = np.abs(np.asarray(co2_3[:n]) - old[3:3 + n]) / np.maximum(old[3:3 + n], 0.03)
                dflt = 0.8 * self.p["co2_rel"] * np.sqrt(h / 3.0)          # E|e| = 0.8 sigma
                r = float(np.mean(rel) / np.mean(dflt))
                if np.isfinite(r):
                    self._cstat = float(np.clip(0.97 * self._cstat + 0.03 * np.clip(r, 0.3, 3.0), 0.5, 2.0))
            for k in [k for k in self._fc_hist if k < t - 6]:
                del self._fc_hist[k]
        except Exception:
            pass
        return out

    # -------------------------------------------------------------- LP estocastic
    def _struct(self, H, S):
        key = (H, S)
        if key in self._lpcache:
            return self._lpcache[key]
        nS = 2 * (H - 1) + H                    # c1.., d1.., imp0..
        nv = 2 + S * nS
        rows, cols, vals = [], [], []
        r = 0
        for s in range(S):
            o = 2 + s * nS

            def cidx(k): return 0 if k == 0 else o + (k - 1)

            def didx(k): return 1 if k == 0 else o + (H - 1) + (k - 1)

            def iidx(k): return o + 2 * (H - 1) + k
            for sgn in (1, -1):                 # soc superior / inferior
                for k in range(H):
                    for j in range(k + 1):
                        rows += [r, r]; cols += [cidx(j), didx(j)]; vals += [sgn * ETA, -sgn / ETA]
                    r += 1
            for k in range(H):
                rows += [r, r, r]; cols += [cidx(k), didx(k), iidx(k)]; vals += [1.0, -1.0, -1.0]; r += 1
        A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
        self._lpcache[key] = (A, nS, nv)
        return self._lpcache[key]

    def _sigmas(self, H):
        h = np.arange(H, dtype=float)
        hh = np.maximum(h, 3)
        sc = np.where(h >= 3, self.p["co2_rel"] * np.sqrt(hh / 3.0) * self._cstat, 0.0)
        sp_ = np.where(h >= 3, self.p["price_rel"] * np.sqrt(hh / 3.0), 0.0)
        s0 = self.p["net_sig"]
        try:
            if len(self.err) >= 12:
                s0 = s0 * float(np.clip(np.mean([abs(e) for _, e in self.err]) / 80.0, 0.5, 2.0))
        except Exception:
            pass
        sn = s0 * np.sqrt(h)
        return sc, sp_, sn

    def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
        try:
            r = self._slp(w, net, soc0, v_term, dmax0, scen)
            if r is not None and np.isfinite(r).all():
                return r
        except Exception:
            pass
        return super()._lp(w, net, soc0, v_term, dmax0, scen)

    def _slp(self, w, net, soc0, v_term, dmax0, scen):
        H = len(w); S = int(self.p["S"])
        if H < 3:
            return None
        cap, pm = self.cap, self.pmax
        A, nS, nv = self._struct(H, S)
        w = np.asarray(w, float); net = np.asarray(net, float)
        sc, sp_, sn = self._sigmas(H)
        wS = np.empty((S, H)); netS = np.empty((S, H))
        fc = self._fc
        ok = fc is not None and len(fc[1]) >= H and len(fc[0]) >= H
        ratio = self.p["ratio"]
        for s in range(S):
            if ok:
                co2 = np.maximum(fc[1][:H] * (1 + sc * self.Zc[s, :H]), 0.0)
                pr = fc[0][:H] * (1 + sp_ * self.Zp[s, :H])
                wS[s] = 0.7 * co2 + 0.3 * ratio * pr + (w - (0.7 * fc[1][:H] + 0.3 * ratio * fc[0][:H]))
            else:
                wS[s] = w * (1 + sc * self.Zc[s, :H])
            netS[s] = net + sn * self.Zn[s, :H]
        wS = np.maximum(wS, 0.0)
        # hora 0: quantils d'error de v12 si hi son
        if scen is not None and len(scen) >= 3:
            rk = np.argsort(np.argsort(self.Zn[:, 1]))
            e0 = np.quantile(np.asarray(scen, float), (rk + 0.5) / S)
        else:
            e0 = self.Zn[:, 1] * (self.p["net_sig"] * 1.2)
        netS[:, 0] = net[0] + e0
        b = np.empty(A.shape[0]); cost = np.zeros(nv)
        r = 0
        bu = np.full(nv, np.inf); bu[0] = pm; bu[1] = max(0.0, min(pm, dmax0))
        cost[0] += -v_term * ETA; cost[1] += v_term / ETA      # valor terminal -v*soc_final (mitjana)
        for s in range(S):
            b[r:r + H] = cap - 1e-3 - soc0; r += H
            b[r:r + H] = soc0 - 1e-3; r += H
            b[r:r + H] = -netS[s]; r += H
            o = 2 + s * nS
            cost[o + 2 * (H - 1): o + 2 * (H - 1) + H] = wS[s] / S
            cost[o:o + H - 1] += -v_term * ETA / S
            cost[o + H - 1:o + 2 * (H - 1)] += v_term / ETA / S
            bu[o:o + 2 * (H - 1)] = pm
        res = linprog(cost, A_ub=A, b_ub=b, bounds=np.column_stack([np.zeros(nv), bu]), method="highs")
        if res.status != 0 or res.x is None:
            return None
        out = np.zeros(H); out[0] = res.x[0] - res.x[1]
        return out

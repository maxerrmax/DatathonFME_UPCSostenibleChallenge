"""s8dis: MPC estocàstic amb escenaris de DESACORD entre models.

En lloc de centrar tots els escenaris a la mitjana LightGBM/xarxa, la meitat es centren a la previsió de LightGBM
(v12) i l'altra meitat a la de la xarxa (nn_pre): la decisió de l'hora actual ha de ser bona sota tots dos models
(incertesa epistèmica), no només sota la seva mitjana. Mateix soroll i mateixes restriccions que stoch_mpc."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base, load_strategy

_B = load_base(); Base = _B.BatteryAgent; ETA = _B.ETA
ST = load_strategy("stoch_mpc").BatteryAgent
NP = load_strategy("nn_pre").BatteryAgent


class _Cap(Base):
    def _forecast(self, t, price3, co2_3):
        out = super()._forecast(t, price3, co2_3)
        self._fcL = (np.array(out[0], float), np.array(out[1], float))
        return out


class BatteryAgent(ST, NP, _Cap):
    def _centers(self, H):
        fc = self._fc; L = getattr(self, "_fcL", None)
        if fc is None or L is None or len(L[1]) < H or len(fc[1]) < H:
            return None
        a, b = float(self.p.get("nn_mix", 0.5)), float(self.p.get("nn_mix_price", 0.3))
        cL, pL = L[1][:H], L[0][:H]
        cM, pM = np.asarray(fc[1][:H], float), np.asarray(fc[0][:H], float)
        cN = cL.copy(); pN = pL.copy()
        if a > 0:
            cN = np.where(np.arange(H) >= 3, (cM - (1 - a) * cL) / a, cM)
        if b > 0:
            pN = np.where(np.arange(H) >= 3, (pM - (1 - b) * pL) / b, pM)
        return [(np.maximum(cL, 0), pL), (np.maximum(cN, 0), pN)]

    def _slp(self, w, net, soc0, v_term, dmax0, scen):
        H = len(w); S = int(self.p["S"])
        if H < 3:
            return None
        cen = self._centers(H)
        if cen is None:
            return super()._slp(w, net, soc0, v_term, dmax0, scen)
        cap, pm = self.cap, self.pmax
        A, nS, nv = self._struct(H, S)
        w = np.asarray(w, float); net = np.asarray(net, float)
        sc, sp_, sn = self._sigmas(H)
        ratio = self.p["ratio"]
        fc = self._fc
        wmix = 0.7 * np.asarray(fc[1][:H]) + 0.3 * ratio * np.asarray(fc[0][:H])
        wS = np.empty((S, H)); netS = np.empty((S, H))
        for s in range(S):
            c0, p0 = cen[s % 2]
            co2 = np.maximum(c0 * (1 + sc * self.Zc[s, :H]), 0.0)
            pr = p0 * (1 + sp_ * self.Zp[s, :H])
            wS[s] = 0.7 * co2 + 0.3 * ratio * pr + (w - wmix)
            netS[s] = net + sn * self.Zn[s, :H]
        wS = np.maximum(wS, 0.0)
        if scen is not None and len(scen) >= 3:
            rk = np.argsort(np.argsort(self.Zn[:, 1]))
            e0 = np.quantile(np.asarray(scen, float), (rk + 0.5) / S)
        else:
            e0 = self.Zn[:, 1] * (self.p["net_sig"] * 1.2)
        netS[:, 0] = net[0] + e0
        b = np.empty(A.shape[0]); cost = np.zeros(nv); r = 0
        bu = np.full(nv, np.inf); bu[0] = pm; bu[1] = max(0.0, min(pm, dmax0))
        cost[0] += -v_term * ETA; cost[1] += v_term / ETA
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

"""regime_switch: controlador de regles tipus duck curve, sense LP.
Les previsions v12 nomes serveixen per ordenar hores per w = 0.7*CO2 + 0.3*ratio*preu:
  1) excedent solar (net<0): sempre carregar (es gratis, si no es perd);
  2) hores de descarrega D = fraccio top_frac mes cara de w amb net>0; es descarrega per cobrir el consum net,
     reservant energia per a les hores de D mes cares que queden (l'energia dura fins a la propera finestra barata);
  3) carrega de xarxa a les K hores de menor w (K = les necessaries per omplir el que demanen les descarregues),
     nomes si w_carrega < eff*w_descarrega_minima (rendiment 0.9).
S'implementa sobreescrivint _lp (rep w i net del v12) i retornant un pla hora a hora.
"""
import sys
import math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

_B = load_base()
Base = _B.BatteryAgent
ETA = _B.ETA


class BatteryAgent(Base):
    def __init__(self, params=None):
        p = {"scen": 0, "k_future": 0.0}
        p.update(params or {})
        super().__init__(p)
        for k, v in {"rs_top_frac": 0.35, "rs_eff": 0.9, "rs_grid": 1}.items():
            self.p.setdefault(k, v)

    def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
        try:
            return self._rules(np.asarray(w, float), np.asarray(net, float), soc0, dmax0)
        except Exception:
            return super()._lp(w, net, soc0, v_term, dmax0, scen)

    def _rules(self, w, net, soc0, dmax0):
        H = len(w)
        cap, pm = self.cap, self.pmax
        solar = np.clip(-net, 0, pm)
        nd = max(1, int(round(self.p["rs_top_frac"] * H)))
        cand = [int(h) for h in np.argsort(-w) if net[h] > 0]
        D = cand[:nd]
        dem = {h: min(pm, net[h]) for h in D}
        demand_batt = sum(dem.values()) / ETA
        G = set()
        if self.p["rs_grid"] and D:
            have = soc0 + ETA * solar.sum()
            need = max(0.0, min(cap - have, demand_batt - have))
            K = int(math.ceil(need / (ETA * pm))) if need > 0 else 0
            wmin = min(w[h] for h in D)
            for h in np.argsort(w):
                h = int(h)
                if len(G) >= K:
                    break
                if h in dem or net[h] < 0:
                    continue
                if w[h] < self.p["rs_eff"] * wmin:
                    G.add(h)
        P = np.zeros(H)
        soc = soc0
        for h in range(H):
            if net[h] < 0:
                a = min(solar[h], (cap - soc) / ETA)
            elif h in G:
                a = min(pm, (cap - soc) / ETA)
            elif h in dem:
                reserve = sum(dem[j] for j in dem if j > h and w[j] > w[h]) / ETA
                dis = min(dem[h], max(0.0, (soc - reserve) * ETA))
                if h == 0:
                    dis = min(dis, dmax0)
                a = -dis
            else:
                a = 0.0
            P[h] = a
            soc += ETA * a if a > 0 else a / ETA
            soc = min(cap, max(0.0, soc))
        return P

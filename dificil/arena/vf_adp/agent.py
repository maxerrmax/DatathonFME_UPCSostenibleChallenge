"""vf_adp: programació dinàmica aproximada amb valor après de l'energia emmagatzemada.

En lloc de preveure el CO2/preu de les hores +3..+23 i resoldre un LP de 24 h, l'agent:
  1. aprèn (offline, `build_vf.py`) lambda(s) = valor marginal d'1 kWh a la bateria d'aquí a 3 h amb SOC s,
     a partir dels duals de LPs amb previsió perfecta sobre l'historial (model monòton en s, escala relativa);
  2. cada hora resol un LP de només 3 h (les hores de mercat conegudes EXACTAMENT) amb valor terminal còncau
     a trossos sum_j lambda_j * tram_j.
Consum/solar de les 3 hores i escenaris de l'hora actual: els del v12. Qualsevol fallada -> LP del v12.
Params: vf (fitxer del model; per defecte deduït de `models`), vf_scale (multiplicador de lambda, 1.0), L (3).
"""
import sys, pickle
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from common import load_base  # noqa: E402

_B = load_base()                       # posa dificil/agent_hard (feats) al path
from vf_feats import base_features, with_soc, SOC_MID, SEG, NSEG  # noqa: E402
Base = _B.BatteryAgent
ETA = _B.ETA


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__(params)
        self.vf = None
        try:
            import lightgbm as lgb
            name = self.p.get("vf")
            if not name:
                m = str(self.p.get("models", "models_all7.pkl"))
                name = "vf_all.pkl" if "all" in m else "vf_" + m.replace("models_", "").replace("7.pkl", ".pkl")
            d = pickle.load(open(HERE / name, "rb"))
            self.vf = lgb.Booster(model_str=d["vf"])
        except Exception:
            self.vf = None
        self.lam_log = self.p.get("_lam_log")

    def _lam(self, t):
        f, scale = base_features(self.hist.get, t, self.p["ratio"])
        lam = self.vf.predict(with_soc(f, SOC_MID, self.cap if self.cap > 0 else 6000.0), num_threads=1) * scale
        lam = np.maximum(np.minimum.accumulate(lam * self.p.get("vf_scale", 1.0)), 0.0)
        return lam

    def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
        try:
            if self.vf is not None and len(w) >= 24 and abs(self.cap - 6000.0) < 1:
                r = self._vlp(w, net, soc0, dmax0, scen)
                if r is not None and np.isfinite(r).all():
                    return r
        except Exception:
            pass
        return super()._lp(w, net, soc0, v_term, dmax0, scen)

    def _vlp(self, w, net, soc0, dmax0, scen):
        L = int(self.p.get("L", 3)); cap, pm = self.cap, self.pmax
        w = np.asarray(w[:L], float); net = np.asarray(net[:L], float)
        lam = self._lam(self.t_prev)
        if self.lam_log is not None:
            self.lam_log[self.t_prev] = lam
        K = 0 if scen is None else len(scen)
        # variables: c(L), d(L), imp(L), seg(NSEG), impS(K)
        nv = 3 * L + NSEG + K
        Lo = np.tril(np.ones((L, L)))
        Asoc = np.zeros((L, nv)); Asoc[:, :L] = Lo * ETA; Asoc[:, L:2 * L] = -Lo / ETA
        Aimp = np.zeros((L, nv)); Aimp[:, :L] = np.eye(L); Aimp[:, L:2 * L] = -np.eye(L); Aimp[:, 2 * L:3 * L] = -np.eye(L)
        A = np.vstack([Asoc, -Asoc, Aimp])
        b = np.concatenate([np.full(L, cap - 1e-3 - soc0), np.full(L, soc0 - 1e-3), -net])
        Aeq = np.zeros((1, nv)); Aeq[0, :2 * L] = Asoc[-1, :2 * L]; Aeq[0, 3 * L:3 * L + NSEG] = -1.0
        beq = np.array([-soc0])
        cost = np.zeros(nv); cost[2 * L:3 * L] = w; cost[3 * L:3 * L + NSEG] = -lam
        bounds = [(0, pm)] * (2 * L) + [(0, None)] * L + [(0, SEG)] * NSEG + [(0, None)] * K
        bounds[L] = (0, max(0.0, min(pm, dmax0)))
        if K:
            S = np.zeros((K, nv)); S[:, 0] = 1.0; S[:, L] = -1.0
            S[np.arange(K), 3 * L + NSEG + np.arange(K)] = -1.0
            A = np.vstack([A, S]); b = np.concatenate([b, -(net[0] + np.asarray(scen, float))])
            cost[2 * L] = 0.0; cost[3 * L + NSEG:] = w[0] / K
            b[2 * L] = 1e9
        r = linprog(cost, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        if r.status != 0 or r.x is None:
            return None
        out = np.zeros(L); out[:] = r.x[:L] - r.x[L:2 * L]
        return out

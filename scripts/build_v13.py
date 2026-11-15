"""
Genera el lliurament v13 (Mode Difícil) com un agent.py PLA (sense sys/os/importlib, prohibits pel validador):

    v12 (dificil/agent_hard/agent.py)
  + stoch_mpc (dificil/arena/stoch_mpc): MPC estocàstic de dues etapes, 8 escenaris
  + nn_pre   (dificil/arena/nn_pre): previsió CO2/preu amb xarxes pre-entrenades amb historial REE (GPU)
  + guarda de temps: si la mitjana de ms/pas supera `budget_ms`, aquell pas usa el LP determinista del v12.

    python scripts/build_v13.py [n_assumed=1008] [budget_s=40] -> dificil/v13/ (agent.py, feats.py, nnfeat.py, parametres.json,
                                              models_all7.pkl, nn_all.npz) i dificil/lliuraments/hard_v13.zip
"""
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AH = ROOT / "dificil" / "agent_hard"
AR = ROOT / "dificil" / "arena"
OUT = ROOT / "dificil" / "v13"
ZIP = ROOT / "dificil" / "lliuraments" / "hard_v13.zip"
V12ZIP = ROOT / "dificil" / "lliuraments" / "hard_v12_public_23.53.zip"


def body_from(src, marker):
    i = src.index(marker)
    return src[i:]


def main(n_assumed=1008, budget_s=40.0, zip_path=ZIP):
    base = (AH / "agent.py").read_text(encoding="utf-8")
    assert "\nclass BatteryAgent:" in base
    base = base.replace("\nclass BatteryAgent:", "\nclass _V12:")
    base = base.replace("import json, math\n", "import json, math, re, time\nfrom statistics import NormalDist\n"
                        "import scipy.sparse as sp\n", 1)
    base = base.replace("from feats import", "import nnfeat as nf\nfrom feats import", 1)

    stoch = body_from((AR / "stoch_mpc" / "agent.py").read_text(encoding="utf-8"), "class BatteryAgent(Base):")
    stoch = stoch.replace("class BatteryAgent(Base):", "class _Stoch(_V12):", 1)

    nn = (AR / "nn_co2" / "agent.py").read_text(encoding="utf-8")
    nn = body_from(nn, "def variant_of(")
    nn = nn.replace("class BatteryAgent(Base):", "class _NN(_V12):", 1)

    guard = """

class BatteryAgent(_Stoch, _NN):
    \"\"\"v13 = v12 + MPC estocàstic (8 escenaris) + xarxes de CO2/preu pre-entrenades amb historial REE, amb guarda de temps.\"\"\"

    # Guarda de temps: pressupost total `budget_s` per a `n_assumed` passos (test públic: 1.008 h).
    # Modes = rànquing de l'arena (3 plecs); cap és el v12 ja lliurat:
    #   0 stoch8_nn_pre (19,25)   1 nn_pre = LP del v12 + xarxes pre-entrenades (19,16, el terra)
    # A cada pas es fa servir el mode 0 si encara deixa acabar la resta de passos en el mode 1 dins del pressupost.
    MODES = ("stoch8_nn_pre", "nn_pre")

    def __init__(self, params=None):
        p = {"S": 8}
        p.update(params or {})
        super().__init__(p)
        self.p.setdefault("budget_s", __BUDGET__)
        self.p.setdefault("n_assumed", __NASSUMED__)
        self._t_start = None
        self._nst = 0
        self._cost = [0.040, 0.030]                     # cost estimat per pas de cada mode (s), s'actualitza
        self._fast = False
        self.n_mode = [0, 0]

    def _choose(self):
        fm = self.p.get("force_mode")
        if fm is not None:
            return int(fm)
        if self._t_start is None:
            return 0
        if self._nst >= self.p["n_assumed"]:           # més passos dels previstos: sempre el més barat
            return 1
        el = time.perf_counter() - self._t_start
        rem = max(self.p["n_assumed"] - self._nst - 1, 0)
        return 0 if el + self._cost[0] + rem * self._cost[1] < float(self.p["budget_s"]) else 1

    def act(self, obs):
        t0 = time.perf_counter()
        if self._t_start is None:
            self._t_start = t0
        try:
            mode = self._choose()
        except Exception:
            mode = 1
        self.n_mode[mode] += 1
        kf = self.p.get("k_future", 0.0)
        try:
            self._fast = mode == 1
            if mode == 1:                               # LP del v12 amb la seva prudència futura
                self.p["k_future"] = self.p.get("fast_k_future", 0.4)
            a = super().act(obs)
        finally:
            self.p["k_future"] = kf; self._fast = False
        dt = time.perf_counter() - t0
        self._cost[mode] = 0.9 * self._cost[mode] + 0.1 * dt
        self._nst += 1
        return a

    def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
        if self._fast:
            return _V12._lp(self, w, net, soc0, v_term, dmax0, scen)
        return super()._lp(w, net, soc0, v_term, dmax0, scen)
"""
    guard = guard.replace("__NASSUMED__", str(int(n_assumed))).replace("__BUDGET__", repr(float(budget_s)))
    code = base.rstrip() + "\n\n\n" + stoch.rstrip() + "\n\n\n" + nn.rstrip() + "\n" + guard
    code = code.replace('"""\nBatteryAgent — Mode Difícil', '"""\nBatteryAgent v13 — Mode Difícil', 1)
    for bad in ("import sys", "importlib", "sys.path", "import os"):
        assert bad not in code, bad
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "agent.py").write_text(code, encoding="utf-8")
    shutil.copy(AH / "feats.py", OUT / "feats.py")
    shutil.copy(AR / "nn_co2" / "nnfeat.py", OUT / "nnfeat.py")
    shutil.copy(AH / "parametres.json", OUT / "parametres.json")
    shutil.copy(AR / "nn_pre" / "nn_all.npz", OUT / "nn_all.npz")   # xarxes pre-entrenades (REE)
    with zipfile.ZipFile(V12ZIP) as z:                 # models exactes del v12 (historial inicial complet)
        (OUT / "models_all7.pkl").write_bytes(z.read("models_all7.pkl"))
    files = ["agent.py", "feats.py", "nnfeat.py", "parametres.json", "models_all7.pkl", "nn_all.npz"]
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(OUT / f, f)
    print("desat", zip_path, f"{zip_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    import sys as _s
    n = int(_s.argv[1]) if len(_s.argv) > 1 else 1008
    b = float(_s.argv[2]) if len(_s.argv) > 2 else 40.0
    main(n, b, ZIP)

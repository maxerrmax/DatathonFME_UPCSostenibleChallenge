"""
Genera el lliurament v14 (v12 + MPC estocàstic, SENSE xarxes neuronals; el v13 amb xarxes va baixar al públic: 23,07)

Basat en el generador del v13 (Mode Difícil) com un agent.py PLA (sense sys/os/importlib, prohibits pel validador):

    v12 (dificil/agent_hard/agent.py)
  + stoch_mpc (dificil/arena/stoch_mpc): MPC estocàstic de dues etapes, 8 escenaris
  + nn_pre   (dificil/arena/nn_pre): previsió CO2/preu amb xarxes pre-entrenades amb historial REE (GPU)
  + guarda de temps: si la mitjana de ms/pas supera `budget_ms`, aquell pas usa el LP determinista del v12.

    python scripts/build_v14.py [n_assumed=1008] [budget_s=40] -> dificil/arxiu/v14/ (agent.py, feats.py, parametres.json,
                                              models_all7.pkl) i dificil/arxiu/lliuraments/hard_v14.zip
"""
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AH = ROOT / "dificil" / "agent_hard"
AR = ROOT / "dificil" / "arena"
OUT = ROOT / "dificil" / "arxiu" / "v14"
ZIP = ROOT / "dificil" / "arxiu" / "lliuraments" / "hard_v14.zip"
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

    stoch = body_from((AR / "stoch_mpc" / "agent.py").read_text(encoding="utf-8"), "class BatteryAgent(Base):")
    stoch = stoch.replace("class BatteryAgent(Base):", "class _Stoch(_V12):", 1)

    guard = """

class BatteryAgent(_Stoch):
    \"\"\"v14 = v12 (mateixes previsions LightGBM i calibratge de CO2) + MPC estocàstic de dues etapes, amb guarda de temps.\"\"\"

    # Guarda de temps: pressupost total `budget_s` per a `n_assumed` passos (test públic: 1.008 h).
    # Modes (arena, 3 plecs; cap és el v12 ja lliurat): 0 stoch_s12 (19,13)  1 stoch_mpc S=8 (19,11)  2 stoch_s6 (19,07, el terra)
    # A cada pas es tria el mode més car que encara deixa acabar la resta de passos en el mode 2 dins del pressupost.
    MODES = ("stoch_s12", "stoch_s8", "stoch_s6")
    SIZES = (12, 8, 6)

    def __init__(self, params=None):
        p = {"S": 12}
        p.update(params or {})
        super().__init__(p)
        self.p.setdefault("budget_s", __BUDGET__)
        self.p.setdefault("n_assumed", __NASSUMED__)
        self._Z = {}
        for S in self.SIZES:                            # escenaris idèntics als de cada estratègia de l'arena
            self.p["S"] = S; self._mkscen(); self._Z[S] = (self.Zc, self.Zp, self.Zn)
        self.p["S"] = 12; self.Zc, self.Zp, self.Zn = self._Z[12]
        self._t_start = None
        self._nst = 0
        self._cost = [0.040, 0.034, 0.030]              # cost estimat per pas de cada mode (s), s'actualitza
        self.n_mode = [0, 0, 0]

    def _choose(self):
        fm = self.p.get("force_mode")
        if fm is not None:
            return int(fm)
        if self._t_start is None:
            return 0
        if self._nst >= self.p["n_assumed"]:           # més passos dels previstos: sempre el més barat
            return 2
        el = time.perf_counter() - self._t_start
        rem = max(self.p["n_assumed"] - self._nst - 1, 0)
        for m in (0, 1):
            if el + self._cost[m] + rem * self._cost[2] < float(self.p["budget_s"]):
                return m
        return 2

    def act(self, obs):
        t0 = time.perf_counter()
        if self._t_start is None:
            self._t_start = t0
        try:
            mode = self._choose()
        except Exception:
            mode = 2
        self.n_mode[mode] += 1
        try:
            S = self.SIZES[mode]
            self.p["S"] = S; self.Zc, self.Zp, self.Zn = self._Z[S]
            a = super().act(obs)
        finally:
            self.p["S"] = 12; self.Zc, self.Zp, self.Zn = self._Z[12]
        dt = time.perf_counter() - t0
        self._cost[mode] = 0.9 * self._cost[mode] + 0.1 * dt
        self._nst += 1
        return a
"""
    guard = guard.replace("__NASSUMED__", str(int(n_assumed))).replace("__BUDGET__", repr(float(budget_s)))
    code = base.rstrip() + "\n\n\n" + stoch.rstrip() + "\n" + guard
    code = code.replace('"""\nBatteryAgent — Mode Difícil', '"""\nBatteryAgent v14 — Mode Difícil', 1)
    for bad in ("import sys", "importlib", "sys.path", "import os"):
        assert bad not in code, bad
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "agent.py").write_text(code, encoding="utf-8")
    shutil.copy(AH / "feats.py", OUT / "feats.py")
    shutil.copy(AH / "parametres.json", OUT / "parametres.json")
    with zipfile.ZipFile(V12ZIP) as z:                 # models exactes del v12 (historial inicial complet)
        (OUT / "models_all7.pkl").write_bytes(z.read("models_all7.pkl"))
    files = ["agent.py", "feats.py", "parametres.json", "models_all7.pkl"]
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

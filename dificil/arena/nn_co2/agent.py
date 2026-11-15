"""nn_co2: v12 + previsió CO2 (i preu) barrejada amb xarxes neuronals (MLP/TCN) entrenades a la GPU.

Llegeix nn_<variant>.npz d'aquesta carpeta: variant = f<k> si params["models"] és models_f<k>.pkl / models_f<k>7.pkl,
si no "all". Només numpy. Qualsevol fallada -> previsió base.
Paràmetres: nn_mix (0.5), nn_price (True), nn_mix_price (0.3).
"""
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import load_base
import nnfeat as nf

Base = load_base().BatteryAgent


def variant_of(models):
    m = re.search(r"models_f(\d)", str(models))
    return f"f{m.group(1)}" if m else "all"


class BatteryAgent(Base):
    NN_DIR = HERE
    DEFAULTS = {"nn_mix": 0.5, "nn_price": True, "nn_mix_price": 0.3}

    def __init__(self, params=None):
        super().__init__(params)
        for k, v in self.DEFAULTS.items():
            self.p.setdefault(k, v)
        self._nn = None
        try:
            f = Path(self.NN_DIR) / f"nn_{variant_of(self.p.get('models'))}.npz"
            if f.exists():
                self._nn = nf.NNBundle(f)
        except Exception:
            self._nn = None

    def _forecast(self, t, price3, co2_3):
        price, co2, load, pv = super()._forecast(t, price3, co2_3)
        if self._nn is None or len(co2_3) < 3 or len(price3) < 3:
            return price, co2, load, pv
        try:
            sl = slice(nf.H0, nf.H1 + 1)
            if self.p["nn_mix"] > 0:
                x = nf.build(self.hist, t, "co2")
                if x is not None:
                    pr = self._nn.predict("co2", x, nf.last_known(self.hist, t, "co2"))
                    if pr is not None and np.isfinite(pr).all():
                        a = float(self.p["nn_mix"])
                        co2 = co2.copy(); co2[sl] = np.clip(a * pr + (1 - a) * co2[sl], 0.0, 1.0)
            if self.p["nn_price"] and self.p["nn_mix_price"] > 0:
                x = nf.build(self.hist, t, "price")
                if x is not None:
                    pr = self._nn.predict("price", x, nf.last_known(self.hist, t, "price"))
                    if pr is not None and np.isfinite(pr).all():
                        a = float(self.p["nn_mix_price"])
                        price = price.copy(); price[sl] = a * pr + (1 - a) * price[sl]
        except Exception:
            pass
        return price, co2, load, pv

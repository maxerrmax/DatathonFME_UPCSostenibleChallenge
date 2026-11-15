"""shrink_far: v12 amb les previsions de CO2 i preu llunyanes (h>=shrink_start) encongides cap a un prior robust.

Prior = mitjana de (a) perfil d'ahir a la mateixa hora desplacat additivament pel desnivell entre els valors
coneguts ara i els d'ahir a les mateixes hores (retallat) i (b) mediana per hora del dia dels ultims 7 dies.
Pes = shrink_max * min(1, (h-shrink_start+1)/(23-shrink_start+1)), creixent amb l'horitzo.
Params: shrink_max (0.3), shrink_start (8), shrink_co2 (1), shrink_price (1), shift_clip_co2 (0.08),
        shift_clip_price (0.1), prior_mix (0.5, pes del perfil d'ahir).
Causal: nomes self.hist. Si falten dades, no s'encongeix.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base
import numpy as np

Base = load_base().BatteryAgent


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__(params)

    def _prior(self, key, t, hs, known, clip):
        g = self.hist.get
        diffs = [known[i] - g(key, t + i - 24) for i in range(min(3, len(known)))]
        diffs = [d for d in diffs if np.isfinite(d)]
        shift = float(np.clip(np.median(diffs), -clip, clip)) if diffs else 0.0
        out = np.full(len(hs), np.nan)
        mix = self.p.get("prior_mix", 0.5)
        for n, h in enumerate(hs):
            y = g(key, t + h - 24)
            vals = [g(key, t + h - 24 * d) for d in range(1, 8)]
            vals = [v for v in vals if np.isfinite(v)]
            med = float(np.median(vals)) if len(vals) >= 3 else np.nan
            ys = y + shift if np.isfinite(y) else np.nan
            if np.isfinite(ys) and np.isfinite(med):
                out[n] = mix * ys + (1 - mix) * med
            elif np.isfinite(med):
                out[n] = med
        return out

    def _forecast(self, t, price3, co2_3):
        price, co2, load, pv = super()._forecast(t, price3, co2_3)
        try:
            s0 = int(self.p.get("shrink_start", 8)); smax = float(self.p.get("shrink_max", 0.3))
            hs = list(range(max(s0, 3), len(co2)))
            if hs and smax > 0:
                wgt = smax * np.minimum(1.0, (np.array(hs) - s0 + 1) / max(1.0, 23 - s0 + 1))
                for key, arr, on, clip, known, hi in (
                        ("co2", co2, "shrink_co2", self.p.get("shift_clip_co2", 0.08), co2_3, 1.0),
                        ("price", price, "shrink_price", self.p.get("shift_clip_price", 0.1), price3, 10.0)):
                    if not self.p.get(on, 1):
                        continue
                    pr = self._prior(key, t, hs, list(known), clip)
                    ok = np.isfinite(pr)
                    new = arr[hs].copy()
                    new[ok] = (1 - wgt[ok]) * new[ok] + wgt[ok] * pr[ok]
                    arr[hs] = np.clip(new, 0.0, hi)
        except Exception:
            pass
        return price, co2, load, pv

"""Agent DQN: xarxa Q entrenada amb stable-baselines3 i executada en numpy pur (sense torch)."""
from pathlib import Path
import numpy as np
from common import ACTIONS, wvec, feat, clamp


class BatteryAgent:
    def __init__(self):
        z = np.load(Path(__file__).parent / "qnet.npz")
        idx = sorted({int(k.split("_")[0]) for k in z.files})
        self.layers = [(z[f"{i}_weight"], z[f"{i}_bias"]) for i in idx]

    def act(self, obs):
        soc = float(obs["soc_kwh"])
        h = feat(soc, wvec(obs["co2_properes_24h"], obs["preus_properes_24h"]))
        for k, (Wt, b) in enumerate(self.layers):
            h = Wt @ h + b
            if k < len(self.layers) - 1:
                h = np.maximum(h, 0)
        return clamp(soc, float(ACTIONS[int(np.argmax(h))]))

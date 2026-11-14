"""Agent RL (Fitted Q-Iteration amb LightGBM): tria l'acció discreta amb Q màxima."""
from pathlib import Path
import numpy as np, joblib
from common import ACTIONS, wvec, feat, clamp


class BatteryAgent:
    def __init__(self):
        self.m = joblib.load(Path(__file__).parent / "model.pkl")

    def act(self, obs):
        soc = float(obs["soc_kwh"])
        f = feat(soc, wvec(obs["co2_properes_24h"], obs["preus_properes_24h"]))
        X = np.column_stack([np.tile(f, (len(ACTIONS), 1)), ACTIONS / 500.0])
        q = self.m.predict(X)
        return clamp(soc, float(ACTIONS[int(np.argmax(q))]))

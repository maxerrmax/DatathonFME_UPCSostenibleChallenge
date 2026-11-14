"""
Simulador Mode Difícil: hereta l'environment OFICIAL i només canvia l'observació
(finestra de mercat de 3 h, meteo de l'última hora, cap_max_kwh/power_max_kw).
La física i la puntuació són les oficials.
"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter_kit"))
from environment import CampusBatteryEnv  # noqa: E402
sys.path.pop(0)

SPEC_HARD = {"capacity_kwh": 6000.0, "power_kw": 1500.0, "efficiency": 0.90, "soc_init": 0.50,
             "soc_min": 0.0, "soc_max": 1.0, "dt_hours": 1.0, "forecast_horizon": 3}
METEO_KEYS = ["irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms"]


class HardEnv(CampusBatteryEnv):
    def __init__(self, hourly, spec=None):
        spec = spec or SPEC_HARD
        self.H = int(spec.get("forecast_horizon", 3))
        super().__init__(hourly, spec=spec)

    def _get_obs(self):
        obs = super()._get_obs()
        if obs is None:
            return None
        h_end = min(self.t + self.H, self.n)
        obs["preus_properes_24h"] = self.df["precio_eur_per_kwh"].iloc[self.t:h_end].values.tolist()
        obs["co2_properes_24h"] = self.df["co2_kg_per_kwh"].iloc[self.t:h_end].values.tolist()
        i = self.t - 1 if self.t > 0 else self.t
        for k in METEO_KEYS:
            if k in self.df.columns:
                v = self.df[k].iloc[i]
                obs["noves_dades_5min"][k] = float(v) if pd.notna(v) else float("nan")
        obs["cap_max_kwh"] = self.cap
        obs["power_max_kw"] = self.power
        return obs


def run(agent, df, mask=None):
    env = HardEnv(df.reset_index()); o = env.reset(); d = False
    import time; t0 = time.time()
    while not d:
        o, d = env.step(agent.act(o))
    r = env.evaluate_results(mask=mask)
    r["score_web"] = round(max(0.0, 0.7 * r["co2_estalviat_pct"] + 0.3 * r["cost_estalviat_pct"] - r["infraccions"]), 3)
    r["ms_per_step"] = round(1000 * (time.time() - t0) / len(df), 2)
    r["_actions"] = np.array(env.actions)
    return r

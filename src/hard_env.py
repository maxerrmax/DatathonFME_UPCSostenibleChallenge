"""
Entorn local del Mode Difícil.

Estén el CampusBatteryEnv oficial amb el que descriu la web per al Mode Difícil:
  - previsions de preu i CO2 només de 3 hores (hora actual + 2 següents),
  - lectures meteo de la darrera hora dins de obs["noves_dades_5min"],
  - "cap_max_kwh" i "power_max_kw" dins de obs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))
from environment import CampusBatteryEnv  # noqa: E402

SPEC_HARD = {
    "capacity_kwh": 6000.0,
    "power_kw": 1500.0,
    "efficiency": 0.90,
    "soc_init": 0.50,
    "soc_min": 0.0,
    "soc_max": 1.0,
    "dt_hours": 1.0,
    "forecast_horizon": 3,
}
SPEC_STD = {
    "capacity_kwh": 2000.0,
    "power_kw": 500.0,
    "efficiency": 0.90,
    "soc_init": 0.50,
    "soc_min": 0.0,
    "soc_max": 1.0,
    "dt_hours": 1.0,
    "forecast_horizon": 24,
}
METEO = ["irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms"]


class HardEnv(CampusBatteryEnv):
    def __init__(self, hourly_data: pd.DataFrame, spec: dict = None, export_price: float = 0.0):
        spec = dict(SPEC_HARD if spec is None else spec)
        self.horizon = int(spec.get("forecast_horizon", 24))
        super().__init__(hourly_data, spec=spec, export_price=export_price)

    def _get_obs(self):
        obs = super()._get_obs()
        if obs is None:
            return None
        obs["preus_properes_24h"] = obs["preus_properes_24h"][: self.horizon]
        obs["co2_properes_24h"] = obs["co2_properes_24h"][: self.horizon]
        prev = self.t - 1 if self.t > 0 else 0
        for col in METEO:
            if col in self.df.columns:
                obs["noves_dades_5min"][col] = float(self.df[col].iloc[prev])
        obs["cap_max_kwh"] = self.cap
        obs["power_max_kw"] = self.power
        return obs


def run(agent, data: pd.DataFrame, spec: dict = None):
    env = HardEnv(data, spec=spec)
    obs = env.reset()
    done = False
    while not done:
        obs, done = env.step(agent.act(obs))
    res = env.evaluate_results()
    res["actions"] = np.asarray(env.actions)
    return res

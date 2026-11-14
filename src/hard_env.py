"""
src/hard_env.py
Simulador local del MODE DIFICIL. Reutilitza al 100% la fisica i la metrica de
`starter_kit.environment.CampusBatteryEnv` (no es modifica) i nomes ajusta l'observacio:

  * preus_properes_24h / co2_properes_24h retallats a spec["forecast_horizon"] (3 h)
  * obs["cap_max_kwh"], obs["power_max_kw"]
  * meteo de la darrera hora (fila t-1) dins obs["noves_dades_5min"]

Us:
    env = HardModeEnv(df_hard, spec=SPEC_HARD)     # df_hard["pv_kw"] ja escalat x12
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from starter_kit.environment import CampusBatteryEnv  # noqa: E402
except Exception:                       # Kaggle: environment.py al costat d'aquest fitxer
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
METEO_KEYS = ("irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms")


class HardModeEnv(CampusBatteryEnv):
    def __init__(self, hourly_data, spec=None, export_price=0.0):
        super().__init__(hourly_data, spec=dict(spec or SPEC_HARD), export_price=export_price)

    def _get_obs(self):
        obs = super()._get_obs()
        if obs is None:
            return None
        h = int(self.spec.get("forecast_horizon", 24))
        obs["preus_properes_24h"] = obs["preus_properes_24h"][:h]
        obs["co2_properes_24h"] = obs["co2_properes_24h"][:h]
        obs["cap_max_kwh"] = self.cap
        obs["power_max_kw"] = self.power
        row = self.df.iloc[self.t - 1] if self.t > 0 else self.df.iloc[self.t]
        for k in METEO_KEYS:
            if k in self.df.columns:
                obs["noves_dades_5min"][k] = float(row[k])
        return obs


def run_hard(agent, df_hard, spec=None, return_env=False, desc=None):
    """Simulacio sequencial completa en Mode Dificil. Retorna el diccionari de metriques.
    `desc`: si es dona (i hi ha tqdm), mostra una barra de progres dels passos."""
    import time
    env = HardModeEnv(df_hard, spec=spec)
    obs = env.reset()
    done = False
    lat = []
    bar = None
    if desc:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=env.n, desc=desc, leave=False, mininterval=0.5)
        except Exception:
            bar = None
    while not done:
        t0 = time.perf_counter()
        a = agent.act(obs)
        lat.append(time.perf_counter() - t0)
        obs, done = env.step(a)
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    res = env.evaluate_results()
    res["ms_per_step_mean"] = round(1000 * sum(lat) / len(lat), 3)
    res["ms_per_step_max"] = round(1000 * max(lat), 3)
    return (res, env) if return_env else res

"""
src/run_hard_sim.py
Simulacio local en MODE DIFICIL de l'agent lliurable (src/agent.py + src/model_bateria.pkl).

    python src/run_hard_sim.py            # 12 blocs de validacio + train complet
    python src/run_hard_sim.py --stock    # tambe amb l'entorn oficial sense modificar

Nota: model_bateria.pkl final esta entrenat amb TOT train, de manera que aquestes
puntuacions son in-sample (serveixen per comprovar infraccions, robustesa i latencia).
La puntuacio honesta (models entrenats sense els blocs) la calcula train_pipeline.ipynb.
"""
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

import agent as A  # noqa: E402
import data_prep as D  # noqa: E402
import train_lib as T  # noqa: E402
from hard_env import METEO_KEYS, SPEC_HARD, run_hard  # noqa: E402
from starter_kit.environment import run_agent_simulation  # noqa: E402


def main():
    hourly = D.build_hourly()
    df_hard = D.build_env_frame()                 # pv_kw ja escalat x12
    val_df, blocks = T.make_validation_frame(df_hard, hourly)
    rows = {}
    for name, df in (("12 blocs de validacio", val_df), ("train complet", df_hard)):
        ag = A.BatteryAgent()
        assert ag.ready, "falta src/model_bateria.pkl (executeu train_pipeline.ipynb)"
        r = run_hard(ag, df, spec=SPEC_HARD)
        r["fallbacks"] = ag.n_fallback
        rows[name] = r
    if "--stock" in sys.argv:
        ag = A.BatteryAgent()
        r = run_agent_simulation(ag, df_hard.drop(columns=list(METEO_KEYS)), spec=SPEC_HARD)
        r["fallbacks"] = ag.n_fallback
        rows["entorn oficial sense modificar"] = r
    cols = ["score", "co2_estalviat_pct", "cost_estalviat_pct", "infraccions", "fallbacks",
            "ms_per_step_mean", "ms_per_step_max"]
    out = pd.DataFrame(rows).T
    print(out[[c for c in cols if c in out.columns]].to_string())


if __name__ == "__main__":
    main()

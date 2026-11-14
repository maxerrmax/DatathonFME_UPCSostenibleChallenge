# Difficult Mode solution — UPC Sostenible Challenge (Datathon FME 2026)

Forecasting + rolling-horizon control agent for **Mode Difícil**: 6,000 kWh / ±1,500 kW
battery, η = 0.90, solar ×12, and only **3 hours** of price / CO₂ look-ahead.

| File | Purpose | In `submission.zip` |
|---|---|---|
| `agent.py` | `BatteryAgent`: online history buffer, feature builders, forecasters, planner, safety clamp | yes |
| `model_bateria.pkl` | Artifact: train history, clear-sky table, LightGBM + Ridge models, control parameters | yes |
| `train_pipeline.ipynb` | **Self-contained** Kaggle / Colab notebook: preprocessing, model comparison, tuning, export | no |
| `data_prep.py`, `hard_env.py`, `train_lib.py` | Offline preprocessing, Difficult Mode simulator wrapper, training library (also embedded in the notebook) | no |
| `run_hard_sim.py` | Local Difficult Mode simulation of the deliverable agent | no |
| `make_notebook.py` | Regenerates the notebook from the `.py` sources | no |
| `IMPLEMENTATION.md` | Architecture and validation details | no |

> `model_bateria.pkl` / `submission.zip` currently in this folder are a **provisional build**
> produced locally with the default parameters. Running the notebook on Kaggle regenerates
> both (with the model family and control parameters you pick there).

## 1. Train on Kaggle or Google Colab

The notebook is a single file and runs unchanged on Kaggle, Colab or locally. Its first cells
write `agent.py`, the official `environment.py` and the helper modules into `hm_build/`, and
locate the data automatically (loose CSVs or a `.zip` containing them).

1. Provide the six CSV files of `data/train/`:
   * **Kaggle:** create a *Dataset* with them (or with a zip) and attach it to the notebook.
     No Internet required.
   * **Colab:** drop the CSVs or a `data.zip` into the file panel (`/content`), or keep them in
     Google Drive and set `MOUNT_DRIVE = True`. If nothing is found the notebook opens an upload
     dialog. CatBoost is pip-installed automatically if missing.
   * Or set `DATA_DIR` to the folder explicitly.
2. Upload `src/train_pipeline.ipynb`. GPU is optional and auto-detected.
3. *Run all*. Settings are in the first cell:
   * `FAST = True` for a quick dry run (fewer trees, no CatBoost / RandomForest).
   * `CHOSEN = "blend" | "lgb" | "ridge"` — the forecaster exported into the agent. Section 6 of
     the notebook prints the official Difficult Mode score of every model family on our
     12-block validation set so you can choose.
4. Download `hm_build/submission.zip` (contains `agent.py` + `model_bateria.pkl`) from the
   notebook output. For local use, copy `hm_build/model_bateria.pkl` into `src/`.

If you change any `.py` in `src/`, run `python src/make_notebook.py` to refresh the notebook.

## 2. Simulate Difficult Mode locally

```bash
pip install numpy pandas scipy scikit-learn joblib lightgbm
python src/run_hard_sim.py            # 12 validation blocks + full train, 3 h horizon
python src/run_hard_sim.py --stock    # also against the unmodified starter_kit environment
```

Or directly, exactly as specified by the organisers:

```python
import sys; sys.path[:0] = ["src", "."]
from starter_kit.environment import CampusBatteryEnv
from data_prep import build_env_frame          # hourly table, pv_kw already x12
from hard_env import HardModeEnv, SPEC_HARD    # SPEC_HARD: 6000 kWh, 1500 kW, forecast_horizon 3
from agent import BatteryAgent

df_hard = build_env_frame()
env = HardModeEnv(df_hard, spec=SPEC_HARD)     # subclass of CampusBatteryEnv (3 h lists + weather in obs)
obs, done, agent = env.reset(), False, BatteryAgent()
while not done:
    obs, done = env.step(agent.act(obs))
print(env.evaluate_results())
```

`HardModeEnv` only changes the observation (truncates the price / CO₂ lists to
`forecast_horizon`, adds `cap_max_kwh`, `power_max_kw` and last-hour weather); physics and
scoring are inherited unchanged from `starter_kit/environment.py`. The agent also runs on the
plain `CampusBatteryEnv(df_hard, spec=spec_hard)`.

Scores from the provisional `model_bateria.pkl` on train data are **in-sample**; the honest
estimate is the notebook's "validació 12 blocs" row (models trained without those blocks).

## 3. Verify the submission

```bash
cd starter_kit
python check_submission.py ../src/submission.zip
```

Expected: security scan OK, `BatteryAgent` found, 24 test steps, "LA SOLUCIÓ ÉS VÀLIDA".

## Results (local development run, same code as the notebook)

| Simulation (Difficult Mode) | Score | ΔCO₂ | ΔCost | Infractions |
|---|---|---|---|---|
| **Agent, 12-block validation (models trained without the blocks)** | **20.04** | 18.90 % | 22.71 % | 0 |
| Perfect-foresight LP (upper bound) | 21.92 | 21.08 % | 23.89 % | 0 |
| Rule-based fallback (no model) | 10.82 | 10.52 % | 11.50 % | 0 |
| Agent, synthetic November block only | 16.76 | 16.45 % | 17.48 % | 0 |
| Agent, full train, final artifact (in-sample) | 19.53 | 18.40 % | 22.19 % | 0 |

Latency with the default DP planner: 3.5 to 9 ms mean per step on a laptop CPU depending on
background load (isolated spikes of tens of ms); the `scipy.linprog` planner is roughly twice
as slow. Details in `IMPLEMENTATION.md`.

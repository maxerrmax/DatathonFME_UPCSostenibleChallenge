From the instructions and the repository files (`starter_kit/`, `ENUNCIAT.md`, `data/train/`), build an end-to-end ML forecasting + rolling-horizon control pipeline exclusively for **Mode Difícil (Difficult Mode)** of the **Datathon FME — UPC Sostenible Challenge**.

## Context & Difficult Mode Specification (Read Carefully)

- **Evaluation Interface (`agent.py` & `environment.py`):** During evaluation, `CampusBatteryEnv` steps sequentially through 12 concatenated 7-day blocks (2,016 hourly steps total) without resetting `soc_kwh` between weeks. At each hourly step $t$, `BatteryAgent.act(self, obs) -> float` receives a single observation dict `obs` configured strictly for **Difficult Mode**:
  - **Battery & Solar Scale:** `obs["cap_max_kwh"] = 6000.0` ($6\text{ MWh}$), `obs["power_max_kw"] = 1500.0` ($\pm 1.5\text{ MW}$), round-trip efficiency $\eta = 0.90$ ($\eta_c = \eta_d = \sqrt{0.90}$), and solar PV scaled **$\times 12$** (up to $\sim 1,650\text{ kW}$ peak).
  - **Truncated 3-Hour Market Lookahead (`forecast_horizon = 3`):** `obs["preus_properes_24h"]` and `obs["co2_properes_24h"]` provide **only the next 3 hours** ($t, t+1, t+2$). Planning a full 24-hour cycle for the $6,000\text{ kWh}$ battery requires **ML forecasting models for hours `+3` to `+24`** across prices, $\text{CO}_2$, load, and solar.
  - **Live Weather & Telemetry in `obs["noves_dades_5min"]`:** Contains `ultim_consum_kw`, `ultima_solar_kw` (already scaled $\times 12$), plus real-time weather station readings from `meteo.csv`: `irradiancia_wm2` ($\text{W/m}^2$), `temp_ambient_c` ($^\circ\text{C}$), `temp_panel_c` ($^\circ\text{C}$), and `vent_ms` ($\text{m/s}$).
- **Why Accurate 24h Forecasting Across All 4 Series Is Critical in Difficult Mode:**
  1. **Free Solar Surplus Absorption (`load_kw - pv_kw < 0`):** On sunny middays, $12\times$ solar exceeds campus consumption by up to $1,000\text{ kW}$, generating free, $0\text{ kg CO}_2$ surplus that is otherwise exported at $0\text{ €}$.
  2. **Zero-Export Discharge Ceiling (`600–1,300 kW` net load):** Because max discharge is $-1,500\text{ kW}$ while net campus load (`load_kw - pv_kw`) typically sits between $600\text{ kW}$ and $1,300\text{ kW}$, discharging blindly at $-1,500\text{ kW}$ dumps excess energy into the grid for $0\text{ €}$ and $0\text{ kg CO}_2$ saved.
  3. **+3h to +24h Price & $\text{CO}_2$ Forecasting:** Since `obs` only reveals 3 hours of exact prices and $\text{CO}_2$, the agent must forecast hours $k \in \{3, \dots, 23\}$ (anchored smoothly onto the exact $k \in \{0, 1, 2\}$ values) to decide whether to charge overnight, absorb midday solar, or hold SOC for evening peaks.

## Preprocessing & Online Feature Engineering

Build an offline preprocessing pipeline (scaling `pv_kw * 12.0` on `data/train/`) AND an identical **incremental online state buffer (`self.historial`)** inside `BatteryAgent` so features compute step-by-step at test time without static test DataFrames:

- **Sensor Cleaning & Missing-November Handling:**
  - **Missing November 2025 & Late August 2026:** In `data/train/`, `precio_pvpc.csv` and `co2_red.csv` cover 6,768 hours across all 12 months, whereas `contador.csv` and `meteo.csv` are **completely missing November 2025** (6,187 hours), and `inversor.csv` stops on August 16, 2026 (5,994 hours). Note that `vent_ms` (`VELOCIDAD VIENTO`) is identically `0.0` in `data/train/`.
  - **November 2025 in the Test Set:** Because November 2025 is missing from `data/train/` telemetry, expect a November 2025 week in `data/test_hidden/` with zero prior November meter/PV/weather history:
    - **Offline Training:** Do **not** linearly interpolate or forward-fill across the 30-day November gap (that corrupts `24h` and `168h` lags entering December). Mask/drop training rows whose lag windows cross the Oct $\rightarrow$ Dec outage boundary.
    - **Online Cold-Start Priors:** Seed November's expected load, solar ($12\times$), and irradiance profiles in `self.historial` using the **October/December academic-term hourly & day-of-week profiles** whenever `agent.py` encounters November timestamps.
  - Replace `0.0` sensor dropouts in `load_kw` with `NaN` prior to lagging, and compare clean imputation strategies (same-month/day-of-week/hour median profile vs. short-gap interpolation $\le 3\text{h}$).
- **Multi-Target Lags & Rolling Statistics (Seam-Aware):**
  - Track rolling histories in `self.historial` for all core signals: `load_kw`, `pv_kw` ($12\times$), `irradiancia_wm2`, `temp_ambient_c`, `temp_panel_c`, `precio_eur_per_kwh`, and `co2_kg_per_kwh`.
  - Compute lags ($1\text{h}, 2\text{h}, 3\text{h}, 24\text{h}, 48\text{h}, 168\text{h}$) and rolling statistics (`mean`, `std`, `min`, `max` over $24\text{h}$ and $168\text{h}$). Note: `load_kw` has $r_{168\text{h}} = 0.873$, `pv_kw` correlates $r = 0.997$ with `irradiancia_wm2` and $r = 0.882$ with `temp_panel_c`, and `co2_kg_per_kwh` has $r_{24\text{h}} = 0.841$.
  - **Test Week Seam Handling:** Detect timestamp jumps (`ts - prev_ts > 1 hour`) across the 12 concatenated test weeks so lag buffers fall back cleanly to regime priors at week boundaries without corrupting rolling windows, while preserving `soc_kwh` carryover across months.
- **Cyclical & Barcelona/UPC Calendar Features (`is_x` binary flags):**
  - Cyclical `sin`/`cos` encodings for hour-of-day ($24\text{h}$), day-of-week ($7\text{d}$), and day-of-year ($365.25\text{d}$).
  - Binary `is_x` flags computable purely from `timestamp` for 2025–2026: `is_weekend`, `is_bcn_holiday` (Catalonia & Barcelona public holidays: *Diada* Sept 11, *La Mercè* Sept 24, *Sant Joan* June 24, *Sant Esteve* Dec 26, *Tots Sants* Nov 1, *Puríssima* Dec 8, *Reis* Jan 6, *Setmana Santa*, etc.), `is_august_recess` (August low-load regime), `is_exam_period` (UPC Jan & May–June exam weeks), and `is_datathon_weekend` (Nov 14–15, 2026).
- **Validation Strategy (Mirroring Difficult Mode Hidden Test Set):**
  - Construct `df_hard` by scaling `pv_kw * 12.0` and initializing `CampusBatteryEnv(df_hard, spec=spec_hard)` with:
    ```python
    spec_hard = {
        "capacity_kwh": 6000.0,
        "power_kw": 1500.0,
        "efficiency": 0.90,
        "soc_init": 0.50,
        "soc_min": 0.0,
        "soc_max": 1.0,
        "dt_hours": 1.0,
        "forecast_horizon": 3
    }
    ```
  - Use a **12-block held-out validation split** (days 17–23 of every available month in `data/train/` as contiguous 168-hour blocks, PLUS a synthetic November 2025 block built from real November `precio_pvpc.csv` + `co2_red.csv` paired with Oct/Dec load/PV/weather profiles) concatenated together to mirror `data/test_hidden/`.

## Modeling & Control Optimization

1. **Offline Training Notebook (`train_pipeline.ipynb` — Kaggle / CUDA Compatible):**
   - Train and benchmark multi-horizon forecasting models (direct/recursive LightGBM, XGBoost, CatBoost/RandomForest, Ridge/SARIMAX baseline) for `+3h` to `+24h` predictions of:
     - **Campus Load ($\hat{L}_{t+k}$)** and **Solar PV ($\widehat{\text{PV}}_{t+k}$, scaled $\times 12$)** using weather telemetry (`irradiancia_wm2`, `temp_panel_c`, `temp_ambient_c`) + lags + calendar flags.
     - **PVPC Price ($\widehat{\text{Price}}_{t+k}$)** and **Grid $\text{CO}_2$ ($\widehat{\text{CO}}_{2,t+k}$)** for $k \in \{3, \dots, 23\}$, anchored smoothly onto the exact observed values at $k \in \{0, 1, 2\}$.
   - Tune rolling-horizon LP/DP hyperparameters (terminal SOC shadow price $V_{\text{term}}$ by weekday/weekend/month regime, net-load export safety buffer, and forecast uncertainty discounting for $k \ge 3$) directly against the Difficult Mode `CampusBatteryEnv` competition metric ($70\%$ $\text{CO}_2$ reduction + $30\%$ cost reduction).
   - Export lightweight, self-contained model artifacts/lookup tables for `agent.py`.
2. **Online Control Agent (`agent.py`):**
   - Assemble the 24-hour planning horizon at each step $t$:
     - Exact ground truth for $k \in \{0, 1, 2\}$ from `obs["preus_properes_24h"]` and `obs["co2_properes_24h"]`, combined with ML forecasts for $k \in \{3, \dots, 23\}$ (handling end-of-simulation tapering cleanly).
     - Forecasted 24-hour net load $\hat{L}_{\text{net}, k} = \hat{L}_{t+k} - \widehat{\text{PV}}_{t+k}$ (incorporating step-0 real-time `ultim_consum_kw - ultima_solar_kw`).
   - Solve a fast 24-step **Linear Program (`scipy.optimize.linprog`, HiGHS)** with explicit grid-import slack variable $G_k \ge 0$:
     - Constraint: $G_k \ge \hat{L}_{\text{net}, k} + P_{c,k} - P_{d,k}$ and $G_k \ge 0$.
     - Because $G_k \ge 0$ is penalized in the objective ($0.70 \frac{\widehat{\text{CO}}_{2,k}}{\overline{\text{CO}}_{2,\text{base}}} + 0.30 \frac{\widehat{\text{Price}}_k}{\overline{\text{Price}}_{\text{base}}}$), the LP **automatically**:
       - Absorbs free midday solar surplus ($P_{c,k}$ up to $-\hat{L}_{\text{net}, k}$ when $\hat{L}_{\text{net}, k} < 0$) at zero grid import cost!
       - Caps discharge at $P_{d,k} \le \max(0, \hat{L}_{\text{net}, k})$ so zero stored energy is wasted into unremunerated export!
   - **Hard Safety Clamp for Difficult Mode (Zero Infractions):** Every infraction deducts $1.00$ full point. Enforce exact clamping before returning `action`:
     ```python
     cap_max = float(obs.get("cap_max_kwh", 6000.0))
     p_max = float(obs.get("power_max_kw", 1500.0))
     soc_kwh = float(obs["soc_kwh"])
     eta = 0.90 ** 0.5
     eps = 1e-4
     max_carrega = max(0.0, min(p_max, (cap_max - soc_kwh) / eta - eps))
     max_descarrega = max(0.0, min(p_max, soc_kwh * eta - eps))
     consigna_segura = float(max(-max_descarrega, min(max_carrega, consigna_desitjada)))
     return consigna_segura
     ```
   - **Latency & Offline Constraints:** Total `act(obs)` inference + LP solve must run in `< 10 ms` per step with zero internet/network calls.

## Format & Deliverables

Follow all repository submission rules and validate with `check_submission.py`. Deliver:
- `agent.py` (and any self-contained lightweight artifact file required by it)
- `train_pipeline.ipynb` (Kaggle/CUDA-ready training, feature engineering, Difficult Mode validation, and hyperparameter tuning notebook)
- `README.md` (clear execution guide for Kaggle training, local Difficult Mode simulation, and submission verification)
- `IMPLEMENTATION.md` (architecture details, Difficult Mode 3h-horizon + $12\times$ PV handling, missing-November cold-start priors, LP formulation with free-solar absorption & zero-export constraints, validation results)

## Multi-Agent Execution Workflow

Act strictly as **Orchestrator** (do not write large blocks of implementation code directly; delegate subtasks to smaller executor agents—Haiku for modular data/feature/doc tasks, Sonnet for multi-target forecasting, LP math, and debugging). Maintain a dedicated **Critical Auditor Agent** that reviews every stage and enforces a strict pass/fail checklist:
1. **Zero Data Leakage:** Verify that `load_kw`, `pv_kw`, `meteo`, and future prices/$\text{CO}_2$ beyond the allowed 3-hour window (`forecast_horizon=3`) are never accessed prior to step $t+1$.
2. **Missing-November & Seam Integrity:** Verify no lag features interpolate across the Oct $\rightarrow$ Dec 2025 gap in training, and confirm `agent.py` handles November 2025 test timestamps cleanly without `NaN` crashes.
3. **Zero Infractions in Difficult Mode:** Confirm `infraccions == 0` across full-train and 12-block validation simulations under `spec_hard` (`6,000 kWh` / `±1,500 kW` / `12× PV` / `forecast_horizon=3`).
4. **Submission Compliance:** Confirm `python check_submission.py` passes with zero errors, offline imports only, and `< 10 ms`/step runtime.
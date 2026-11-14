# Implementation notes — Difficult Mode agent

## 1. Problem recap

| | Difficult Mode |
|---|---|
| Battery | 6,000 kWh, ±1,500 kW, round-trip η = 0.90 (η_c = η_d = √0.90), SOC₀ = 50 %, SOC carried across the 12 weeks |
| Solar | inverter output ×12 (peaks ≈ 1,700 kW; net load `load − pv` is negative in ≈ 8 % of hours) |
| Market look-ahead | `preus_properes_24h` / `co2_properes_24h` contain only hours t, t+1, t+2 |
| Telemetry | last hour's `ultim_consum_kw`, `ultima_solar_kw` (×12) and weather station readings |
| Score | `0.7·ΔCO₂% + 0.3·ΔCost% − infractions`, exports are worth 0 € and 0 kg |

Consequences that drive the design:

* Everything beyond 3 h must be forecast: price and CO₂ for k = 3..23, load and PV for k = 0..23
  (even the current hour's load is unknown — the last measurement is hour t−1).
* Midday solar surplus is free, zero-carbon energy that is otherwise wasted.
* Net load is usually 600–1,300 kW, below the 1,500 kW inverter limit: discharging above the
  net load exports stored energy for nothing.

## 2. Architecture

```
obs ─► self.historial (absolute-hour buffers, pre-seeded with train history + November priors)
     ─► Features (same code offline and online)
     ─► forecasts, 24 h:  load, pv (LightGBM buckets + Ridge)   price, co2 (exact k<3, model k>=3)
     ─► planner over 24 steps with grid import G_k >= 0  (DP on a SOC grid, or scipy.linprog/HiGHS)
     ─► exact physical clamp ─► action (kW)
```

### 2.1 Online state buffer (`self.historial`)

Seven arrays indexed by absolute hour (hours since 2025-08-01): `load`, `pv` (×12), `irr`,
`tamb`, `tpan`, `price`, `co2`.

* **Pre-seeded with the train history.** Each hidden test week is the direct continuation of
  that month's train segment, so 24 h / 48 h / 168 h lags are available from the first step of
  every week. There is no cold start at week seams, except November.
* **Update rule.** Market values are written at t..t+2. Telemetry is written at the timestamp of
  the *previous environment row* (`prev_idx`), which is what `ultim_consum_kw` actually refers
  to. This is automatically correct at week seams (the stale value lands in the previous
  month, where it belongs) and across dropped rows — no lag window is ever corrupted.
  Market values that would fall beyond a block boundary (month end / start of the last 7 days)
  are used for planning but not stored, because the environment's next row is another date.
* **Missing weather** (the unmodified starter-kit environment does not provide it): irradiance
  is imputed from PV (r = 0.997), temperatures are carried forward.
* **Meter dropouts.** A `0.0` load reading is a sensor dropout: it is not stored, and the current
  hour's load is assumed to be 0 as well (the simulator's truth is also 0 during a dropout, so
  discharging would be pure export).

### 2.2 Missing November 2025 and late August 2026

Train has no meter / inverter / weather data for November 2025 (prices and CO₂ exist), and the
inverter stops on 2026-08-16.

* **Offline:** nothing is interpolated or forward-filled across long gaps. Only gaps ≤ 3 h are
  interpolated (the notebook compares this against a month / weekday / hour median profile and
  against unlimited interpolation). Training rows whose lag window (up to 168 h) crosses the
  Oct → Dec outage (t in 2025-11-01 .. 2025-12-07) are dropped for the load and PV targets.
* **Online:** `seed_gap_priors` fills November's empty telemetry hours with the hourly profile by
  day type (teaching day / weekend-holiday) of the October + 1–21 December academic term, and
  late-August PV with the August hourly profile. Real measurements overwrite the priors as they
  arrive, so after 24 h the short lags are real again.

### 2.3 Features (shared offline/online, `agent.Features`)

* Calendar (from the timestamp only): hour, weekday, sin/cos of hour-of-day, day-of-week and
  day-of-year; `is_weekend`, `is_bcn_holiday` (Catalonia + Barcelona 2025–2026, incl. Good
  Friday / Easter Monday / Pentecost Monday), `is_august_recess`, `is_exam_period`,
  `is_datathon_weekend`, plus `is_term_break` and a combined `is_offday`.
* Load: lags 1/2/3 h, target-hour lags 24/48/168 h, 24 h mean/std/min/max, 168 h mean,
  last temperature / irradiance / PV, and `l_adj = load[t−1] + load[h−24] − load[t−25]`.
* PV: lags 1/2/3 h, last irradiance and temperatures, target-hour lags 24/48/168 h, a clear-sky
  envelope `cs[h]` (97th percentile by day-of-year ±15 d and hour, interpolated through
  November), clearness of the last hour and of the last 24 h, and the persistence forecast
  `kt·cs[h]`. Predictions are clipped to `[0, 1.2·cs + 20]` and forced to 0 at night.
* Price / CO₂: the three exact values, lags −1/−3/−6/−12 h, target-hour lags 24/48/168 h,
  today-vs-yesterday differences, rolling stats over the last 24 known hours and 168 h mean,
  the other market series, irradiance, and the anchored naive forecast
  `y_adj = y[t+2] + y[h−24] − y[t+2−24]`, which ties k ≥ 3 smoothly onto the last exact value.
  `vent_ms` is identically 0 in train and is not used.

**Leakage rule:** telemetry is only indexed up to t−1 and market data up to t+2.
`train_lib.leakage_test` replaces all later values with noise and requires the feature
matrices to be bit-identical (result: 0.0).

### 2.4 Forecasters

One model family per target, 24-step output:

* **LightGBM per horizon bucket** — k = 0, 1, 2, 3–5, 6–11. A single stacked model with k as a
  feature was markedly worse at short range (it ignores the latest readings), e.g. PV MAE at
  k = 0: 74 kW stacked vs 40 kW per-bucket.
* **Ridge per horizon** (pure numpy at inference) for all k; it is the only model for k ≥ 12,
  where trees gave no measurable gain, and the fallback if LightGBM cannot be imported.
* **Blend (default):** mean of LightGBM and Ridge where both exist.
* **Quantile models at k = 0** (load 20th percentile, PV 80th percentile) give a prudent lower
  bound of the current net load, used as the discharge ceiling.

Validation MAE by horizon bucket (development run, models trained without the validation blocks):

| Target | k=0 | k=1 | k=2 | k=3–5 | k=6–11 | Same-hour-yesterday baseline |
|---|---|---|---|---|---|---|
| Load (kW) | 22.5 | 32.7 | 40.4 | 51.0 | 52.2 | 184 |
| PV ×12 (kW) | 40.4 | 53.9 | 61.4 | 73.7 | 82.2 | 76 |
| Price (€/kWh) | exact | exact | exact | 0.011 | 0.015 | 0.028 |
| CO₂ (kg/kWh) | exact | exact | exact | 0.007 | 0.013 | 0.016 |

PV beyond a few hours is weather-limited: no model beats day-ahead persistence by much. The
notebook reproduces this table for every family (seasonal-naive, Ridge, LightGBM stacked,
LightGBM buckets, XGBoost, CatBoost, RandomForest) **and** reports each family's Difficult Mode
score, since forecast error and control value are not the same thing.

### 2.5 Planner

Per step, over n = 24 hours (n shrinks at the known end of the simulation, 2026-08-31 23:00,
or when the environment's lists shrink, where the terminal value is set to 0):

```
min  Σ_k w_k · G_k  −  V_term · SOC_n  +  ε Σ_k (c_k + d_k)
s.t. G_k ≥ net_k + c_k − d_k,   G_k ≥ 0                 (grid import; exports earn nothing)
     SOC_{k+1} = SOC_k + η·c_k − d_k/η,   0 ≤ SOC ≤ cap
     0 ≤ c_k, d_k ≤ P_max,   d_0 ≤ max(0, net_lo)        (k = 0 discharge ceiling)
w_k = 0.7·co2_k / co2_ref + 0.3·price_k / price_ref
```

`co2_ref` and `price_ref` are the baseline-weighted mean CO₂ and price from train, so `w` is
the marginal effect of one imported kWh on the official score. Because only `G_k ≥ 0` is
penalised, the optimum by construction:

* **charges from solar surplus for free** — when `net_k < 0`, charging up to `−net_k` keeps `G_k = 0`;
* **never discharges above the net load** — extra discharge lowers SOC without lowering `G_k`.

Two interchangeable solvers (`params["solver"]`):

* `"lp"` — `scipy.optimize.linprog(method="highs")`, the formulation above (72 variables).
* `"dp"` (default) — the same problem by dynamic programming on an 81-point SOC grid, with the
  first-step action evaluated on exact candidates (grid transitions, `−net_0`, the limits) and
  linear interpolation of the cost-to-go. It reproduces the LP's score (20.04 vs 20.04 on
  validation) at about half the latency, and has no solver dependency. It is also the automatic
  fallback if `linprog` fails.

Findings from tuning against the official metric:

* **Terminal SOC value is inert** (`lam_term` from 0 to 3 gives identical scores): with ~2
  cycles per day the battery hits full or empty several times inside any 24 h window, which
  decouples the first action from the end of the horizon. The weekday / weekend / month
  structure is kept in the parameters but left at 1.0 — tuning it would be fitting noise.
* **Export margin:** a fixed kW buffer or the k = 0 quantile ceiling are worth about +0.2 points
  over no margin; the quantile ceiling (q = 0.2) with no extra buffer is the default.
* **Uncertainty discounting** of forecast weights for k ≥ 3 (`gamma` < 1) hurts; left at 1.
* Ablation with perfect forecasts on the same rolling planner: 21.27 (all four), 20.65 (load + PV
  only), 20.30 (price + CO₂ only). Knowing only the *current-hour* net load is worth +0.85,
  which is why the k = 0 models get their own buckets.

### 2.6 Safety clamp and robustness

```python
p_max -= 1e-6
max_carrega    = max(0, min(p_max, (cap_max − soc_kwh) / eta − 1e-4))
max_descarrega = max(0, min(p_max, soc_kwh · eta − 1e-4))
action = max(−max_descarrega, min(max_carrega, desired))
```

* `cap_max_kwh` / `power_max_kw` are read from `obs`; if absent (unmodified starter-kit
  environment) capacity is inferred as `round(soc_kwh / soc_pct)` and power as `cap / 4`.
  The rounding matters: without it float error gave `p_max = 1500.0000000000002` and 16
  infractions on the unmodified environment — found and fixed during verification.
  If the very first observation has SOC = 0 and no capacity keys, the capacity cannot be
  inferred and the Difficult Mode values (6,000 kWh / 1,500 kW) are assumed.
* Any exception inside `act` (missing artifact, timestamp out of range, solver failure) falls
  back to a rule-based action that is still clamped; `agent.n_fallback` counts those steps
  (0 in all simulations below).
* No network, no file access other than loading `model_bateria.pkl` next to `agent.py`;
  imports are `math`, `pathlib`, `numpy`, `pandas`, `joblib`, `lightgbm`, `scipy`.

## 3. Validation

**Simulator.** `hard_env.HardModeEnv` subclasses `starter_kit.environment.CampusBatteryEnv`
without touching physics or scoring; it truncates the market lists to `forecast_horizon = 3`,
adds `cap_max_kwh`, `power_max_kw` and the previous row's weather to the observation. The
environment frame follows `starter_kit/tutorial.ipynb` exactly (meter zeros kept, `dropna`),
with PV ×12.

**Validation set (mirror of the hidden test).** For each month the last 168 available train
hours — the week immediately before the hidden one (17–23 for most months; 15–21 Feb;
9–16 Aug because the inverter data ends) — plus a synthetic 17–23 November 2025 block with real
November prices / CO₂ and load / PV / weather averaged from two real weekday-aligned weeks
(13–19 Oct and 1–7 Dec). The 12 blocks are concatenated without resetting SOC. Models and the
agent's history exclude these blocks, and also the telemetry of the two source weeks of the
synthetic November block (`train_lib.NOV_SOURCES`), so that block is not scored on data the
models were trained on.

| Simulation, `spec_hard` | Score | ΔCO₂ % | ΔCost % | Infractions | Fallback steps |
|---|---|---|---|---|---|
| Agent, 12 validation blocks (honest) | **20.04** | 18.90 | 22.71 | 0 | 0 |
| Same, `solver="lp"` | 20.04 | 18.83 | 22.86 | 0 | 0 |
| Perfect-foresight LP over the whole series | 21.92 | 21.08 | 23.89 | 0 | – |
| Rule-based fallback only | 10.82 | 10.52 | 11.50 | 0 | – |
| Agent, synthetic November block alone | 16.76 | 16.45 | 17.48 | 0 | 0 |
| Agent, full train (5,993 h), final artifact — in-sample | 19.53 | 18.40 | 22.19 | 0 | 0 |
| Agent, full train, unmodified `CampusBatteryEnv` (24 h lists, no weather) — in-sample | 20.00 | 19.02 | 22.29 | 0 | 0 |

The agent captures 91 % of the perfect-foresight bound. Per-block validation scores:
Sep 17.4, Oct 17.6, Nov 12.9, Dec 10.2, Jan 12.8, Feb 18.5, Mar 21.7, Apr 27.8, May 26.4,
Jun 24.4, Jul 20.1, Aug 38.6 — winter weeks have little solar surplus and flatter CO₂ profiles.

Latency (laptop CPU, single thread): 3.5 ms mean / ~12 ms max per step with the DP planner,
6.8 ms mean with `linprog`; the 2,016-step evaluation takes ≈ 7 s of the 60 s limit.
`check_submission.py` passes for `submission.zip`.

Those latency figures are from a quiet machine. Under background load the same runs measured
8 to 9 ms mean per step (p99 about 30 ms, isolated spikes higher), i.e. up to about 20 s for a
2,016-step evaluation against the 60 s limit, and `linprog` 9 to 12 ms. The 10 ms target is met
on average but with little margin on a busy machine; `dp_grid` (81) and the LightGBM tree count
are the knobs if the server is slower.

An independent audit (separate agent, own scripts) confirmed: no feature changes when all
disallowed future data is replaced by noise (150 random steps) and identical actions at
runtime; no telemetry in November or in any hidden-week window of the shipped history; 0
infractions and 0 fallback steps on the validation and full-train frames under both
`HardModeEnv` and the unmodified `CampusBatteryEnv`; zip contents byte-identical to
`src/agent.py` and to the copies embedded in the notebook. Two of its findings were fixed: the
November source weeks were in the validation training data (see above), and capacity inference
failed at very low SOC.

## 4. Known limits and assumptions

* The hard-mode observation format (weather keys, `cap_max_kwh`, 3-value lists) is implemented
  from the challenge description; the repository's `environment.py` does not emit it. The agent
  accepts both formats.
* The November block is synthetic, so its score is indicative only; whether the hidden set even
  contains a November week with meter data is unknown.
* `sim_end = 2026-08-31 23:00` assumes the 12 weeks are evaluated chronologically. If the hidden
  set ends earlier, at most the energy left in the battery at that point is unused.
* Validation blocks are single weeks per month (2,014 h); differences below ~0.05 points between
  configurations are within noise, which is why the tuning cell only leaves a default when the
  gain exceeds 0.03.
* The notebook has not been executed end-to-end locally (it is meant to run on Kaggle or
  Colab); every function it calls, and its data-discovery cell, were exercised locally through
  scripts with the same data. The Colab upload dialog and Drive mount could not be tested here.
* The final artifact's clear-sky envelope uses all train PV (a 15-day climatology window around
  each date); the validation artifact uses only non-validation data.
* Outside Difficult Mode (e.g. the 2,000 kWh / 500 kW spec) the agent runs without infractions
  on the starter-kit environment, but it is not tuned for it.

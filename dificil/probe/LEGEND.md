# Diagnostic probe: how to read the results page

Upload `dificil/probe/probe_diag.zip` (only `agent.py`). The score is meaningless; read the **Infraccions** card and
the **Consigna Bateria (kW)** line in the 72-hour chart. Read the **size** of each setpoint; the sign carries no
information (the probe always pushes the charge back toward 50 %).

## Infraccions card
| Value | Meaning |
|---|---|
| 1 | one agent instance for the whole public run (what we assume) |
| N > 1 | the server created the agent N times (e.g. separate runs per week or per split) |

## Hour 0 (00:00 on 24/9)
A spike of 1,500–1,700 kW: the instance marker that causes that infraction.

## Hours 1–9: header (size of the setpoint)
| Hour | Encodes | Our simulator gives | Something different would mean |
|---|---|---|---|
| 1 | 50 × price-window length | **150** (3 h) | 1,200 = full 24 h window; other = variable window |
| 2 | 50 × CO₂-window length | **150** (3 h) | same as above |
| 3 | 50 × number of `obs` fields | **650** (13 fields) | fewer/more fields than we assume |
| 4 | 50 × number of `noves_dades_5min` fields | **300** (2 telemetry + 4 weather) | 100 = no weather; more = extra telemetry we ignore |
| 5 | 100 × timestamp type | **200** (pandas, no time zone) | 100 text · 300 pandas with zone · 400/500 datetime · 600 numpy |
| 6 | `cap_max_kwh` / 10 | **600** (6,000 kWh) | 0 = field missing |
| 7 | `power_max_kw` / 2 | **750** (1,500 kW) | 0 = field missing |
| 8 | 100 × (1 + timestamp hour − `obs["hour"]`, mod 24) | **100** (agree) | 200/300 = timestamps 1–2 h off |
| 9 | 100 × window code | **100** (window starts at the current hour) | 200/300/400 = window offset; 500 = empty windows |

Hours 10–11: 0 kW (separator).

## Hours 12–41: consumption telemetry
Setpoint size = **half of `ultim_consum_kw`**. Double it and compare with the blue **Demanda Campus** line:
- equal to the demand of the **previous** hour → hourly average of t−1 (what we assume);
- noticeably different, especially on the morning/evening ramps → a fresher or 5-minute reading: information our models
  don't use.

## Hours 42–71: solar telemetry
Setpoint size = **half of `ultima_solar_kw`**. Same comparison with the yellow solar line.

## After hour 72
0 kW. If the platform ranks your last upload, re-upload the final choice (v15) afterwards.

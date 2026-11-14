# Agent MPC-LP

Control predictiu: a cada hora resol un LP (scipy/HiGHS) sobre les 24 h de previsió
de preu i CO2 i aplica només la primera acció.

- Senyal per kWh importat: `w_t = 0.7*co2_t + 0.3*ratio*preu_t`, amb `ratio = CO2_base/Cost_base ≈ 0.587` (train).
- Valor terminal de l'energia: quantil del senyal × η_d; 0 quan l'horitzó toca el final de la simulació.
- Marge 1e-3 kWh als límits de SOC → 0 infraccions. No descarrega per sobre del consum net observat si < 500 kW.

| Train (5.993 h, simulador oficial) | Score | ΔCO2 | ΔCost | Infraccions |
|---|---|---|---|---|
| Plantilla | 0,00 | 0,88 % | 2,18 % | 46 |
| **MPC-LP** | **5,67** | 5,26 % | 6,62 % | 0 |
| Cota (LP previsió perfecta, tot l'any) | 5,68 | 5,27 % | 6,62 % | – |

Sensibilitat: `ratio` entre 0,45 i 0,75 → score 5,66–5,67. ~2 ms per pas.

`submission.zip` = `agent.py` + `parametres.json`, validat amb `check_submission.py`.

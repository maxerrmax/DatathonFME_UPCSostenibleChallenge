# Pipeline de comparació d'agents

Compara enfocaments de control de la bateria amb el **simulador oficial** (`starter_kit/environment.py`),
en una sola simulació contínua com al servidor, i contra la cota superior (LP amb previsió perfecta).

```bash
python estandard/run_all.py                 # tots els agents -> estandard/results/resultats.{csv,md}
python estandard/run_all.py mpc_lp rl_dqn   # només alguns
python estandard/tune.py mpc_lp_tuned       # cerca de paràmetres (evolució diferencial) sobre 'fit'
python estandard/agents/<nom>/train.py      # entrena els agents apresos
python estandard/package.py <nom>           # submissions/<nom>.zip + check_submission
```

## Dades i particions (`data.py`)
- Dataset horari **idèntic al tutorial oficial** (el que probablement usa el servidor).
- `validacio` = darrers 7 dies de cada bloc mensual de train (imita el test: última setmana de cada mes).
- `fit` = la resta. Tots els agents apresos/ajustats només veuen `fit`.
- `score_public` / `score_private`: suposem que el 50% públic és la primera meitat de cada setmana (no documentat).
- `score_web` aplica la penalització de la web (−1 per infracció; l'environment local resta 0,2).

## Afegir un agent
Crea `estandard/agents/<nom>/agent.py` amb `class BatteryAgent` (+ fitxers auxiliars). Ha de ser
autocontingut (és el que s'envia). Els scripts `train*.py` no s'empaqueten i poden usar qualsevol llibreria.

## Agents actuals

| Agent | Família | Idea |
|---|---|---|
| `plantilla` | regles | plantilla oficial (referència) |
| `rbc_rank` | regles | carrega a les k hores més netes/barates de les 24 h, descarrega a les k pitjors |
| `rbc_tuned` | cerca de polítiques | `rbc_rank` amb paràmetres optimitzats per evolució diferencial |
| `mpc_lp` | optimització | MPC amb LP (HiGHS) sobre les 24 h de previsió + valor terminal |
| `mpc_lp_tuned` | optimització + cerca | `mpc_lp` amb ràtio de pesos i valor terminal ajustats |
| `dp_soc` | programació dinàmica | DP sobre SOC discretitzat (81 nivells) |
| `imitacio_lgbm` | imitation learning | LightGBM imita les accions de l'oracle LP |
| `rl_fqi` | RL offline | Fitted Q-Iteration amb LightGBM, 5 accions discretes |
| `rl_dqn` | deep RL | DQN (stable-baselines3) exportat a numpy (servidor sense torch) |

## Resultats

Vegeu `results/resultats.md`. Resum (score oficial, validació / train complet; oracle 6,083 / 5,675):

| Agent | Validació | Train | Gap a l'oracle (val.) |
|---|---|---|---|
| mpc_lp | 6,08 | 5,67 | 0,003 |
| mpc_lp_tuned | 6,08 | 5,67 | 0,003 |
| dp_soc | 6,04 | 5,62 | 0,043 |
| imitacio_lgbm | 5,49 | 5,35 | 0,59 |
| rl_dqn | 5,02 | 4,61 | 1,06 |
| rl_fqi | 4,85 | 4,42 | 1,23 |
| rbc_tuned | 4,69 | 4,41 | 1,39 |
| rbc_rank | 3,84 | 3,73 | 2,24 |
| plantilla | 0,94 (0 amb −1/infracció) | 0,00 | 5,14 |

Conclusió: amb previsions exactes de preu i CO2, l'optimització (LP/DP) és pràcticament òptima; els
mètodes apresos perden entre un 10% i un 20% respecte a l'oracle, coherent amb la literatura
(CityLearn 2021–2023 el van guanyar enfocaments basats en optimització).

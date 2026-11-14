# Mode Difícil (6.000 kWh, ±1.500 kW, solar ×12, mercat a 3 h)

```bash
python dificil/train_forecast.py fit   # models sense validació -> agent_dificil/models_fit.pkl (+ MAE a validació)
python dificil/train_forecast.py all   # models amb tot train   -> agent_dificil/models_all.pkl (lliurament)
python dificil/heval.py '[{"k_dis":0.5}]'   # avalua a validació amb models_fit
cd dificil/agent_hard && zip ../submissions/hard_mpc_lgbm.zip agent.py feats.py parametres.json models_all.pkl
```

- `henv.py`: environment oficial amb l'observació del Mode Difícil (finestra 3 h, meteo, cap/power).
- `horacle.py`: òptim amb previsió perfecta (LP amb import/excedent).
- `agent_dificil/`: MPC-LP + previsions LightGBM (preu, CO2, consum, solar). Funciona també en Mode Estàndard.

| Validació (models entrenats sense validació) | Score |
|---|---|
| Òptim (tot conegut) | 22,34 |
| MPC + previsions LightGBM (`k_dis=0.5`) | **19,04** |
| MPC + previsió ingènua (lag 24 h) | 16,02 |

Pèrdua per variable (donant-la real a l'agent): mercat +1,4, solar +0,7, consum +0,5.

## Pipeline (dues etapes) — `python dificil/hpipe.py stage1 | stage2 <families>`

Etapa 1, famílies de previsió (validació, decisió per defecte; òptim 22,34):

| Família | Score | MAE consum | MAE solar | ms/pas |
|---|---|---|---|---|
| Previsió perfecta | 21,79 | 0 | 0 | 4 |
| LightGBM base | 19,04 | 62 | 76 | 6 |
| XGBoost | 19,03 | 77 | 78 | 11 |
| Ensemble (LGBM+XGB+HGB) | 19,03 | 67 | 76 | 91 (massa lent) |
| HistGradientBoosting | 18,95 | 70 | 77 | 41 |
| Ingènua (lag 24 h) | 16,50 | 186 | 97 | 3 |

Etapa 2, 60 configuracions de decisió per família (`results/etapa2_decisio.md`):
- error per hora del dia (`sigma_mode=hod`) > error global;
- prudència a la planificació futura `k_future` òptima a 0,3–0,5 (0,8 ja empitjora);
- prudència a la descàrrega `k_dis` òptima a 0,5–0,75; horitzó 16 vs 24 h: indiferent.

**Configuració final** (`parametres.json`): LightGBM base, `k_dis=0.75`, `sigma_mode=hod`, `k_future=0.3` → **19,19** a validació (abans 19,04).

## v3 (`hard_mpc_lgbm_v3.zip`) — validació 19,40

- **Calendari** (festius Catalunya/Barcelona, vacances UPC, exàmens, i si fa 24 h / 7 dies era festiu) als models
  de consum i solar. En dies festius apartats de l'entrenament, l'error de consum baixa un 37 % (175 → 110 kW).
  Al preu i al CO2 no ajuda (−4,8 % / 0 %) i no s'hi fa servir.
- **Detecció del final de la simulació**: si la finestra de mercat s'escurça, només es planifica fins al final
  i l'energia restant val 0. Tanca la diferència en Mode Estàndard (5,92 → 6,07, `mpc_lp` 6,08).
- Robustesa (validació): el mateix agent funciona en Estàndard, en mercat a 3 h, amb solar ×12 i en Difícil.

## v4 (`hard_mpc_lgbm_v4.zip`) — validació 19,42

Calendari de graus de la **FIB (Campus Nord)**: exàmens (30/10–5/11, 7/1–19/1, 7/4–13/4, 4/6–19/6) i
**períodes sense classes** (fora de 8/9–19/12 i 9/2–29/5). Afecta setmanes de test com 25–31/1 i 24–30/6.
MAE consum a validació 58,9 → 56,9 kW; en festius/períodes sense classes apartats, −35 % d'error.

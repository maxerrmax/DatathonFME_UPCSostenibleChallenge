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

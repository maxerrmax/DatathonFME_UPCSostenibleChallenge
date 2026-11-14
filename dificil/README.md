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

## v5 (`hard_mpc_lgbm_v5.zip`) — validació 19,42

Règims d'ocupació dels edificis segons els horaris publicats per la UPC (no n'hi ha un del Campus Nord;
s'usa el patró dels edificis UPC): **tancat** (25/12–6/1, 3–21/8) i **horari reduït de matí** (22–24/12,
22/6–31/7, 24/8–4/9). La setmana de test 25–31/8 queda com a horari reduït (no tancada).
Holdout de festius/vacances: MAE consum 109,9 (v4) → 109,1 kW; validació igual (19,42).

## v6 (`hard_mpc_lgbm_v6.zip`) — validació 19,80

Revisió (diagnòstic a validació, `models_fit`): 0 excepcions i 0 fallades del solver en 1.848 passos;
descàrrega perduda 2,2 %; excedent solar no aprofitat 13,5 %; l'error de demanda neta de l'hora actual
pujava a 200–250 kW entre les 10 i les 15 h (solar).

- **Cel clar**: taula 366×24 (quantil 95 % de la solar per dia de l'any i hora, finestra ±12 dies) desada dins
  del zip. Features noves: cel clar a tau, índex de claredat de l'última hora i la seva persistència.
  MAE solar a l'hora actual 148 → 76 kW. Validació 19,42 → 19,71.
- Sensibilitat: `pv_blend` 0,15 (més pes a la persistència de cel clar), `k_dis` 1,0, `k_future` 0,4;
  el valor terminal no influeix (19,74–19,75). Validació **19,80**.

## v7 / v8 (`hard_v7.zip`, `hard_v8.zip`) — validació 20,12 / 20,44

Resultats reals al rànquing públic: v2 18,17 · v6 **22,79** · agent d'un company (arquitectura equivalent) 22,95.

Diagnòstic nou (`dificil/exp.py`, puntuació per blocs mensuals; `dificil/lab.py`, laboratori de previsió):

| Si sabéssim exactament… | Guany a validació |
|---|---|
| el CO₂ de les 24 h | +1,2 (les 5 hores següents a les 3 conegudes: +0,6) |
| el preu de les 24 h | +0,2 |
| consum i solar de l'hora actual | +0,7 |

Canvis (tots validats amb models entrenats només amb `fit`):

- **Historial inicial**: el fitxer de models porta les sèries d'entrenament; l'agent comença amb l'historial ple
  (el dia abans de cada setmana de test és al train). Error del primer dia: consum 84 → 57 kW, preu −30 %.
- **Fora el lag setmanal** (168 h): empitjorava CO₂ i consum quan estava disponible.
- **CO₂** (`mkt_features2`): perfil de les últimes 24 h, pendent de les 3 hores conegudes, perfil d'ahir desplaçat;
  ensemble de 4 LightGBM sobre el **logaritme del quocient** respecte a l'últim valor conegut. MAE 0,0146 → 0,0128.
- **Preu**: mateixes features, residu additiu (ensemble de 4). MAE 0,0171 → 0,0158 (gairebé no mou el score).
- **Consum** (`load_features2`): perfil d'ahir desplaçat al nivell actual per a les hores properes, lag24 + residu
  per a les llunyanes. MAE de l'hora actual 49 → 38 kW.
- **Hora actual estocàstica**: el LP minimitza el cost esperat de la importació sobre 15 escenaris de l'error del
  consum net (quantils dels errors recents a la mateixa franja horària), en lloc del marge fix `k_dis`.
- Provat i descartat: biaixos `load_bias`/`pv_bias` (equivalents a la cautela que ja tenim), contrast del CO₂,
  preu en logaritme, valor terminal i horitzó (no influeixen).

| Versió | Validació (`fit`) | Públic |
|---|---|---|
| v6 | 19,80 | 22,79 |
| v7 (CO₂ nou + historial inicial) | 20,12 | pendent |
| v8 (+ CO₂ log, consum v2, hora actual estocàstica) | 20,44 | pendent |

Reproduir: `python dificil/train_v7.py fit` (validació) i `python dificil/train_v7.py all` (lliurament).

# Lliuraments del Mode Difícil

| Fitxer | Públic | Validació (`fit`) |
|---|---|---|
| `hard_v9_public_23.45.zip` | **23,45** (millor, 14/11 20:10) | 20,52 |
| `hard_v8_public_23.41.zip` | 23,41 (14/11 19:14) | 20,44 |
| v10 (no desat, PR #9) | 22,95 | 20,85 |

`hard_v8_public_23.41.zip` és exactament el zip enviat: `agent.py`, `feats.py`, `parametres.json` i `models_all7.pkl`.
Per pujar-lo de nou no cal tocar res. Per regenerar els models des del codi: `python dificil/train_v7.py all`
(el codi de `main` ja inclou el v9, que afegeix un model extra; l'agent del v8 l'ignora).

`hard_v9_public_23.45.zip` és el zip exacte enviat (idèntic a `submissions_hard_diag/hard_v9_hist0ext.zip`).
Models del v9 (`train_v7.py all`) amb l'**historial inicial complet per sèrie**: `hist0` s'omple amb totes les dades
disponibles de cada sèrie, no només les hores amb tots els sensors (preu/CO2 de novembre i de finals d'agost,
consum i meteo fins al 24/08). Així les setmanes de test 24-30/11 i 25-31/08 també comencen amb el dia anterior.
Regenerar: `python dificil/train_v7.py all` i després
`python scripts/extend_hist0_v8.py dificil/agent_hard/models_all7.pkl <sortida>/models_all7.pkl`.
El v10 (models per trams d'horitzó +3..+11 h) millorava la validació però va baixar al públic: no fer-lo servir.

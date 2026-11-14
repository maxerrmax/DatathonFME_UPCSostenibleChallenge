# Lliuraments del Mode Difícil

| Fitxer | Públic | Validació (`fit`) |
|---|---|---|
| `hard_v9_public_23.45.zip` | **23,45** (millor, 14/11 20:10) | 20,52 |
| `hard_v8_public_23.41.zip` | 23,41 (14/11 19:14) | 20,44 |
| v10 (no desat, PR #9) | 22,95 | 20,85 |

`hard_v8_public_23.41.zip` és exactament el zip enviat: `agent.py`, `feats.py`, `parametres.json` i `models_all7.pkl`.
Per pujar-lo de nou no cal tocar res. Per regenerar els models des del codi: `python dificil/train_v7.py all`
(el codi de `main` ja inclou el v9, que afegeix un model extra; l'agent del v8 l'ignora).

`hard_v9_public_23.45.zip` és el zip exacte enviat (models reentrenats per l'equip amb `train_v7.py all`).
El v10 (models per trams d'horitzó +3..+11 h) millorava la validació però va baixar al públic: no fer-lo servir.

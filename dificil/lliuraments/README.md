# Lliuraments del Mode Difícil

| Fitxer | Score públic | Observació |
|---|---:|---|
| `hard_v14_v12_final_agost.zip` | pendent | v12 + buidar la bateria al final de la setmana d'agost (final del test). Fora d'aquesta setmana, idèntic a la v12 |
| `hard_v13_v12equip_salt.zip` | **23,05** (pitjor; no fer-lo servir) | v12 + planificació a través del salt entre setmanes de test. Validació (`fit`): A 21,18 · B 19,19 (v12: 20,75 · 18,91) |
| **`hard_v12_public_23.53.zip`** | **23,53** | **Millor versió de JAMM (14/11/2026)**; calibratge online causal del CO₂ |
| `hard_v9_public_23.45.zip` | 23,45 | Predictor directe de demanda neta |
| `hard_v8_public_23.41.zip` | 23,41 | MPC i previsions LightGBM |
| v10 (no desat) | 22,95 | Millorava localment però va empitjorar al públic |
| `hard_v9_salt.zip` | no enviat | v9 + salt (abans `hard_v12a_salt.zip`). Validació A 20,90 · B 18,76 |
| `hard_v9_salt_trams.zip` | no enviat | v9 + salt + models per trams (abans `hard_v12b_salt_trams.zip`). Validació A 21,25 · B 19,13 |

Consulteu [`JAMM_v12_RESUM.md`](JAMM_v12_RESUM.md) per l'explicació de v12 i les comparacions en validació.

Els ZIP anteriors es conserven per traçabilitat. El directori `dificil/agent_hard/` conté la versió v12, amb `agent.py`, `feats.py`, `parametres.json` i `models_all7.pkl`.

v13 = mateix `models_all7.pkl` que la v9/v12 (historial complet); només canvien `agent.py` i `parametres.json`. Detalls a `docs/TRASPASO_MODE_DIFICIL.md` §8–9.

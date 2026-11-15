# Lliuraments del Mode Difícil

| Fitxer | Score públic | Observació |
|---|---:|---|
| `hard_v13_setmanes.zip` | pendent | v12 + calendari de setmanes de test (validació 20,75 → **21,12**) |
| **`hard_v12_public_23.53.zip`** | **23,53** | **Millor versió de JAMM (14/11/2026)**; calibratge online causal del CO₂ |
| `hard_v9_public_23.45.zip` | 23,45 | Predictor directe de demanda neta |
| `hard_v8_public_23.41.zip` | 23,41 | MPC i previsions LightGBM |
| v10 (no desat) | 22,95 | Millorava localment però va empitjorar al públic |

Consulteu [`JAMM_v12_RESUM.md`](JAMM_v12_RESUM.md) per l'explicació de v12 i les comparacions en validació.

Els ZIP anteriors es conserven per traçabilitat. El directori `dificil/agent_hard/` conté la versió v12, amb `agent.py`, `feats.py`, `parametres.json` i `models_all7.pkl`.

## v13 — salts entre setmanes de test (validació 21,12, v12 20,75)

El test encadena 12 setmanes no contigües: els **últims 7 dies de cada mes** (set-2025 → ago-2026), que són
exactament els forats del train (l'historial `hist0` no té cap hora d'aquestes setmanes). El v12 planificava les
hores posteriors al final de cada setmana com si el calendari continués. Amb informació perfecta, això sol ja
costava ~0,4 punts (MPC 24 h: 22,34 creuant bé els salts vs 21,96 sense saber què ve després).

- A les últimes 23 h de cada setmana, les hores de l'horitzó posteriors al final es preveuen com les primeres hores
  de la **setmana següent** (amb els mateixos models i l'historial de train del dia anterior a aquella setmana): +0,19.
- A l'última setmana, l'agent sap quan acaba la simulació i no es queda energia a la bateria: +0,18.
- Blindatge: si s'observa un salt de temps que no és l'inici d'una setmana coneguda, es desactiva (comportament v12).
  Amb un calendari que no coincideix (p. ex. a validació sense `_weeks`) dona exactament el v12 (20,75).
- Mateixos models i fitxers que el v12; només canvia `agent.py`. Validació: `python dificil/v13/eval_setmanes.py <carpeta_agent>`
  (passa a l'agent el calendari de setmanes de la validació).

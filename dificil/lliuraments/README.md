# Lliuraments del Mode Difícil

| Fitxer | Score públic | Observació |
|---|---:|---|
| `hard_v13_setmanes.zip` | 23,04 | v12 + calendari de setmanes de test: validació 21,12 però **baixa al públic**, no fer-lo servir |
| `hard_v14_analegs.zip` | pendent | v12 + dies anàlegs CO₂ (A 20,79 · B 17,42 · C 17,42) — **primer a enviar** |
| `hard_v15_setmanes_analegs.zip` | pendent | v13_rapid2 + anàlegs |
| `hard_v13_nomes_final.zip` | pendent | v12 + només buidar al final de la simulació (A 20,93) |
| `hard_v16_analegs_final.zip` | pendent | v14 + buidar al final (A 20,97) |
| `hard_v13_rapid2.zip` | pendent | v13 amb la previsió de l'inici de cada setmana precalculada (cap pas més lent que el v12 original) |
| `hard_v13_rapid.zip` | no enviat | v13 amb les optimitzacions de temps (mateixes accions que el v13) |
| `hard_v12_rapid.zip` | 23,53 | v12 amb les optimitzacions de temps (**mateixes accions que el v12**, ~20 % més ràpid) |
| **`hard_v12_public_23.53.zip`** | **23,53** | **Millor versió de JAMM (14/11/2026)**; calibratge online causal del CO₂ |
| `hard_v9_public_23.45.zip` | 23,45 | Predictor directe de demanda neta |
| `hard_v8_public_23.41.zip` | 23,41 | MPC i previsions LightGBM |
| v10 (no desat) | 22,95 | Millorava localment però va empitjorar al públic |

Consulteu [`JAMM_v12_RESUM.md`](JAMM_v12_RESUM.md) per l'explicació de v12 i les comparacions en validació.

Els ZIP anteriors es conserven per traçabilitat. El directori `dificil/agent_hard/` conté la versió v12, amb `agent.py`, `feats.py`, `parametres.json` i `models_all7.pkl`.

## v13 — salts entre setmanes de test (validació 21,12, v12 20,75) — PÚBLIC 23,04: DESCARTAT

`dificil/agent_hard/agent.py` torna a ser el v12. El codi del v13 és dins de `hard_v13_setmanes.zip`.

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

## Versions ràpides (hipòtesi: límit de 60 s al servidor)

Tres enviaments diferents de JAMM (22:23 → 23,03; 00:59 → 22,99; v13 01:02 → 23,04) cauen a ~23,0 amb CO₂ i cost
baixant alhora i 0 infraccions: pèrdua equivalent a 1–2 dies sense bateria, compatible amb un tall per temps.
Optimitzacions sense canviar cap decisió (accions idèntiques a validació, comprovat pas a pas):
- `feats.py`: memòria cau de `cs_value` i `day_flags`; `_nanmean` sense numpy per a llistes curtes.
- `agent.py`: la solar (`kind="dir"`) ja no calcula el model residual que després es descartava.
Temps a validació: v12 18,6 → 14,3 ms/pas; v13 19,1 → 16,6 ms/pas (mesurat amb 2 processos alhora).

**Resultat:** v12_rapid 23,53 (igual que el v12, com s'esperava). Les versions ràpides del v13 no s'han enviat:
la hipòtesi del temps continua sense confirmar. Ms/pas a validació (mitjana / p99 / màx):
v12 16,7 / 23,3 / 32,1 · v13 17,6 / 30,0 / 36,5 · v13_rapid 15,1 / 24,6 / 59,4 · v13_rapid2 14,5 / 20,3 / 27,7.

Informe complet de la nit: [`docs/INFORME_NIT_15-11.md`](../../docs/INFORME_NIT_15-11.md).

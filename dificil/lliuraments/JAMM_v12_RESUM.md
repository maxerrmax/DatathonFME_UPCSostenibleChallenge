# JAMM v12 — calibratge online del CO₂

**Repte UPC Sostenible · Datathon FME 2026 · Mode Difícil**

## Resultat públic (14/11/2026)

| Versió | Score | Reducció CO₂ | Estalvi cost | Infraccions |
|---|---:|---:|---:|---:|
| v9 | 23,45 | 22,25% | 26,25% | 0 |
| **v12** | **23,53** | **22,34%** | **26,31%** | **0** |

La v12 ha millorat la puntuació pública **+0,08 punts** respecte de la v9.

## Diferències tècniques respecte de v9

Manté els LightGBM de consum, solar, demanda neta, preu i CO₂ i el controlador MPC de 24 hores. La novetat és un **calibratge causal en línia del CO₂**: a cada hora, compara prediccions de CO₂ fetes hores abans amb els tres valors de mercat que ara coneix, estima amb la mediana el biaix recent (amb límit de magnitud) i corregeix les prediccions de CO₂ futures amb un efecte que decreix exponencialment amb l'horitzó. El MPC resol llavors l'LP amb les previsions corregides i aplica només la primera consigna.

No fa servir dades futures no observables ni necessita reentrenar models per fer el calibratge.

## Validació local amb models fit-only

| Agent | Holdout A (1.848 h) | Holdout B (1.512 h) |
|---|---:|---:|
| v9 | 20,48 | 16,96 |
| v12 | **20,75** | **17,30** |

Les millores locals no garanteixen el test privat; el resultat públic de 23,53 sí està observat. Zero infraccions als dos holdouts segons els experiments.

## Fitxers i conservació de versions

- `dificil/agent_hard/`: codi i models utilitzats per la v12.
- `dificil/lliuraments/hard_v12_public_23.53.zip`: ZIP de submissió v12 complet (4 fitxers).
- **No eliminar** `hard_v9_public_23.45.zip` ni `hard_v8_public_23.41.zip`.

**Nota:** `models_all7.pkl` s'inclou en el directori operatiu per reproduir exactament la v12; els noms `models_all7.pkl` i `parametres.json` són els que espera l'agent.

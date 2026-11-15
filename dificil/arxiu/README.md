# Arxiu: versions anteriors i variants descartades

Res d'aquí forma part del lliurament final (vegeu [`../../entrega_final/`](../../entrega_final/README.md)). Es conserva per traçabilitat:
mostra el que s'ha provat i amb quin resultat. Totes les rutes dels scripts ja apunten a aquesta carpeta.

| Carpeta | Què és | Resultat públic |
|---|---|---|
| `v13/` … `v20/` | Variants de l'agent de l'equip (codi, `feats.py`, `parametres.json` i models). Generades amb `scripts/build_v13.py` / `build_v14.py` i les eines de `../diag/`. | v15 23,53 · v17 23,29 · v18 23,13 · v13 23,07 · v16 22,97 · v14/v19/v20 sense puntuació anotada |
| `v19_dst/` | v19 + hora solar: `agent.py` de la v19 amb el `feats.py` de la B (sense models: són els de `entrega_final/solucio_B_v12_hora_solar.zip`). Públic **23,75**. Zip: `lliuraments/hard_v19_dst.zip` (sha256 `a97b8cd851044da74757cb8c1164e93929ec2dacc084f25ed36767fdcf1c9093`). Té el final del test (31/8/2026 23:00) i l'encadenament de setmanes codificats; no és una solució neta. | v19+DST 23,75 |
| `v19_dst_near/` | `v19_dst` + models de CO₂ especialitzats en h+3..+11 (`scripts/add_near_co2.py all <pkl de la B> <sortida>`) + no trepitjar `hist[t-1]` al canvi de setmana. Validació 20,80 (v19_dst 20,72), **públic 23,63 (−0,12)**. Descartat; zip no desat (sha256 `37eacce2fea7cf767a93757ec4daf30df03e74a9df07bc6cd550a1d6ed073e83`). | 23,63 |
| `lliuraments/` | Els zips de les variants anteriors (`hard_v13.zip` … `hard_v19.zip`, `hard_v19_dst.zip`). | |
| `submission_hard/` | Primer agent MPC + LightGBM del Mode Difícil (`agent_core.py` + `src/`). Es regenera amb `scripts/build_agent.py`; entrenament amb `scripts/train_forecast.py`. | 22,95 |
| `submissions_hard_diag/` | Zips de diagnòstic (v8/v9 amb historial complet, agent MPC antic, agent «quiet» de proves). | |

Les puntuacions de v8, v9 i v12 (les versions de referència) són a [`../lliuraments/`](../lliuraments/README.md).
Resum del que no va funcionar: [`../RECERCA_v13.md`](../RECERCA_v13.md) i [`../../docs/TRASPASO_MODE_DIFICIL.md`](../../docs/TRASPASO_MODE_DIFICIL.md).

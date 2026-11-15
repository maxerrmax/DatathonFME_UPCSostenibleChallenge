# Lliuraments del Mode Difícil

> **Els lliuraments finals són a [`../../entrega_final/`](../../entrega_final/README.md).** Aquesta carpeta guarda la resta de zips
> de referència (v8, v9, v12). Les variants descartades són a [`../arxiu/`](../arxiu/README.md).

Puntuació al rànquing públic (50 % de cada setmana de test; soroll entre llavors ≈ ±0,2):

| Fitxer | Públic | Observació |
|---|---:|---|
| `hard_v12B_natcal.zip` | pendent | Solució B + calendari nacional i període tarifari 2.0TD a preu i CO₂ (val 20,75; setmanes de festius 19,10 vs 18,99). Candidat |
| `../../entrega_final/solucio_B_v12_hora_solar.zip` | **23,63** | v12 + taula de cel clar en hora estàndard (canvi d'hora). **Final B** |
| `hard_v12_public_23.53.zip` | **23,53** | v12: calibratge causal en línia del CO₂. **Final A** |
| `../arxiu/lliuraments/hard_v15.zip` | 23,53 | v12 + final del test conegut; el guany cau a la meitat privada. No seleccionat |
| `hard_v9_public_23.45.zip` | 23,45 | Predictor directe del consum net de l'hora actual |
| `hard_v8_public_23.41.zip` | 23,41 | MPC + previsions LightGBM amb historial inicial |
| `../arxiu/lliuraments/hard_v17.zip` | 23,29 | |
| `../arxiu/lliuraments/hard_v18.zip` | 23,13 | Hipòtesi de reinici de SOC per setmana (falsa) |
| `../arxiu/lliuraments/hard_v13.zip` | 23,07 | |
| `../arxiu/lliuraments/hard_v16.zip` | 22,97 | Amb xarxa neuronal |
| v10 (no desat) | 22,95 | Millorava la validació (20,85) i va baixar al públic |
| `../arxiu/lliuraments/hard_v14.zip`, `hard_v19.zip` | — | Sense puntuació pública anotada aquí (vegeu `../../CLAUDE.md`); no seleccionats |
| `../probe/probe_diag.zip` | — | Sonda de format del servidor (no és un agent) |

Explicació de la v12: [`JAMM_v12_RESUM.md`](JAMM_v12_RESUM.md).

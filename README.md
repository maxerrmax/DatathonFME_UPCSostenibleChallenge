# UPC Sostenible — Control de la bateria del campus · Datathon FME-UPC 2026

Equip **JAMM** (FIB-UPC): Jordi Xu Wang, Albert Contreras, Max Villalba i Marcel Tiburcio.

Reptes de [UPC Sostenible](https://repteupc.datathon.cat): programar una política de control (`BatteryAgent`) que decideix cada hora la
càrrega i descàrrega d'una bateria connectada al Campus Nord, minimitzant les emissions de CO₂ (70 % de la nota) i la factura
elèctrica (30 %) respecte al campus sense bateria. La simulació és seqüencial: l'agent només veu l'estat actual i el passat.

```
Score = max(0, 0,70·ΔCO₂(%) + 0,30·Δcost(%) − 1,00·Infraccions)
```

## Resultats

| Mode | Bateria | Què s'avalua | El nostre resultat |
|---|---|---|---|
| **Estàndard** | 2.000 kWh, ±500 kW, mercat a 24 h | Control amb previsions de mercat | **6,43** (òptim global: MPC-LP) |
| **Difícil** | 6.000 kWh, ±1.500 kW, solar ×12, mercat a 3 h | Previsió de preu, CO₂, consum i solar | **23,63** al públic (solució B) · 23,53 (solució A) |

**Lliurament final del Mode Difícil → [`entrega_final/`](entrega_final/README.md)**: dues solucions (A: v12 original; B: v12 + hora solar)
amb els zips, el codi legible, els hashes i les instruccions per verificar-les i regenerar-les.

## Mapa del repositori

| Carpeta | Contingut |
|---|---|
| [`entrega_final/`](entrega_final/) | **Les 2 solucions finals del Mode Difícil**, codi, SHA-256 i guia de verificació. Començar aquí. |
| [`dificil/`](dificil/README.md) | Mode Difícil: agent MPC + LightGBM (`agent_hard/`), entrenament (`train_v7.py`), validació, historial de versions, recerca (`RECERCA_v13.md`), experiments (`arena/`, `diag/`) i lliuraments antics (`lliuraments/`). |
| [`estandard/`](estandard/README.md) | Mode Estàndard: comparació de 10 agents (regles, MPC-LP, DP, imitació, RL) amb el simulador oficial i el seu lliurament. |
| [`docs/`](docs/) | Enunciat oficial ([`ENUNCIAT.md`](docs/ENUNCIAT.md), PDF/LaTeX), [guia de variables](docs/GUIA_VARIABLES.md), [traspàs del Mode Difícil](docs/TRASPASO_MODE_DIFICIL.md) i [informe de recerca](docs/recerca/estrategies_control_bateries.md). |
| [`scripts/`](scripts/) | Utilitats: `package_final.py` (regenera la solució B), `eval_agent_dir.py` (validació d'un agent tal com es lliura), `extend_hist0_v8.py`, `oracle_hard.py`, etc. |
| [`starter_kit/`](starter_kit/README.md) | Kit oficial: entorn de simulació, `check_submission.py` i plantilla d'agent. |
| [`data/train/`](data/) | Dades d'entrenament oficials (6 CSV: comptador, inversors, meteo, entrades/sortides, PVPC, CO₂). El test és ocult. |
| [`dashboard/`](dashboard/README.md) | Dashboard de decisió i explorador de dades (Streamlit), repte secundari. |
| [`notebooks/`](notebooks/) · [`src/`](src/) | Anàlisi exploratòria i codi base compartit (dades, previsions, MPC-LP) de les primeres versions dels dos modes. |
| `submission_hard/`, `submissions_hard_diag/` | Arxiu: primer agent MPC del Mode Difícil (22,95) i zips de diagnòstic. |

## Com reproduir el lliurament final

```bash
pip install -r requirements.txt

# 1. Verificar els zips lliurats
python starter_kit/check_submission.py entrega_final/solucio_A_v12_original.zip
python starter_kit/check_submission.py entrega_final/solucio_B_v12_hora_solar.zip
(cd entrega_final && sha256sum -c SHA256SUMS.txt)

# 2. Regenerar la solució B des del codi (~2 min, determinista, idèntica byte a byte)
python scripts/package_final.py
```

Detalls, validació de referència i limitacions: [`entrega_final/README.md`](entrega_final/README.md).

## Resum tècnic del Mode Difícil

MPC de 24 h amb LP (`scipy`/HiGHS) sobre previsions de LightGBM entrenades només amb `data/train/`: preu i CO₂ a +3..+23 h, consum,
solar (amb taula de cel clar) i consum net de l'hora actual; calibratge causal en línia del CO₂ i hora actual estocàstica.
Només `numpy`, `scipy` i `lightgbm` a l'inferència (≈ 16 ms per pas; límit del servidor: 60 s per simulació completa).

El que **no** va funcionar (xarxes neuronals, RL, CBR, programació estocàstica de dos nivells, regles d'expert, funcions de valor
apreses, calibratge en línia del consum/preu, etc.), amb xifres, és a [`dificil/RECERCA_v13.md`](dificil/RECERCA_v13.md) i a
[`docs/TRASPASO_MODE_DIFICIL.md`](docs/TRASPASO_MODE_DIFICIL.md). El valor de la informació perfecta de cada variable i el sostre
teòric (22,34 a validació amb previsió perfecta) també hi són documentats.

## Dades externes

Els dos lliuraments finals **no fan servir dades externes**: només `data/train/`. El repositori conté scripts per baixar historial públic de
Red Eléctrica (`scripts/download_ree_history.py`, `scripts/build_ree_hourly.py`) i l'opció `ree` de `dificil/train_v7.py` com a
experiment; no s'han fet servir en cap lliurament i les dades baixades (`data/external/`) no es versionen.

## Notes per a l'equip

- Treballem amb **4 nuclis**: mai dos processos pesats alhora. Vegeu [`CLAUDE.md`](CLAUDE.md).
- `git pull` abans de cada `git push`.
- Els models (`*.pkl`) no es versionen: es regeneren amb `dificil/train_v7.py`.
- Les claus d'API no es versionen (`.streamlit/secrets.toml`, `opencode.json` són a `.gitignore`).

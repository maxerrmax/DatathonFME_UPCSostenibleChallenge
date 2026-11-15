# Dashboard de decisió · Bateria del Campus Nord

Tot el dashboard respon una pregunta: **quin agent de bateria lliurem, i per què?**
Compara totes les versions del Mode Difícil, mostra què ha confirmat el rànquing públic
i què ha desmentit, i assenyala on són els punts que queden.

## Com s'engega

Des de l'arrel del repositori:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Pàgines

| Pàgina | Decisió que ajuda a prendre | Evidència principal |
|---|---|---|
| **Veredicte** | Quina versió va com a lliurament final | Puntuació pública de cada lliurament, validació contra públic, 34 aspirants contra la v12 en 29 setmanes aparellades, condicions (sí/no) per a cada aspirant |
| **Cara a cara** | Què fan diferent dos agents | Marcador, puntuació per setmana, punts perduts respecte a l'òptim per hora del dia, explorador d'una setmana |
| **On són els punts** | En què treballar i què deixar de provar | Guany de cada canvi, valor de la informació perfecta per entrada, idees descartades |
| **Campus i xarxa** | Quan carregar, com de gran ha de ser la bateria, fins on refiar-se de les dades | Cost horari d'un kWh en punts, corba de dimensionament, mapa de cobertura del comptador |
| **Simulador** | Comprovar qualsevol `agent.py` del Mode Estàndard | Simulador oficial contra un programa lineal amb previsió perfecta |

## D'on surten les xifres

- `submissions.json` es manté **a mà**: una fila per lliurament amb la puntuació pública, la foto del
  rànquing i els experiments descartats. Cal editar-lo després de cada lliurament.
- `model_data/` el genera `python dashboard/build_models.py` (uns 5 minuts, un sol procés):
  - `arena.json`: resultats de `dificil/arena/results/store*.json` i l'òptim per plec i per setmana
  - `traces.parquet`, `runs.json`: execucions hora a hora de la v9, la v12 i els nuclis de la v13 i la v14 (plec 1)
  - `sizing.json`: puntuació amb previsió perfecta per a sis mides de bateria, amb la solar real i la ×12
- Les pàgines del campus llegeixen `data/train/` a través de `data.py` (memòria cau a `data/processed/hourly.parquet`).

Dues comprovacions mantenen les pàgines coherents entre si: les execucions hora a hora reprodueixen
exactament les puntuacions de l'arena (v9 20,52; v12 20,74), i els punts perduts per hora del dia sumen
la distància de puntuació fins a l'òptim (22,34).

## Disseny

Pàgina blanca i tinta negra, com una secció de dades d'un diari: sense targetes, ombres ni degradats.
Una sola tipografia (Libre Franklin). El color té significat fix: blau UPC = l'agent que lliurem,
vermell = l'aspirant o l'avís, negre = l'òptim, gris = context. Cada títol de gràfic diu la conclusió.

## Fitxers

- `app.py` estructura de la pàgina · `ui.py` colors, CSS, taules i estil dels gràfics
- `p_verdict.py`, `p_models.py`, `p_gap.py`, `p_campus.py`, `p_sim.py` un fitxer per pàgina
- `models.py` càrrega de la comparació de models · `build_models.py` precàlcul
- `data.py` taula horària a partir dels CSV · `sim.py` simulador i òptim del Mode Estàndard

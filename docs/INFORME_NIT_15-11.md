# Informe de la noche (15/11, 01:45–02:45 UTC) — Mode Difícil

Base: v12 (23,53 en el público). Validación con **tres holdouts** (A = validación del equipo, últimos 7 días
de cada bloque; B = los 7 días anteriores; C = días −21..−15), con modelos entrenados sin cada holdout.
v12: A 20,75 · B 17,29 · C 17,46. Scripts: `dificil/v14/` (`splits.py`, `train_any.py`, `ev.py`, `smoke2.py`).

## Candidatos (todos en `dificil/lliuraments/`, pasan `check_submission.py`, 0 infracciones)

Todos llevan las optimizaciones de tiempo (mismas decisiones, ~15 % menos por paso) y son **más rápidos que el v12
original en cada paso** (p99 ~19 ms frente a 23 ms; peor paso ~27–36 ms frente a 35 ms).

| Zip | Qué es | Validación | Riesgo |
|---|---|---|---|
| `hard_v14_analegs.zip` | v12 + días análogos para el CO₂ (mezcla 40 %) | A 20,79 · B 17,42 · C 17,42 (media +0,04) | bajo |
| `hard_v13_rapid2.zip` | v13 (calendario de semanas), sin pasos lentos | A 21,12 (con calendario) | alto: el v13 dio 23,04 |
| `hard_v15_setmanes_analegs.zip` | v13_rapid2 + análogos | A 21,15 (con calendario) | alto |
| `hard_v13_nomes_final.zip` | v12 + solo vaciar la batería en las últimas horas de la simulación (31/08) | A 20,93 (+0,18) | medio |
| `hard_v16_analegs_final.zip` | v14 + vaciar al final | A 20,97 · B 17,42 · C 17,42 | medio |

**Orden recomendado de envío**
1. `hard_v14_analegs.zip` (el más seguro).
2. `hard_v13_rapid2.zip`: decide igual que el v13 (23,04) pero sin ningún paso más lento que el v12. Si supera
   23,53, el v13 falló por tiempo → enviar `hard_v15_setmanes_analegs.zip`.
3. Si (2) vuelve a dar ~23,0: probar `hard_v13_nomes_final.zip`; si supera al v12, enviar `hard_v16_analegs_final.zip`.

## Qué funcionó (poco)
- **Días análogos para el CO₂**: buscar en el historial las 8 horas con la forma de CO₂ de las últimas 27 h más
  parecida (misma hora del día) y usar su continuación. Solo: MAE h3–7 0,0087 vs LightGBM 0,0094; mezcla 40 %:
  −13 % de error en h3–7. Puntuación: +0,04 / +0,13 / −0,04 en A/B/C. Estable con pesos 0,2–0,5.

## Qué NO funcionó (no repetir)
| Idea | Resultado |
|---|---|
| Parámetros de decisión (`net0_mix`, `scen`, `scen_win`, `k_future`, `load_tau`, `pv_blend`, `cs_tau`, `q_term`, `term_scale`) | todos ±0,03 |
| Peso CO₂/precio `ratio` 0,5–1,1 | óptimo en el valor teórico (0,62); más peso al coste empeora |
| Corrección online del precio (como la del CO₂ del v12) | 20,68–20,75 |
| Ajustar la corrección del CO₂ del v12 (decaimiento, fuerza) | 20,65–20,72 |
| Análogos con restricción de época del año / tipo de día / forma del precio | igual o peor |
| Análogos como variable del LightGBM (stacking) | 38,12 (A+B) vs 38,21 de la mezcla simple |
| LP estocástico de dos etapas con escenarios de CO₂ de los análogos | +0,00–0,03 en A y más lento |
| Hiperparámetros del CO₂ (hojas 31/127, huber, L1, colsample 0,6) | 38,13–38,23 vs 38,21 |
| Escenarios de la hora actual condicionados a la variabilidad del cielo | ±0,02 |
| **Consumo nocturno exacto**: de 0 h a 5 h el consumo es idéntico al de las 23 h (96–100 % de las noches, también en el test) | incluso con consumo nocturno perfecto: 20,74. De noche la batería no se juega nada |
| Corrección AR del error de la hora actual | correlación de día −0,07: nada que corregir |

## Diagnóstico útil
- Con previsiones perfectas el agente llega a 21,86 en A; el resto hasta el óptimo (22,34) son los saltos entre
  semanas (la siguiente semana es otro mes).
- La puntuación sube casi en proporción al error de CO₂ eliminado: −25 % → +0,15; −50 % → +0,30; −100 % → +0,48.
- Hora actual perfecta: +0,45 (consumo de día +0,15, solar +0,12; el error es sobre todo solar a mediodía, 100–140 kW).
- Flujos frente al óptimo: la solar sobrante se captura casi entera (66/67,6 MWh) y la energía movida es casi la
  misma; solo ~20 % de la pérdida es descarga desperdiciada. El resto es elegir las horas (previsión del CO₂).

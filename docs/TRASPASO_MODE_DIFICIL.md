# Traspaso — Mode Difícil (estado a 14/11/2026, ~19:30)

Resumen para continuar en una sesión nueva sin repetir pruebas. Las instrucciones generales están en `CLAUDE.md`.

## 1. Estado actual

| | |
|---|---|
| **Mejor entrega (base)** | v9 con historial inicial completo: **23,45** en el ranking público (7.º) |
| Zip exacto | `dificil/lliuraments/hard_v9_public_23.45.zip` (= `submissions_hard_diag/hard_v9_hist0ext.zip`) |
| Validación de referencia | **20,52** (modelos `fit`, validación del equipo; óptimo con previsión perfecta 22,34) |
| Tiempo | ~16–18 ms/paso en local → ~35 s para las 2.016 h (**límite del servidor: 60 s**) |
| Líder | SanHackobos_v7 **24,10** (CO₂ 22,86 %, coste 26,99 %); máximo anunciado ~25 |

Ranking público a las 19:23 (top): SanHackobos_v7 24,10 · AAA 23,71 · Sanhackobos_v6 23,70 · Ommetje 23,57 ·
SSA Gourmet 23,50 · Riviera Maya Boys 23,45 · **JAMM 23,45** (CO₂ 22,25 %, coste 26,25 %).

Totales de referencia del test público deducidos del ranking: CO₂ base ≈ 60.250 kg, coste base ≈ 94.975 €
(cociente 0,634 kg/€; el agente usa `ratio` = 0,6213 → coherente).

## 2. Cómo validar en igualdad de condiciones

```bash
python dificil/train_v7.py fit                       # ~50 s -> dificil/agent_hard/models_fit7.pkl (sin la validación)
mkdir -p /tmp/v9fit && cp dificil/agent_hard/{agent.py,feats.py,parametres.json} /tmp/v9fit/
cp dificil/agent_hard/models_fit7.pkl /tmp/v9fit/models_all7.pkl
python scripts/eval_agent_dir.py /tmp/v9fit          # debe dar 20,52
python scripts/eval_agent_dir.py /tmp/v9fit '{"ratio":0.53}' '{"k_dis":0.8}'   # varios params en paralelo
```

Entrega: `python dificil/train_v7.py all` y luego
`python scripts/extend_hist0_v8.py dificil/agent_hard/models_all7.pkl <dir>/models_all7.pkl` (historial completo),
copiar `agent.py`, `feats.py`, `parametres.json` y zip con los 4 ficheros en la raíz. Validar con
`python starter_kit/check_submission.py <zip>`.

**Ojo:** mejoras pequeñas del MAE muchas veces NO mejoran la puntuación. Decidir siempre por puntuación de validación
y, si es posible, confirmar en el público (la v10 subía en validación, 20,85, y bajó en el público, 22,95).

## 3. Lo que funcionó (y está en la v9)

| Cambio | Efecto |
|---|---|
| Historial inicial (datos de train en el .pkl; el día antes de cada semana de test está en train) | consumo 1.er día 84→57 kW, precio −30 %; público 22,79→23,41 junto con otros cambios del v8 |
| Historial completo por serie (precio/CO₂ de noviembre y fin de agosto, consumo hasta 24/8) | semanas 24–30/11 y 25–31/8 arrancan con historial; público 23,41→23,45 junto con la v9 |
| Modelo directo del consumo neto de la hora actual (v9, `net0`, mezcla 50 %) | validación 20,44→20,52 |
| CO₂ con ensemble de log-cociente, consumo v2, hora actual estocástica (v8) | validación 20,12→20,44 |
| Cielo despejado / claridad para la solar (v6) | validación 19,42→19,71 |

## 4. Lo probado SIN éxito (no repetir sin una idea nueva)

Sobre la v9 (validación del equipo, ref. 20,52):
- **Peso CO₂/precio (`ratio`)**: 0,45 → 20,47 · 0,53 → 20,51 · 0,62 → 20,52 · 0,70 → 20,46. Solo desplaza CO₂↔coste.
- **Variables de dinámica solar para el CO₂** (cambio de cielo despejado entre t+2 y τ, claridad última hora, claridad
  de ayer a esa hora, solar esperada): MAE CO₂ −1,1 % (h+3..+7 −2,5 %) pero puntuación 20,49. Revertido.
- **Ensemble de semillas de CO₂**: 2 semillas 20,54 (+0,02, ruido) a 21 ms/paso; 3 semillas 20,52 a 25 ms/paso
  (riesgo con el límite de 60 s). No adoptado.

Sobre el agente MPC antiguo (`submission_hard/`, 22,95 en el público), conclusiones que también aplican:
- **Horizonte**: 8 h 16,16 · 12 h 16,94 · 16–24 h 17,00 · 30/36/48 h (modelos hasta 48 h) 16,90. 24 h es suficiente.
- **Valor terminal**: no cambia la primera decisión (el LP llena la batería en las horas más baratas, normalmente más
  tarde). No es un bug.
- **Calendario académico** (clases/exámenes/reevaluaciones/puentes): el perfil horario es igual con o sin docencia;
  ±0,05. El consumo depende de la hora, de si el campus está abierto y del clima.
- **Regímenes de apertura cerrado/reducido** (Navidad 25/12–6/1, agosto 1–23, reapertura 24/8/2026): mejoraba los días
  especiales simulados (7,36→9,21) pero **bajó en el público** → revertido.
- **Objetivo residual para precio/CO₂** (diferencia respecto al último valor): baja el MAE pero aplana la forma del día
  y baja la puntuación (16,88→16,65). **Amplificar el contraste** de las horas previstas: tampoco.
- **Cuantiles** para consumo/solar en lugar de sesgo fijo: igual. **Pesimismo creciente con el horizonte**: igual.
- **Mezcla con las previsiones de otro agente**: +0,07 como mucho.
- **Precio de exportación**: el agente lo trata como 0 € (el profesor confirmó que no se paga).

## 5. Diagnóstico: dónde queda margen

- Sobre la v8 (`dificil/README.md`): saber el CO₂ exacto de 24 h valdría **+1,2** (las 5 h siguientes a las 3
  conocidas, +0,6); el precio exacto +0,2; consumo y solar de la hora actual +0,7.
- Ranking: los de arriba ganan sobre todo en **CO₂**. SanHackobos subió de 23,70 a 24,10 en 40 min mejorando CO₂ y
  coste **a la vez** (+0,61 y +0,74 puntos porcentuales sobre nosotros), patrón de mejor información general, no de
  ajuste de un parámetro.
- Posibilidad (no verificable, **no seguir**): incluir en el zip los datos públicos reales de REE del periodo de test
  explicaría un salto así, pero es usar información del futuro (prohibido; el jurado audita el código).

## 6. Ideas pendientes (no probadas)

- Mejorar el modelo de la hora actual (+0,7 de margen teórico): p. ej. usar más señales de corto plazo (meteo de la
  última hora, tendencia de las últimas 3 h) en `net0`, o ajustar la mezcla `net0_mix`.
- Escenarios de la hora actual: probar `scen` (ahora 15) y `scen_win` (ahora 2), `k_future`.
- CO₂ a corto plazo con otro enfoque (p. ej. modelo de la forma completa del día), con prudencia por lo visto en v10.

## 7. Avisos prácticos

- **4 núcleos**: nunca dos procesos pesados a la vez (un entrenamiento de 20 s llegó a tardar 16 min).
  `scripts/eval_agent_dir.py` y `scripts/tune_hard.py` ya usan 1 hilo por proceso.
- `git pull` antes de cada `push`. Las ramas del equipo: `main`, `dificil-v7`, etc.
- Los pickles (`*.pkl`) de `dificil/agent_hard/` no se versionan; se regeneran con `train_v7.py`.

## 8. Salto entre semanas de test sobre la v9 — pendiente de confirmar en el público

Las 12 semanas de test se simulan seguidas: tras la última hora de un mes viene la primera de la semana de test del mes
siguiente y **el SOC se conserva**. La v9 planificaba como si después viniera el día 1 del mes siguiente. Cambios en
`dificil/agent_hard/agent.py` (todo desactivable con `{"jump": false, "jump_meas": false}` → v9 exacta, 20,52):

- **Planificar a través del salto**: en las últimas 24 h de cada semana, las horas posteriores al salto se rellenan con
  los valores del día anterior a la semana siguiente (está en train → `hist0`) y con los valores reales de la ventana de
  mercado cuando ya cruza el salto. Calendario: semanas = últimos 7 días de cada mes (`seg_off` = 0).
- **Final de la simulación** conocido por calendario (tras la última semana no hay datos de train): se vacía la batería
  con 24 h de margen en lugar de reaccionar en las 2 últimas horas.
- **Historial tras el salto**: la "última hora" que da el servidor en la primera hora de cada semana es de un mes antes;
  ya no pisa el valor real de train.
- Salvaguarda: si el salto observado no coincide con el calendario, se desactiva solo (vuelve al comportamiento v9).

| Variante (modelos `fit`) | Val. A (días 17–23) | Val. B (días 10–16) | ms/paso |
|---|---|---|---|
| v9 | 20,52 | 18,45 | 17 |
| v9 + salto (`hard_v9_salt.zip`) | 20,90 | 18,76 | 17 |
| v9 + modelos por tramos de la v10 | 20,85 | 18,77 | 15 |
| v9 + salto + tramos (`hard_v9_salt_trams.zip`) | **21,25** | **19,13** | 15 |

Validar: `python scripts/eval_agent_dir.py <dir> --segs` (`--segs` le pasa al agente los tramos reales de la validación;
sin él, el agente usa el calendario de test). Segunda partición: `VAL_SHIFT=7` al entrenar (`train_v7.py fit`) y evaluar.
Modelos por tramos: `python dificil/train_v7.py fit|all --bands` → `models_*7b.pkl` (el agente los usa si están).

Zips (historial completo, validados con `check_submission.py`): `dificil/lliuraments/hard_v9_salt.zip` (mismo `.pkl`
que el 23,45, solo cambia el agente) y `hard_v9_salt_trams.zip` (además, modelos por tramos). Superados por la v13 (§9).

Sobre la v10: su mejora se reproduce en las DOS particiones (+0,33 / +0,32), así que el 22,95 del público no cuadra con
un sobreajuste; es exactamente la nota del agente MPC antiguo → revisar si se subió el zip correcto.

Más cosas probadas hoy sin efecto (ref. 20,52): `net0_mix` 0/0,25/0,75/1 → 20,45–20,51 · `scen` 7/31 → 20,50/20,52 ·
`scen_win` 0/1/4/12 → 20,46/20,53/20,49/20,40 · `k_future` 0/0,2/0,6/0,8 → 20,49–20,53 · `scen_scale` 0,8/1,25 →
20,52/20,49 · tramos más finos (+3, +4, +5, 6–8, 9–11) → igual que los de la v10 (A 21,23 / B 19,20).
Con información perfecta (sobre v9): consumo neto de la hora actual +0,47; CO₂ de +3..+7 h +0,63; precio 24 h +0,19.

## 9. v13 = v12 del equipo (calibración online del CO₂, 23,53) + salto entre semanas — **23,05 en el público: descartada**

Zip: `dificil/lliuraments/hard_v13_v12equip_salt.zip` (mismo `.pkl` que v9/v12, historial completo; validado con
`check_submission.py`; pseudo-test con el calendario real: 29,6 s para las 2.016 h, 0 infracciones).
Interruptores: `{"jump": false, "jump_meas": false}` → v12 del equipo exacta; además `{"co2_cal": false}` → v9.

| Agente (modelos `fit`) | Val. A | Val. B |
|---|---|---|
| v9 (23,45) | 20,52 | 18,45 |
| v12 del equipo (23,53) | 20,75 | 18,91 |
| **v13** = v12 + salto | **21,18** | **19,19** |
| v13 + modelos por tramos | 21,11 | 19,20 |

La calibración del CO₂ y los modelos por tramos corrigen lo mismo (CO₂ de las próximas horas): juntos no suman, y los
tramos bajaron en el público con la v10 → la v13 no los lleva.

**Resultado público v13: 23,05** (v12: 23,53) → descartada; la base sigue siendo la v12 del equipo (23,53).
El ranking público usa **1.008 h = 6 semanas**, no las 12. Se intentó reproducir la caída en validación (v12 → v13):

| Escenario simulado (el agente cree que vienen las 11 semanas) | v12 | v13 | dif. |
|---|---|---|---|
| 11 semanas seguidas | 20,75 | 21,18 | +0,43 |
| 6 primeras seguidas | 15,75 | 15,91 | +0,16 |
| 6 alternas (1.ª, 3.ª, ...) | 21,24 | 21,72 | +0,48 |
| 5 alternas (2.ª, 4.ª, ...) | 20,17 | 20,00 | −0,17 |
| 6 últimas | 26,69 | 27,35 | +0,66 |
| Cada semana como episodio independiente (SOC 50 %) | 21,58 | 21,52 | −0,06 |

Ninguno da −0,48 → la causa no está identificada. Patrón preocupante: v10 (+0,33 val. → −0,50 público) y v13 (+0,43 →
−0,48) suben en validación y bajan en el público; la v12 del equipo (+0,22/+0,46 → +0,08) apenas se traslada.
No volver a probar el salto sin saber antes cómo simula el servidor las semanas (p. ej. con un envío de diagnóstico).

## 10. Sobre la v12 del equipo (14/11 noche) — nada supera el ruido del público

Un compañero comprobó que el servidor encadena las semanas (un agente que se para al ver un salto de fecha bajó muchísimo).
El público usa 6 de las 12 semanas: el ruido de una nota de 6 semanas es grande para cambios que ganan en unas semanas y
pierden en otras (v13: −2,5 a +3,4 puntos por semana; con 6 semanas sale peor que la v12 el 14–27 % de las veces).
Criterio: subir en A **y** B, con ganancia repartida por semanas (`/tmp/.../lab/noise.py` → probabilidad de salir peor).

Base reproducida: `parametres.json` por defecto = v12 exacta (A 20,75 · B 18,91; `jump` desactivado por defecto).

| Probado sobre la v12 (modelos `fit`) | A | B |
|---|---|---|
| Calibración CO₂: `co2_cal_tau` 4/16/sin decaimiento | −0,04/−0,03/−0,07 | −0,05/−0,04/−0,04 |
| `co2_cal_k` 0,5/1,5 · `co2_cal_clip` 0,03/0,12 · media en lugar de mediana | −0,10/−0,09 · 0,00/0,00 · −0,01 | −0,16/−0,16 · −0,07/−0,01 · −0,01 |
| `co2_cal_lags` [1,2,3] + `co2_cal_mult` (cociente) | +0,09 | +0,02 |
| Calibración también del precio (`price_cal`, tau 8/4) | −0,03/0,00 | −0,01/+0,01 |
| `net0` con señales de corto plazo + pérdida L1 (`train_v7.py --net0x`): MAE −5 % (A), −3 % (B) | −0,01 | +0,01 |
| Modelos por tramos (tipo v10) encima de la calibración | −0,09 | −0,01 |

Valor de mejorar la hora actual: unos +0,05 por cada 10 % menos de error del consumo neto (MAE actual 62 kW).
Margen que queda con información perfecta: CO₂ de +3..+7 h (+0,6) — es la única vía con recorrido, pero requiere un
modelo nuevo de CO₂ a corto plazo, no ajustes de parámetros.

## 11. v14 = v12 + vaciar la batería al final del test (31/8) — pendiente del público

Diagnóstico v12 frente al óptimo (val. A): casi no tira solar (1,5 MWh) ni descarga de más (4,5 MWh); la diferencia está
en la calidad de las previsiones y en que **acaba la simulación con 2.838 kWh en la batería** (energía pagada y no usada).
La v12 solo ve el final cuando la ventana de mercado se acorta (2 h antes).

- Tratar **cada** final de mes como final: A −0,68 · B −0,23 (no).
- Saber el final real (11 semanas como episodios independientes, 22 finales): +0,52/+0,53; **sin la prudencia `k_future`
  en esas horas: +0,57/+0,58**, mejora en 14 de 22 finales y lo peor es −0,04. Con prudencia, el agente descargaba de
  madrugada y cargaba poco por la tarde (ver pseudo-test).
- Regla de la v14 (`end_cal`, `end_kf`=0): la única semana tras la que seguro no hay más es la de agosto (no hay datos
  de train después). En sus últimas 24 h se planifica hasta el 31/8 23:00 con valor final 0 y sin prudencia.
  Fuera de esa semana el agente es **idéntico a la v12** (val. A 20,75). Pseudo-test con calendario real: 20,21 → 20,30.
- En el público solo cambia algo si incluye la semana de agosto (si sale 23,53 exacto, no la incluye).
Zip: `dificil/lliuraments/hard_v14_v12_final_agost.zip` (mismo `.pkl` que la v12).

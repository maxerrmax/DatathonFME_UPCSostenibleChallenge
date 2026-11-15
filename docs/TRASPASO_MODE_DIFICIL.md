# Traspaso — Mode Difícil (estado a 14/11/2026, ~19:30)

Resumen para continuar en una sesión nueva sin repetir pruebas. Las instrucciones generales están en `CLAUDE.md`.

## 1. Estado actual

| | |
|---|---|
| **Mejor entrega (base)** | v9 con historial inicial completo: **23,45** en el ranking público (7.º) |
| Zip exacto | `dificil/lliuraments/hard_v9_public_23.45.zip` (= `dificil/arxiu/submissions_hard_diag/hard_v9_hist0ext.zip`) |
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
- **`net0` entrenado con ventanas horarias desplazadas cada 5 min** (15/11, `dificil/diag/net0_aug.py`, sobre v12 en
  los 3 pliegues): ×12 filas, pero MAE de la hora actual peor en los 3 pliegues (±25 min: +1,4/+2,2/+7,3 kW sobre el
  control; ±10 min: +0,6/−0,3/+4,5; con el desfase como variable: +2,1/+1,0/+6,9), sobre todo de 9 a 16 h (la solar
  depende mucho de la hora exacta). Puntuación ±10 min: 19,023 vs 19,02 (ruido). Las ventanas desplazadas comparten
  casi todas las muestras con las oficiales: no aportan información nueva.
- **Modelos de mercado especializados en h+3..+11** (15/11, `dificil/diag/mkt_near.py`, sobre v12, 3 pliegues; el valor
  de la información de mercado está todo en 3..11): MAE CO₂ −13/−15/−13 % (h+3 −31..−36 %), precio −7/−5/−8 %.
  Puntuación: CO₂ y precio 19,073 (+0,05; +0,01/+0,13/+0,02); solo CO₂ **19,083 (+0,06; +0,04/+0,11/+0,04)**.
  Positivo en los 3 pliegues pero pequeño: el calibrado en línia del v12 ya recoge buena parte del error de h+3..+5.
  No subido al público (el ruido del público, ±0,35, no permite confirmar +0,06).
  Variantes (`dificil/diag/strats_near.py`, Δ sobre v12 por pliegue): sin el calibrado en línea −0,07
  (−0,04/−0,11/−0,06: el calibrado sigue haciendo falta); media proper/original +0,02; modelos por franjas
  (3-5, 6-8, 9-11) +0,02. **v19 + CO₂ proper: 19,230 (+0,21; +0,30/+0,24/+0,09)** frente a v19 solo 19,177
  (+0,16): los dos cambios se suman. Candidato a subir una vez (cambiar solo si supera 23,53).
- **Ensemble de semillas de CO₂**: 2 semillas 20,54 (+0,02, ruido) a 21 ms/paso; 3 semillas 20,52 a 25 ms/paso
  (riesgo con el límite de 60 s). No adoptado.

Sobre el agente MPC antiguo (`dificil/arxiu/submission_hard/`, 22,95 en el público), conclusiones que también aplican:
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

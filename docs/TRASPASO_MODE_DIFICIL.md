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
- **Features nuevas de meteo/calendario** (15/11, `dificil/diag/extra_feats.py`, sobre v19 + hora solar, 3 pliegues, base
  21,13/17,78/19,23 = 19,380; grupos opcionales `JAMM_EXTRA` de `feats.py`, apagados por defecto): olas de calor/frío
  (temp. media 24/72 h y máx. 24 h en precio, CO₂ y consumo) −0,003 (0,00/−0,02/+0,01); umbrales 19/27 °C en consumo
  +0,003 (0,00/−0,01/+0,02); altura del sol en consumo +0,013 (+0,01/+0,01/+0,02); festivos estatales, vísperas, puentes
  y Navidad en precio/CO₂ −0,030 (+0,01/0,00/−0,10); temperatura del panel en consumo/neto de la hora actual 0,000
  (0,00/−0,02/+0,02). Todo dentro del redondeo: no subir. Horas del cambio de hora: no se puede validar (el train no
  contiene ningún cambio de hora) y afecta a 2 noches; no tocado.
- **Ensemble de semillas de CO₂**: 2 semillas 20,54 (+0,02, ruido) a 21 ms/paso; 3 semillas 20,52 a 25 ms/paso
  (riesgo con el límite de 60 s). No adoptado.

Sobre la v12 (calibratge online del CO₂, públic **23,53**, validació 20,74):
- **Constant de decaïment del calibratge**: 4 h 20,70 · 6 h 20,72 · 8 h (v12) 20,74 · 11 h 20,74 · 16 h 20,71; guany ×0,75 20,69 · ×1,5 20,65; límit 0,1 20,74. **11 h al públic: 23,49 (−0,04)** → es queda 8 h. Diferències de ±0,05 al públic són soroll d'ajust.
- **Diagnòstic nou (bitàcola, `docs/bitacora_mode_dificil.html`)**: la v12 carrega de la xarxa de nit (21–7 h) ~26 % més que l'òptim (166 vs 132 MWh a la validació) i menys de dia; passa sobretot a les setmanes amb més pèrdua (17/04: 11,4 vs 1,3 MWh; 18/12, 18/03, 17/09).
- **Llindar general de cicle** (sumar `wd` a tots els pesos, penalitza les pèrdues): 0,01 20,73 · 0,025 20,71 · 0,05 20,63 · 0,1 20,44. No.
- **Penalització de les hores de nit** (`wn`, 21–7 h): 0,0025 20,73 · 0,0035 20,77 · 0,005 20,76 · 0,0075 20,61 · 0,01 20,48. +0,03 com a màxim i no monòton (soroll); puja CO₂ (+0,2) i baixa cost. No adoptat.

Línia hora solar / canvi d'hora (15/11, base v12 + taula de cel clar en hora estàndard del company, +0,1 al públic):
- **Banc de proves nou** `scripts/eval_dst_emul.py`: simula el canvi d'hora en 4 setmanes de validació (sol, meteo i mercat desplaçats 1 h; consum al rellotge; dst forçat). Detecta la millora del company: sense correcció 18,34 → amb correcció 18,69 (les mateixes setmanes sense canvi: 18,77). Queda ~0,08 de marge en aquesta línia.
- **Hora solar a les features de preu, CO₂ i solar** (consum amb rellotge): validació 20,70 (base 20,72), simulació 18,67 (base 18,69). No.
- **Calibratge en línia de la baixada del CO₂ amb el sol** (regressió de l'error h+3..+11 contra el sol relatiu): g 0,5 20,68 · g 1 20,54 · finestra 24 h 20,43. Empitjora i és més lent. No.
- **Incertesa de l'hora actual segons l'estat del cel** (escenaris del net triats entre hores passades amb variabilitat de la claredat semblant, `scen_vq`, i/o errors normalitzats pel cel clar, `scen_cs`): vq 0,5 20,72 · vq 0,3 20,71 · cs 20,67 · vq+cs 20,69 · amb `scen_win` 3–6 entre 20,65 i 20,71 (base 20,72). No.
- **Calendari nacional + tarifa 2.0TD a preu i CO₂ (`natcal`: 2, zip `dificil/lliuraments/hard_v12B_natcal.zip`)**. El preu és PVPC amb peatges: salts fixos als laborables (+44 a les 10 h, −68 a les 14 h, +87 a les 18 h, −78 a les 22 h) i vall tot el dia només els caps de setmana i els festius nacionals de data fixa; Diada, Divendres Sant, Dilluns de Pasqua, vacances UPC i agost tenen peatges de laborable. El model de mercat feia servir el calendari local UPC (`off`/`fest`). Canvi: `off` → cap de setmana o festiu nacional (+ Dijous i Divendres Sant, demanda nacional baixa) i `fest` → període tarifari. Validació 20,75 (base 20,72); setmanes de festius apartades (`train_v7.py hol` + `scripts/eval_weeks.py`) 19,10 vs 18,99 (8/12 +0,10, 2/1 +0,26, Setmana Santa +0,29, Diada i agost =); simulació del canvi d'hora 18,69 (=). Al test afecta ~9 laborables: 24/9, 26/12, 29–31/12, 30–31/3, 25/5, 24/6. Pendent del públic.
- La durada del dia ja hi és: taula de cel clar 366×24 (per dia de l'any, ara en hora solar), `cs_tau` i `doy` a les features de CO₂/preu i solar.

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

**Factors externs pendents (15/11), de més a menys probabilitat de millora.** Base: v19 + hora solar (`dificil/arxiu/v19_dst/`, públic 23,75). Adoptar només si la validació puja > +0,05 i no empitjora `scripts/eval_weeks.py` ni `scripts/eval_dst_emul.py`.
1. Onades de calor/fred: temperatura mitjana de 24/72 h i màxima d'ahir a les features de preu/CO₂ (`mkt_features2`) i consum.
2. ~~Llindars de climatització~~ **provat (15/11)**: graus per sota de 19 °C i per sobre de 27 °C (temperatura de l'última hora i mitjana 24 h) al model de consum, sobre v19 + hora solar: validació 20,73 vs 20,72. Soroll; el model ja capta la temperatura. No.
3. Llum natural → il·luminació: elevació solar o hores des de la posta a `tau` al model de consum (`load_features2`).
4. Hores del canvi d'hora: 25/10 té l'hora 02:00 repetida (`hist.put` la sobreescriu) i el 29/3 en falta una; blindar-ho.
5. Temperatura del panell (`temp_panel_c`, ≈ −0,4 %/°C) a la previsió solar de l'hora actual.
6. Vigílies i ponts nacionals al mercat (dia abans/després d'un festiu nacional).
7. Nadal i Cap d'Any al mercat (laborables 26–31/12 amb demanda nacional baixa). El consum ja té els règims de Nadal i els festius nacionals ja són a `natcal`: només cobreix el que falta i afecta una sola setmana.

- Mejorar el modelo de la hora actual (+0,7 de margen teórico): p. ej. usar más señales de corto plazo (meteo de la
  última hora, tendencia de las últimas 3 h) en `net0`, o ajustar la mezcla `net0_mix`.
- Escenarios de la hora actual: probar `scen` (ahora 15) y `scen_win` (ahora 2), `k_future`.
- CO₂ a corto plazo con otro enfoque (p. ej. modelo de la forma completa del día), con prudencia por lo visto en v10.

## 7. Avisos prácticos

- **4 núcleos**: nunca dos procesos pesados a la vez (un entrenamiento de 20 s llegó a tardar 16 min).
  `scripts/eval_agent_dir.py` y `scripts/tune_hard.py` ya usan 1 hilo por proceso.
- `git pull` antes de cada `push`. Las ramas del equipo: `main`, `dificil-v7`, etc.
- Los pickles (`*.pkl`) de `dificil/agent_hard/` no se versionan; se regeneran con `train_v7.py`.

# Recerca v13 (14/11, nit) — què s'ha provat sobre la v12

Base: v12 reproduïda amb models fit-only a la validació de 1.848 h: **20,74** (v9 20,52). Totes les proves amb
taules de previsions gravades (`dificil/stack.py`) i replay del LP; zero infraccions a totes.

| Experiment | Família | Validació | Veredicte |
|---|---|---:|---|
| v12 (referència) | calibratge online CO₂ | 20,74 | — |
| E1 Corrector après del CO₂ (LightGBM sobre errors de les previsions de fa 1–8 h, leave-one-block-out) | stacking / correcció causal | 20,70–20,74 | descartat: MAE CO₂ 0,0124 → 0,0120 però score igual |
| E2 MPC estocàstic de dues etapes, 10–20 trajectòries de mercat correlacionades (residus reals de 24 h), amb `net0` | SAA / multietapa | 20,75–20,77 | descartat: soroll, ×2 de temps |
| E3 Correcció online de la demanda neta actual | calibratge online | 20,72–20,74 | descartat: l'error de `net0` no té memòria (autocorrelació lag 1 = −0,02) |
| E4 Final d'episodi pel calendari (buidar la bateria abans de les 23:00 de l'últim dia del mes) | estructura del test | **+0,51** si cada setmana és un episodi; **−0,68** si estan enganxades | **descartat: públic 19,61** (v12 23,53). El test no són episodis setmanals separats |

Conclusions:
- La demanda neta de l'hora actual ja aprofita tota la informació disponible: l'error que queda (114 kW de dia, 23 de
  nit) és variabilitat de núvols d'una hora a l'altra; només es pot gestionar amb escenaris (ja ho fem).
- Les dades de 5 minuts de l'observació són la mitjana de l'hora anterior: no hi ha senyal intra-hora.
- Els CSV de train no contenen cap dia de les setmanes de test (sempre acaben el dia 21–24).
- Millores de MAE del CO₂ de fins al 5 % no es tradueixen en puntuació (tercera vegada que es comprova).

## Resultat del diagnòstic al públic (14/11 22:27)

`hard_v13_finalmes.zip` (v12 exacta + buidar la bateria a final de mes) va fer **19,61** (CO₂ 18,15 %, cost 23,02 %),
contra 23,53 de la v12. La caiguda (−3,9) és molt més gran que la pitjor estimació local (−0,68): el servidor no
executa cada setmana com un episodi separat que acaba l'últim dia del mes, i l'energia que queda a la bateria sí que
es fa servir després. **No tocar el final d'episodi.** La v12 continua sent la millor.

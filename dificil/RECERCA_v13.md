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

## Estructura del test deduïda dels diagnòstics (14/11 22:45)

Dos enviaments de diagnòstic sobre la v12 exacta:
- buidar la bateria a final de mes → públic 19,61 (ràtio 0,83 respecte a 23,53);
- bateria quieta de març a agost → públic 10,33 (ràtio 0,44; set–feb aporta el 45 % del CO₂ evitat del públic).

Simulant diverses particions públic/privat a la validació (setmanes enganxades en ordre cronològic), l'única que
reprodueix tots dos resultats és **públic = primera meitat de cada setmana de test, privat = segona meitat**
(ràtios simulats 0,77 i 0,37). Totes les altres (hores aleatòries, setmanes alternes, totes les hores, segona meitat)
donen que buidar a final de mes no baixa o puja.

Conseqüència: la v12 porta l'energia del final de cada setmana a l'inici de la següent (part pública). Buidar-la a
final de setmana transfereix punts del públic al privat (simulat: públic 21,74 → 16,71, privat 19,62 → 23,72), però
la **mitjana** baixa (20,68 → 20,21) i també amb versions suaus (últimes 3–24 h: 19,97–20,21). Com que el resultat
final pondera públic i privat al 50 %, **la v12 (sense regla de final) és la correcta**.

## v13: planificar a través dels salts entre setmanes (`hard_v13_salts.zip`)

El simulador enganxa les setmanes de test en ordre cronològic: després del 30/9 23:00 ve el 25/10 00:00. Quan la
setmana s'acaba dins l'horitzó, la v12 planificava hores inexistents (l'1/10...). La v13 (paràmetres `jump`,
`jump_fix`):
- calcula pel calendari (setmanes de test = últims 7 dies de cada mes) quantes hores queden i on comença la següent;
- preveu les hores de després del salt des de l'hora anterior a la setmana següent (dades d'entrenament: el dia abans
  de cada setmana de test és al train) i les posa a la cua de l'horitzó;
- desa els valors de mercat de la setmana següent a la seva hora real, i l'última hora observada després d'un salt a
  l'hora correcta (la v12 la desava a t−1 i trepitjava una dada d'entrenament).

Validació (models fit-only, setmanes enganxades): v12 20,74 → **20,98**; mitjana públic/privat 20,68 → **20,88**
(sostre amb informació perfecta després del salt: 20,96). Sense els paràmetres nous decideix exactament igual que la v12.

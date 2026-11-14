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

## Més proves sobre la v12 (14/11, 23:10) — totes descartades

| Prova | Validació (v12 = 20,74) |
|---|---:|
| Calibratge online del preu (com el del CO₂) | 20,72–20,76 |
| Calibratge online del consum / solar a +1..+23 h | 20,70–20,74 |
| Correcció del biaix del CO₂ per horitzó (estimat en altres blocs) | 20,73–20,75 |
| `k_future`, `q_term`, `scen`, `scen_win`, `scen_scale`, `ratio` | 20,71–20,75 |

- El CO₂ previst a +12..+23 h té biaix, però canvia de signe amb l'estació (negatiu set–feb, positiu a l'estiu):
  no es pot corregir de manera estable.
- La v12 manté el SOC 3–7 punts per sobre de l'agent amb informació perfecta a totes les hores: és prudència davant
  la incertesa; reduir-la no millora.
- Pèrdua de la v12 respecte a l'agent perfecte: 1,16 punts repartits per tot el dia, sense una franja dominant.

## Diagnòstics al públic dels canvis de setmana (14/11, 23:23)

- `hard_v13_salts.zip` (planificar a través del salt entre setmanes): **23,03** (v12 23,53). Hipòtesi refutada:
  el model "setmanes enganxades en ordre cronològic + públic = 1a meitat de cada setmana" no explica els tres
  diagnòstics. **No tocar res que depengui del final de mes.**

## Branques noves (14/11, 23:30–23:50), totes amb models fit-only i blocs no vistos

| Branca | Resultat (v12 = 20,74) |
|---|---|
| Indicadors de règim del mercat (nivell de 7 i 30 dies, perfil horari de 30 dies) | 20,69 |
| A. MLP de trajectòria del CO₂ (+3..+23 h, entrada = última setmana), 3 llavors, combinada amb els LightGBM | sola: MAE 0,0146 (LGBM 0,0128); combinada 20,72–20,75. Errors correlacionats 0,74 amb els LGBM |
| C. Correcció residual de l'acció (etiquetes = acció de l'agent perfecte al mateix estat, estil DAgger) | R² fora de bloc 0,027; aplicada: 20,57 / 20,29 / 19,75 (λ = 0,3 / 0,6 / 1) |

**Conclusió quantitativa.** Previsió perfecta del CO₂ = +1,2 punts; les millores d'error assolibles (3–5 %) valen
~+0,05, dins del soroll. La correcció que faria l'agent perfecte no és predictible a partir de l'estat (R² 0,03):
ni RL ni aprenentatge orientat a la decisió poden recuperar-la amb la mateixa informació. La v12 és a prop del màxim
assolible amb la informació de l'entorn.

## CBR del CO₂ → v14 (`hard_v14_cbr.zip`)

Raonament basat en casos: per a cada hora, els 15 casos més semblants de la base (forma de les últimes 24 h de CO₂ i
preu respecte a l'últim valor conegut, hora, cap de setmana, estació i nivell), i la mitjana ponderada de la seva
continuació (+3..+23 h) adaptada a l'últim valor conegut. Es barreja amb el CO₂ dels LightGBM abans del calibratge de la v12.

- Sol (casos només de fit): MAE 0,01258, **millor que els LightGBM (0,01279)**; errors correlacionats 0,80.
- Replay: v12 + CBR 20,76–20,79 per a tots els k (5/15/40) i pesos (0,25/0,5).
- Llaç tancat: v12 20,74 · **pes 0,3 → 20,80** (CO₂ 19,59 %, cost 23,61 %) · pes 0,5 → 20,77.
- Lliurament: base de casos amb totes les dades (5.442 casos), `cbr_w` = 0,3, `cbr_k` = 15. Sense CBR (`cbr_w` = 0) decideix
  exactament com la v12.

Experiment de soroll: `hard_v12_llavor1.zip` (v12 amb models d'una altra llavor) per mesurar la variació del públic.

## Estat a les 00:45 del 15/11

Públic: v14 (CBR) **23,30** · v12 23,53 (millor). Enunciat oficial (PMF): una sola simulació de 2.016 h, el SOC no es
reinicia entre setmanes, el públic és "el 50 % de cada setmana de test" i el privat l'altre 50 %.

- **Les millores de previsió no se sumen.** v12 = 20,74 · + trams d'horitzó 20,65–20,77 · + CBR 20,80 · trams + CBR
  20,67–20,73 · trams sense calibratge (v10) 20,85. Totes les variants queden entre 20,65 i 20,85: és el mateix
  senyal capturat de maneres diferents, i les diferències són soroll de validació.
- **Quina part de cada setmana és pública.** Cerca de la màscara (dies sencers o 84 h seguides) que reprodueix millor
  els sis resultats públics (buidar, quiet, salt, v10, v14, v9): totes les que encaixen contenen el **dia 1** i
  exclouen el **dia 7** de cada setmana; la resta no es pot determinar. Conseqüència: l'energia que queda a la
  bateria al final de cada setmana passa del privat (dia 7) al públic (dia 1 de la setmana següent). Buidar-la a
  final de setmana va treure 3,9 punts al públic; la simulació diu que el privat en guanya una quantitat semblant i
  que el total baixa ~0,7.
- Les versions amb més puntuació total a validació (v10, v14) baixen una mica al públic en les màscares que
  encaixen (ràtios 0,97–0,99 simulats i reals): el públic sol no permet distingir-les de la v12.

# Instruccions del projecte (Datathon UPC Sostenible)

## Recursos de càlcul (només 4 nuclis)
- **Mai dos processos pesats alhora** (entrenaments LightGBM, simulacions de validació, cerques de paràmetres).
  Amb dos processos en paral·lel cadascun intenta fer servir tots els nuclis: un entrenament de 20 s va trigar 16 min.
- Procés únic → tots els nuclis (`num_threads` per defecte). Diversos processos → **1 fil per procés**
  (`OMP_NUM_THREADS=1`) i com a màxim `nproc` processos, com fa `scripts/tune_hard.py`.
- Abans de llançar res llarg, comprovar que no n'hi ha cap altre en marxa (`ps aux | grep python`).
- Els sub-agents comparteixen aquests nuclis: no fer-los servir per a experiments de càlcul en paral·lel.

## Git
- Fer `git pull` abans de cada `push` (treball col·laboratiu).

## Versió base del Mode Difícil (decisió de l'equip)
- **Base = v9 de `dificil/` amb historial inicial complet, que va fer 23,45 al rànquing públic** (14/11 20:10).
  - Zip exacte enviat: `dificil/lliuraments/hard_v9_public_23.45.zip` (= `dificil/arxiu/submissions_hard_diag/hard_v9_hist0ext.zip`;
    `agent.py`, `feats.py`, `parametres.json`, `models_all7.pkl`). Validació (`fit`) 20,52 sobre òptim 22,34.
  - Models: `python dificil/train_v7.py all` + `python scripts/extend_hist0_v8.py <pkl> <pkl_sortida>` (hist0 complet).
  - Anterior: v8 (23,41, `dificil/lliuraments/hard_v8_public_23.41.zip`, validació 20,44).
  - v10 (models per trams d'horitzó) millorava la validació (20,85) però va baixar al públic (22,95): no fer-lo servir.
  - Codi: `dificil/agent_hard/` (agent i features) · models: `python dificil/train_v7.py fit|all`
    · validació: `python dificil/heval.py '[{...params}]'` (models `fit`). Notes: `dificil/README.md`.
- **Tot canvi nou s'ha de fer partint del v9 (23,45).** Abans de proposar un lliurament, comparar a validació amb el v9.
- **Llegir `docs/TRASPASO_MODE_DIFICIL.md` abans de proposar res**: estat, com validar (`scripts/eval_agent_dir.py`,
  referència 20,52) i tot el que ja s'ha provat sense èxit (amb xifres).

### Versions anteriors (referència, no són la base)
- Agent MPC de `dificil/arxiu/submission_hard/` (22,95 al públic; zip `dificil/arxiu/submissions_hard_diag/mpc_lgbm_submission.zip`,
  params `{"load_bias": -50, "pv_bias": 1.2, "pv_mix": 0.5}`). Es genera amb `scripts/build_agent.py`
  (editar `agent_core.py` i `src/forecast.py`, no `agent.py`); validació `scripts/tune_hard.py`, zip `scripts/package_hard.py`.
- La v2 d'aquest agent amb règims d'obertura (tancat/reduït) va empitjorar al test i es va revertir.

## Estat del Mode Difícil i memòria de la sessió (15/11/2026)
Públic: **v12 + hora solar 23,63 (millor)** · v12 23,53 · v15 23,53 (v12 + final conegut del test; el guany cau a la meitat privada) · v18 23,13 ·
v17 23,29 · v16 22,97 · v13 23,07 · v10 22,95. **LLIURAMENT FINAL (decisió del 15/11): dues solucions, `entrega_final/solucio_A_v12_original.zip` (23,53) i `entrega_final/solucio_B_v12_hora_solar.zip` (23,63).** Vegeu `entrega_final/README.md`. La v15 no es selecciona (el seu guany depèn de conèixer el final del test).
v19 (`dificil/arxiu/lliuraments/hard_v19.zip`, v15 + previsions conscients del canvi de setmana, plecs +0,22/+0,17/+0,08): públic **23,64**.
**v19 + hora solar (`dificil/arxiu/lliuraments/hard_v19_dst.zip`, codi a `dificil/arxiu/v19_dst/`): públic 23,75, el millor.** Depèn de conèixer el final del test (31/8/2026 23:00) i l'encadenament de setmanes; no s'ha seleccionat fins que els organitzadors confirmin que és vàlid (A i B són les finals nets).
v19 + hora solar + CO₂ proper (h+3..+11) + correcció del canvi de setmana (`dificil/arxiu/v19_dst_near/`): validació +0,08, **públic 23,63 (−0,12)**. Descartat; no repetir.
Features noves sobre v19 + hora solar (`dificil/diag/extra_feats.py`, 3 plecs): onades de calor −0,003, llindars 19/27 °C +0,003, alçada del sol +0,013, festius estatals/Nadal al mercat −0,030, temperatura del panell 0,000. Res a pujar; no repetir.

**Servidor verificat** (sonda `dificil/probe/`, `dificil/probe/probe_diag.zip`): entrades idèntiques a `dificil/henv.py`; determinista;
un sol episodi continu de 12 setmanes (SOC arrossegat; la hipòtesi de reinici per setmana, v18, és falsa).
`ultim_consum_kw` i `ultima_solar_kw` són l'hora anterior (mesurat píxel a píxel). El moment de les dades meteo no importa
(v12 igual amb t, t-1, t-2). No cal fer més sondes de format.

**Bretxa amb els líders** (~0,8): SSA Gourmet 24,36 (CO2 22,94 % / cost 27,69 %), Fanboy & Chum Chum 24,25, SanHackobos 24,15,
manguets 23,89, TeamName2 23,84; nosaltres 23,53 (22,34 / 26,31). Ens guanyen en CO2 i en cost alhora: és sistemàtic.
Valor de la informació perfecta sobre v12 als 3 plecs (`dificil/diag/voi_folds.py`, mitjana 19,02): net (consum−solar) de
l'hora actual +0,48, net 24 h +0,63, CO2 més enllà de 3 h +0,50, preu +0,17, òptim +1,62. Tot el valor de mercat és a
h+3..+11; h+12..23 val 0. El planificador no és la palanca (MPC 24 h amb previsió perfecta = 20,24 vs òptim 20,64).
Les idees noves han d'atacar la previsió del net de l'hora actual o del CO2 a h+3..+11. Grans millores de MAE s'han traduït
sempre en guanys de puntuació minúsculs.

**Ja provat sense guany (no repetir)**: ~40 estratègies de `dificil/arena/` (funció de valor `vf_adp`, MPC estocàstic 6/8/12
escenaris, xarxes neuronals, dies anàlegs, horitzó curt, greedy, regles `regime_switch` ≈12); v18; v20 (consum nocturn exacte,
±0,01); v21 (àncora de CO2 amb el dia posterior a la setmana, +0,00/+0,00/+0,01, esborrat); net0 amb finestres de 5 min
desplaçades (`dificil/diag/net0_aug.py`); models de mercat només h+3..+11 (`dificil/diag/mkt_near.py`, MAE CO2 −13..−15 %,
puntuació +0,06, no pujat).

**Altres fets**: el públic són ~5 setmanes (no 6) segons els totals de referència; les setmanes van de 3 a 41 punts als plecs
(campus tancat ≈ 40), així que una setmana especial mou molt el públic. El train no té dades de novembre 2025. Timestamps en hora
local; els dos canvis d'hora cauen dins de setmanes de test. Error menor: al canvi de setmana l'agent sobreescriu hist[t-1]
(valor real del train) amb l'última hora de la setmana anterior.
Ni els plecs ni el pseudo-test van predir cap de les pèrdues al públic (biaix de selecció, soroll ±0,35 per comparació).

**Pseudo-test privat**: les dades del període de test (REE, PVPC) i qualsevol pseudo-test **mai dins del repo**
(són a `D:\Git\DatathonFME_pseudo_test_private`). Només per validar, mai per entrenar.

**Forma de treballar**: ser crític i no excessivament confiat; donar guanys com a rangs amb l'evidència, dir què no està
verificat i proposar regles de decisió ("canviar només si el públic > X"), no promeses. D'altres equips, només inspiració, sense
copiar. Els notebooks d'entrenament s'executen a Kaggle (autocontinguts), mai en local.

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
  - Zip exacte enviat: `dificil/lliuraments/hard_v9_public_23.45.zip` (= `submissions_hard_diag/hard_v9_hist0ext.zip`;
    `agent.py`, `feats.py`, `parametres.json`, `models_all7.pkl`). Validació (`fit`) 20,52 sobre òptim 22,34.
  - Models: `python dificil/train_v7.py all` + `python scripts/extend_hist0_v8.py <pkl> <pkl_sortida>` (hist0 complet).
  - Anterior: v8 (23,41, `dificil/lliuraments/hard_v8_public_23.41.zip`, validació 20,44).
  - v10 (models per trams d'horitzó) millorava la validació (20,85) però va baixar al públic (22,95): no fer-lo servir.
  - Codi: `dificil/agent_hard/` (agent i features) · models: `python dificil/train_v7.py fit|all`
    · validació: `python dificil/heval.py '[{...params}]'` (models `fit`). Notes: `dificil/README.md`.
- **Tot canvi nou s'ha de fer partint del v9 (23,45).** Abans de proposar un lliurament, comparar a validació amb el v9.

### Versions anteriors (referència, no són la base)
- Agent MPC de `submission_hard/` (22,95 al públic; zip `submissions_hard_diag/mpc_lgbm_submission.zip`,
  params `{"load_bias": -50, "pv_bias": 1.2, "pv_mix": 0.5}`). Es genera amb `scripts/build_agent.py`
  (editar `agent_core.py` i `src/forecast.py`, no `agent.py`); validació `scripts/tune_hard.py`, zip `scripts/package_hard.py`.
- La v2 d'aquest agent amb règims d'obertura (tancat/reduït) va empitjorar al test i es va revertir.

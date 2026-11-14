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

## Mode Difícil
- Agent: `submission_hard/` (es genera amb `scripts/build_agent.py`; no editar `agent.py` a mà, editar `agent_core.py` i `src/forecast.py`).
- Validació: `scripts/tune_hard.py <dir_models> '<json params>' ...` · Lliurament: `scripts/package_hard.py`.

## Versió base del Mode Difícil (decisió de l'equip)
- **Base = versió que va fer 22,95 al rànquing públic** (`submissions_hard_diag/mpc_lgbm_submission.zip`,
  params `{"load_bias": -50, "pv_bias": 1.2, "pv_mix": 0.5}`, validació local 17,00 / equip 20,05).
- Tot canvi nou s'ha de fer partint d'aquesta base. La v2 amb règims d'obertura (tancat/reduït) va empitjorar
  al test i es va revertir (commit "Revert ... règims reals d'obertura").

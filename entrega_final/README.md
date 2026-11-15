# Entrega final — Mode Difícil · equip JAMM

Les dues solucions que triem per a l'avaluació final del **Mode Difícil** (6.000 kWh, ±1.500 kW, solar ×12,
finestra de mercat de 3 h). Les dues són el mateix agent amb una única diferència (vegeu «Diferència entre A i B»).

| | **Solució A** — v12 original | **Solució B** — v12 + hora solar |
|---|---|---|
| Fitxer | [`solucio_A_v12_original.zip`](solucio_A_v12_original.zip) | [`solucio_B_v12_hora_solar.zip`](solucio_B_v12_hora_solar.zip) |
| Codi legible | [`codi_solucio_A/`](codi_solucio_A) | [`codi_solucio_B/`](codi_solucio_B) |
| Puntuació al rànquing públic | **23,53** (CO₂ −22,34 %, cost −26,31 %) | **23,63** (CO₂ −22,38 %, cost −26,55 %) |
| Validació local (models sense la validació) | 20,74 | 20,72 |
| Infraccions | 0 | 0 |
| Mida | 14,4 MB | 13,9 MB |
| SHA-256 del zip | `40d1cb745b22…dd5771d8` | `d17dacd92169…18dcefb2` |

SHA-256 complets: vegeu [`SHA256SUMS.txt`](SHA256SUMS.txt). La validació local és la de l'equip (últims 7 dies de cada bloc
mensual del train, 1.848 h) i es pot reproduir amb `scripts/eval_agent_dir.py` (vegeu «Com verificar»).

> **Com llegir les xifres.** El soroll d'entrenament entre llavors és d'uns ±0,2 punts al públic (la mateixa v12
> amb una altra llavor va fer 23,34 i amb dues llavors promitjades, 23,38). La diferència de +0,10 entre B i A és
> **dins d'aquest soroll**: la correcció d'hora solar és coherent tècnicament, però el públic no permet demostrar-ne l'efecte.
> El rànquing final és sobre el conjunt privat, que no veiem.

## Què fa l'agent

Controlador predictiu (MPC) que cada hora resol un LP de 24 h (`scipy`, HiGHS) i aplica només la primera consigna.

- **Previsions amb LightGBM**, entrenades només amb `data/train/`: preu i CO₂ a +3..+23 h (conjunt de models sobre el residu respecte
  al darrer valor conegut), consum, solar (amb una taula de cel clar) i consum net de l'hora actual (model directe, barrejat al 50 %).
- **Calibratge causal en línia del CO₂** (la novetat de la v12): compara les previsions de CO₂ fetes fa hores amb els valors de mercat
  que ara coneix i corregeix les previsions futures amb un efecte que decau amb l'horitzó.
- **Hora actual estocàstica**: l'LP considera 15 escenaris del consum net de l'hora actual.
- **Historial inicial**: el zip du les dades d'entrenament, de manera que els retards de 24/48/168 h existeixen des de la primera hora.
- **Seguretat**: la consigna es talla als límits físics de potència i de capacitat (0 infraccions); si hi ha qualsevol error o l'LP no convergeix, la bateria queda en repòs (0 kW).

Pes de l'objectiu: `0,7·CO₂ + 0,3·0,6213·preu` (la raó 0,6213 kg CO₂/€ és la del baseline del campus).

## Diferència entre A i B

Els timestamps són hora local amb canvi d'hora. La taula de cel clar (quantil 95 % de la solar per dia de l'any i hora) estava
construïda en **hora de rellotge**, de manera que a les setmanes de test del 25–31 de març (canvi el 29/3) i del 25–31 d'octubre
(canvi el 25/10) quedava desplaçada una hora la meitat dels dies. El train no pot mostrar aquest efecte, perquè exclou els
últims 7 dies de cada mes (justament on cauen els canvis d'hora).

La solució B construeix la taula en **hora estàndard** (rellotge menys l'offset d'estiu, calculat només amb el calendari).
`agent.py` i `parametres.json` són idèntics; canvien `feats.py` (33 línies) i els models (reentrenats amb la nova taula).
No s'afegeix cap dada externa.

Verificat: no empitjora la validació (20,74 → 20,72). No es pot validar el seu benefici localment (no hi ha cap dia de canvi d'hora).

## Compliment de la normativa

- Només biblioteques permeses (`numpy`, `scipy`, `lightgbm`, `joblib` i biblioteca estàndard); passa `starter_kit/check_submission.py`.
- Sense xarxa ni dades externes. Els scripts `scripts/download_ree_history.py` i `scripts/build_ree_hourly.py` i l'opció `ree`
  de `dificil/train_v7.py` **no s'han fet servir** per a cap de les dues solucions.
- Zip < 25 MB. Temps d'execució ≈ 16–18 ms per pas en local (≈ 35 s per a les 2.016 h; límit del servidor: 60 s).

## Com verificar

```bash
python starter_kit/check_submission.py entrega_final/solucio_A_v12_original.zip
python starter_kit/check_submission.py entrega_final/solucio_B_v12_hora_solar.zip
(cd entrega_final && sha256sum -c SHA256SUMS.txt)

# Validació de referència de B (models sense la validació; ha de donar ≈ 20,72)
python dificil/train_v7.py fit                     # -> dificil/agent_hard/models_fit7.pkl  (~1 min)
mkdir /tmp/B_fit && cp entrega_final/codi_solucio_B/* /tmp/B_fit/ && cp dificil/agent_hard/models_fit7.pkl /tmp/B_fit/models_all7.pkl
python scripts/eval_agent_dir.py /tmp/B_fit
```

## Com regenerar la solució B

```bash
python scripts/package_final.py        # entrena (llavor 0), amplia l'historial i empaqueta (~2 min)
```

L'entrenament és determinista: el fitxer regenerat és **idèntic byte a byte** al lliurat (comprovat el 15/11/2026).
La solució A és el zip exacte que es va enviar el 14/11; no es regenera (el codi és anterior al commit `7a5ce8e`).

## Notes d'ús dels lliuraments

- Cada solució es puja com a `submission.zip` al formulari del **Mode Difícil** de la plataforma.
- A la plataforma cal triar aquestes dues com a solucions finals. Convé confirmar amb els organitzadors com es calcula el rànquing
  final (només privat o combinat) i quina de les dues solucions compta.

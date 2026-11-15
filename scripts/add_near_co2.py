"""
Afegeix a un fitxer de models els models de CO2 especialitzats en l'horitzó proper (h=3..11), on hi ha tot el valor
de la informació de mercat (vegeu dificil/diag/mkt_near.py i docs/TRASPASO_MODE_DIFICIL.md).

Mateixes features, bases i hiperparàmetres que dificil/train_v7.py; només canvia el conjunt d'entrenament (h=3..11).
L'agent (dificil/arxiu/v19_dst_near/agent.py) els fa servir per a h<=11 i els originals per a h>=12.

Ús:
    python scripts/add_near_co2.py fit|all <pkl_entrada> <pkl_sortida>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "dificil"), str(ROOT / "dificil" / "agent_hard"), str(ROOT / "estandard"),
                str(ROOT / "dificil" / "diag")]

import joblib


def main():
    which, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    import feats
    from hdata import load_hard
    from data import split_val
    from train_forecast import full_hist
    from mkt_near import matrix, fit_models
    dump = joblib.load(src)
    feats.set_cs(dump["pv_cs"])
    df = load_hard()
    tr = split_val(df)[0] if which == "fit" else df
    hist, eh = full_hist(tr)
    dump["mkt2_near"] = {"co2": fit_models(*matrix(hist, eh, "co2"), "co2")}
    joblib.dump(dump, dst, compress=3)
    print("desat", dst, f"{dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

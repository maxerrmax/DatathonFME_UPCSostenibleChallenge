"""
Regenera el lliurament final B (v12 + correcció d'hora solar) de zero, a partir del codi del repositori.

Passos (tots deterministes; llavor 0):
  1. python dificil/train_v7.py all                 -> dificil/agent_hard/models_all7.pkl  (models amb tot el train)
  2. python scripts/extend_hist0_v8.py <in> <out>   -> historial inicial complet de cada sèrie (hist0)
  3. zip amb agent.py, feats.py, parametres.json (de entrega_final/codi_solucio_B/) i models_all7.pkl a l'arrel

Ús:
    python scripts/package_final.py                      # escriu entrega_final/solucio_B_v12_hora_solar.regenerat.zip
    python scripts/package_final.py --out ruta/fitxer.zip
    python scripts/package_final.py --no-train           # només empaqueta (reutilitza models_all7.pkl ja entrenat)

Notes:
  - NO fa servir dades externes: l'entrenament es fa sense l'opció `ree` de train_v7.py.
  - La solució A (v12 original) no es regenera aquí: és el zip exacte que es va enviar el 14/11
    (`dificil/lliuraments/hard_v12_public_23.53.zip`), construït amb el codi anterior al commit 7a5ce8e.
  - Triga ~2 minuts. Cal executar-ho amb cap altre procés pesat en marxa (vegeu CLAUDE.md).
"""
import argparse
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "dificil" / "agent_hard"                     # entrenament: train_v7.py hi deixa models_all7.pkl
CODI = ROOT / "entrega_final" / "codi_solucio_B"           # codi exacte lliurat (agent.py, feats.py, parametres.json)
FILES = ("agent.py", "feats.py", "parametres.json")


def run(*cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "entrega_final" / "solucio_B_v12_hora_solar.regenerat.zip"))
    ap.add_argument("--no-train", action="store_true", help="reutilitza dificil/agent_hard/models_all7.pkl")
    args = ap.parse_args()

    base = AGENT / "models_all7.pkl"
    if not args.no_train:
        run(sys.executable, "dificil/train_v7.py", "all")
    if not base.exists():
        sys.exit(f"Falta {base}: executeu sense --no-train")

    with tempfile.TemporaryDirectory() as tmp:
        ext = Path(tmp) / "models_all7.pkl"
        run(sys.executable, "scripts/extend_hist0_v8.py", base, ext)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name in FILES:
                z.write(CODI / name, name)
            z.write(ext, "models_all7.pkl")
        with zipfile.ZipFile(out) as z:
            member = hashlib.sha256(z.read("models_all7.pkl")).hexdigest()
    print(f"\nZip: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256 zip           : {sha256(out)}")
    print(f"sha256 models_all7.pkl: {member}")
    print("Comproveu-lo amb: python starter_kit/check_submission.py", out)


if __name__ == "__main__":
    main()

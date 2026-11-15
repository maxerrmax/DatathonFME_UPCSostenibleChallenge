"""nn_pre: igual que nn_co2 però amb xarxes PRE-ENTRENADES amb l'historial públic de REE (2022-01 .. 2025-08)
i afinades a cada plec. Llegeix nn_<variant>.npz d'aquesta carpeta (sortida de train_nn.py --pretrain, reanomenada)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_strategy

_NN = load_strategy("nn_co2").BatteryAgent


class BatteryAgent(_NN):
    NN_DIR = HERE

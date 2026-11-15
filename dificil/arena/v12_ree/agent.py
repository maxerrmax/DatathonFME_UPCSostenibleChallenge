"""v12_ree: v12 amb els LightGBM de preu/CO2 re-entrenats afegint l'historial públic de REE (2022-01..2025-08)
com a files extra (pes 0,5). Fa servir models_<x>7ree.pkl en lloc de models_<x>7.pkl (python dificil/train_v7.py f1 ree)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

Base = load_base().BatteryAgent


class BatteryAgent(Base):
    def __init__(self, params=None):
        p = dict(params or {})
        m = str(p.get("models", "models_all7.pkl"))
        if m.endswith("7.pkl"):
            p["models"] = m[:-len("7.pkl")] + "7ree.pkl"
        super().__init__(p)

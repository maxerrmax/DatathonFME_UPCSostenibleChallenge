"""Utilitats comunes de l'arena d'estratègies (Mode Difícil).

Cada estratègia viu a dificil/arena/<nom>/agent.py i exposa `BatteryAgent(params=None)` amb `act(obs) -> float`.
Per reutilitzar el v12 (previsions LightGBM + MPC-LP):

    from common import load_base
    Base = load_base().BatteryAgent
    class BatteryAgent(Base): ...

El runner (`scripts/arena.py`) passa params = {"models": "models_f<k>7.pkl"} (models del plec k, sense la validació).
"""
import importlib.util
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parent
AGENT_HARD = ARENA.parent / "agent_hard"
_BASE = None


def load_base():
    """Mòdul de l'agent v12 (dificil/agent_hard/agent.py), carregat un sol cop."""
    global _BASE
    if _BASE is None:
        if str(AGENT_HARD) not in sys.path:
            sys.path.insert(0, str(AGENT_HARD))      # per a `import feats`
        spec = importlib.util.spec_from_file_location("jamm_base_agent", AGENT_HARD / "agent.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _BASE = m
    return _BASE


_STRATS = {}


def load_strategy(name):
    """Mòdul d'una altra estratègia de l'arena (per combinar-les amb herència múltiple cooperativa)."""
    if name not in _STRATS:
        d = ARENA / name
        sys.path.insert(0, str(d))
        try:
            spec = importlib.util.spec_from_file_location(f"arena_lib_{name}", d / "agent.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        finally:
            sys.path.pop(0)
        _STRATS[name] = m
    return _STRATS[name]


def combo(names, override=None):
    """Classe BatteryAgent que encadena estratègies (MRO en l'ordre donat) amb paràmetres fixats."""
    bases = tuple(load_strategy(n).BatteryAgent for n in names)
    ov = dict(override or {})

    def __init__(self, params=None):
        super(cls, self).__init__({**ov, **(params or {})})
    cls = type("BatteryAgent", bases, {"__init__": __init__})
    return cls

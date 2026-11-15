"""Referència: v12 tal com és a dificil/agent_hard (públic 23,53)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

BatteryAgent = load_base().BatteryAgent

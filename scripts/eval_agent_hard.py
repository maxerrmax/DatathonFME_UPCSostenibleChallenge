"""Avalua submission_hard/agent.py al Mode Difícil (setmanes de validació, models sense aquestes setmanes)."""
import sys, time, json, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_hourly, validation_weeks
from src.hard_env import run

def load_agent_cls():
    spec = importlib.util.spec_from_file_location("agent_hard", ROOT / "submission_hard/agent.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.BatteryAgent

if __name__ == "__main__":
    model_dir = sys.argv[1]
    params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    val = validation_weeks(load_hourly(pv_scale=12))
    A = load_agent_cls()
    t = time.time(); r = run(A(model_dir=model_dir, params=params), val)
    print(f"{json.dumps(params)} score={r['score']:.2f} CO2={r['co2_estalviat_pct']:.2f}% cost={r['cost_estalviat_pct']:.2f}% infr={r['infraccions']} temps={time.time()-t:.1f}s ({1000*(time.time()-t)/len(val):.1f} ms/pas)")

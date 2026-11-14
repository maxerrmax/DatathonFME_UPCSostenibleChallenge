"""
Cerca de polítiques: optimitza paràmetres d'un agent amb scipy.differential_evolution
sobre la partició 'fit' i en guarda el resultat a agents/<nom>/parametres.json.
La validació es fa després amb run_all.py (dades no vistes).

Ús: python pipeline/tune.py rbc_tuned
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
sys.path.insert(0, str(Path(__file__).parent))
from data import load_hourly, split_val
from evaluate import AGENTS, load_agent, evaluate

SPACES = {
    "rbc_tuned": {"k_charge": (2, 10), "k_dis": (2, 10), "ratio": (0.3, 1.0), "margin": (0.8, 1.0)},
    "mpc_lp_tuned": {"ratio_co2b_cb": (0.3, 1.0), "q_term": (0.2, 0.9), "term_scale": (0.7, 1.1)},
}


def main(name, maxiter=4, popsize=5):
    fit, _ = split_val(load_hourly())
    space = SPACES[name]; keys = list(space)
    folder = AGENTS / name
    def f(x):
        p = dict(zip(keys, map(float, x)))
        s = evaluate(load_agent(folder, params=p), fit)["score_web"]
        print(round(s, 3), {k: round(v, 3) for k, v in p.items()}, flush=True)
        return -s
    r = differential_evolution(f, [space[k] for k in keys], maxiter=maxiter, popsize=popsize,
                               seed=0, polish=False, workers=1)
    best = dict(zip(keys, map(float, r.x)))
    (folder / "parametres.json").write_text(json.dumps(best, indent=1))
    print("MILLOR", -r.fun, best)


if __name__ == "__main__":
    main(sys.argv[1])
